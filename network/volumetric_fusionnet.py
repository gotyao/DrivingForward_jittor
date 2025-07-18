import jittor as jt
from jittor import nn
from utils import axis_angle_to_matrix

from .blocks import conv2d, conv1d, pack_cam_feat


class VFNet(nn.Module):
    """
    Surround-view fusion module that estimates a single 3D feature using surround-view images
    """
    def __init__(self, cfg, feat_in_dim, feat_out_dim, model='depth'):
        super(VFNet, self).__init__()
        self.read_config(cfg) 
        self.eps = 1e-8
        self.model = model
        # define the 3D voxel space(follows the DDAD extrinsic coordinate -- x: forward, y: left, z: up)
        # define voxel end range in accordance with voxel_str_p, voxel_size, voxel_unit_size
        self.voxel_end_p = [self.voxel_str_p[i] + self.voxel_unit_size[i] * (self.voxel_size[i] - 1) for i in range(3)]
            
        # define a voxel space, [1, 3, z, y, x], each voxel contains its 3D position
        voxel_grid = self.create_voxel_grid(self.voxel_str_p, self.voxel_end_p, self.voxel_size)        
        b, _, self.z_dim, self.y_dim, self.x_dim = voxel_grid.size()
        self.n_voxels = self.z_dim * self.y_dim * self.x_dim
        ones = jt.ones((self.batch_size, 1, self.n_voxels))
        self.voxel_pts = jt.contrib.concat([voxel_grid.view(b, 3, self.n_voxels), ones], dim=1)

        # define grids in pixel space
        self.img_h = self.height // (2 ** (self.fusion_level+1))
        self.img_w = self.width // (2 ** (self.fusion_level+1))
        self.num_pix = self.img_h * self.img_w
        self.pixel_grid = self.create_pixel_grid(self.batch_size, self.img_h, self.img_w)
        self.pixel_ones = jt.ones((self.batch_size, 1, self.proj_d_bins, self.num_pix))
        
        # define a depth grid for projection
        depth_bins = jt.linspace(self.proj_d_str, self.proj_d_end, self.proj_d_bins)
        self.depth_grid = self.create_depth_grid(self.batch_size, self.num_pix, self.proj_d_bins, depth_bins)
        
        # depth fusion(process overlap and non-overlap regions)
        if model == 'depth':
            # voxel - preprocessing layer
            self.v_dim_o = [(feat_in_dim + 1) * 2] + self.voxel_pre_dim
            self.v_dim_no = [feat_in_dim + 1] + self.voxel_pre_dim
            
            self.conv_overlap = conv1d(self.v_dim_o[0], self.v_dim_o[1], kernel_size=1) 
            self.conv_non_overlap = conv1d(self.v_dim_no[0], self.v_dim_no[1], kernel_size=1) 

            encoder_dims = self.proj_d_bins * self.v_dim_o[-1]
            stride = 1
            
        else:
            encoder_dims = (feat_in_dim + 1)*self.z_dim
            stride = 2
            
        # channel dimension reduction
        self.reduce_dim = nn.Sequential(*conv2d(encoder_dims, 256, kernel_size=3, stride = stride).children(),
                                        *conv2d(256, feat_out_dim, kernel_size=3, stride = stride).children())          
            
    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
    
    def create_voxel_grid(self, str_p, end_p, v_size):
        grids = [jt.linspace(str_p[i], end_p[i], v_size[i]) for i in range(3)]

        x_dim, y_dim, z_dim = v_size
        grids[0] = grids[0].view(1, 1, 1, 1, x_dim)
        grids[1] = grids[1].view(1, 1, 1, y_dim, 1)
        grids[2] = grids[2].view(1, 1, z_dim, 1, 1)
        
        grids = [grid.expand(self.batch_size, 1, z_dim, y_dim, x_dim) for grid in grids]
        return jt.contrib.concat(grids, 1)
    
    def create_pixel_grid(self, batch_size, height, width):
        y, x = jt.meshgrid(jt.arange(height), jt.arange(width))
        grid_xy = (x, y)
        pix_coords = jt.stack(grid_xy).unsqueeze(0).view(1, 2, height * width)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        ones = jt.ones((batch_size, 1, height * width))
        pix_coords = jt.contrib.concat([pix_coords, ones], 1)
        return pix_coords

    def create_depth_grid(self, batch_size, n_pixels, n_depth_bins, depth_bins):
        depth_layers = []
        for d in depth_bins:
            depth_layer = jt.ones((1, n_pixels)) * d
            depth_layers.append(depth_layer)
        depth_layers = jt.contrib.concat(depth_layers, dim=0).view(1, 1, n_depth_bins, n_pixels)
        depth_layers = depth_layers.expand(batch_size, 3, n_depth_bins, n_pixels)
        return depth_layers

    def type_check(self, sample_tensor):
        d_dtype = sample_tensor.dtype
        if (self.voxel_pts.dtype != d_dtype):
            self.voxel_pts = self.voxel_pts.to(dtype=d_dtype)
            self.pixel_grid = self.pixel_grid.to(dtype=d_dtype)
            self.depth_grid = self.depth_grid.to(dtype=d_dtype)
            self.pixel_ones = self.pixel_ones.to(dtype=d_dtype)

    def backproject_into_voxel(self, feats_agg, input_mask, intrinsics, extrinsics_inv):
        voxel_feat_list = []
        voxel_mask_list = []
        
        for cam in range(self.num_cams):
            feats_img = feats_agg[:, cam, ...]
            _, _, h_dim, w_dim = feats_img.size()
        
            mask_img = input_mask[:, cam, ...]            
            mask_img = nn.interpolate(mask_img, [h_dim, w_dim], mode='bilinear', align_corners=True)
            
            # 3D points in the voxel grid -> 3D points referenced at each view. [b, 3, n_voxels]
            ext_inv_mat = extrinsics_inv[:, cam, :3, :]
            v_pts_local = jt.matmul(ext_inv_mat, self.voxel_pts)

            # calculate pixel coordinate that each point are projected in the image. [b, n_voxels, 1, 2]
            K_mat = intrinsics[:, cam, :, :]
            pix_coords = self.calculate_sample_pixel_coords(K_mat, v_pts_local, w_dim, h_dim)

            # compute validity mask. [b, 1, n_voxels]
            valid_mask = self.calculate_valid_mask(mask_img, pix_coords, v_pts_local)
            
            # retrieve each per-pixel feature. [b, feat_dim, n_voxels, 1]
            feat_warped = nn.grid_sample(feats_img, pix_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
            # concatenate relative depth as the feature. [b, feat_dim + 1, n_voxels]
            feat_warped = jt.contrib.concat([feat_warped.squeeze(-1), v_pts_local[:, 2:3, :]/(self.voxel_size[0])], dim=1)
            feat_warped = feat_warped * valid_mask.float()
            
            voxel_feat_list.append(feat_warped) # [[batch_size, feat_dim, n_voxels], ...]
            voxel_mask_list.append(valid_mask)
        
        # compute overlap region
        voxel_mask_count = jt.sum(jt.contrib.concat(voxel_mask_list, dim=1), dim=1, keepdims=True)
        
        if self.model == 'depth':
            # discriminatively process overlap and non_overlap regions using different MLPs
            voxel_non_overlap = self.preprocess_non_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
            voxel_overlap = self.preprocess_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
            voxel_feat = voxel_non_overlap + voxel_overlap # [batch_size, feat_dim, n_voxels]
            
        elif self.model == 'pose':
            voxel_feat = jt.sum(jt.stack(voxel_feat_list, dim=1), dim=1, keepdims=False)
            voxel_feat = voxel_feat/(voxel_mask_count+1e-7)
            
        return voxel_feat

    def calculate_sample_pixel_coords(self, K, v_pts, w_dim, h_dim):
        """
        This function calculates pixel coords for each point([batch, n_voxels, 1, 2]) to sample the per-pixel feature.
        """        
        cam_points = jt.matmul(K[:, :3, :3], v_pts) # [batch_size, 3, n_voxels]
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)

        d = jt.all(jt.isfinite(pix_coords))
        if not d:
            pix_coords = jt.clamp(pix_coords, min_v=-w_dim*2, max_v=w_dim*2)

        pix_coords = pix_coords.view(self.batch_size, 2, self.n_voxels, 1) #[batch_size, 2, n_voxels] -> [batch_size, 2, n_voxels, 1]
        pix_coords = pix_coords.permute(0, 2, 3, 1) # [batch_size, 2, n_voxels, 1] -> [batch_size, n_voxels, 1, 2]
        pix_coords[:, :, :, 0] = pix_coords[:, :, :, 0] / (w_dim - 1)
        pix_coords[:, :, :, 1] = pix_coords[:, :, :, 1] / (h_dim - 1)
        pix_coords = (pix_coords - 0.5) * 2 # [batch_size, n_voxels, 1, 2]
        return pix_coords

    def calculate_valid_mask(self, mask_img, pix_coords, v_pts_local):
        """
        This function creates valid mask in voxel coordinate by projecting self-occlusion mask to 3D voxel coords. 
        """
        # compute validity mask, [b, 1, n_voxels, 1]
        mask_selfocc = (nn.grid_sample(mask_img, pix_coords, mode='nearest', padding_mode='zeros', align_corners=True) > 0.5)
        # discard points behind the camera, [b, 1, n_voxels]
        mask_depth = (v_pts_local[:, 2:3, :] > 0) 
        # compute validity mask, [b, 1, n_voxels, 1]
        pix_coords_mask = pix_coords.permute(0, 3, 1, 2)
        mask_oob = jt.logical_not(jt.logical_or(pix_coords_mask > 1, pix_coords_mask < -1).sum(dim=1, keepdims=True) > 0)
        valid_mask = mask_selfocc.squeeze(-1) * mask_depth * mask_oob.squeeze(-1)
        return valid_mask
    
    def preprocess_non_overlap(self, voxel_feat_list, voxel_mask_list, voxel_mask_count):
        """
        This function applies 1x1 convolutions to features from non-overlapping features.
        """
        non_overlap_mask = (voxel_mask_count == 1)
        voxel = sum(voxel_feat_list)
        voxel = voxel * non_overlap_mask.float()

        for conv_no in self.conv_non_overlap:
            voxel = conv_no(voxel)
        return voxel * non_overlap_mask.float()

    def preprocess_overlap(self, voxel_feat_list, voxel_mask_list, voxel_mask_count):
        """
        This function applies 1x1 convolutions on overlapping features.
        Camera configuration [0,1,2] or [0,1,2,3,4,5]:
                        3 1
            rear cam <- 5   0 -> front cam
                        4 2
        """
        overlap_mask = (voxel_mask_count == 2)
        if self.num_cams == 3:
            feat1 = voxel_feat_list[0]
            feat2 = voxel_feat_list[1] + voxel_feat_list[2]
        elif self.num_cams == 6:
            feat1 = voxel_feat_list[0] + voxel_feat_list[3] + voxel_feat_list[4]
            feat2 = voxel_feat_list[1] + voxel_feat_list[2] + voxel_feat_list[5]
        else:
            raise NotImplementedError
            
        voxel = jt.contrib.concat([feat1, feat2], dim=1)
        for conv_o in self.conv_overlap:
            voxel = conv_o(voxel)
        return voxel * overlap_mask.float()

    def project_voxel_into_image(self, voxel_feat, inv_K, extrinsics):
        """
        This function projects voxels into 2D image coordinate. 
        [b, feat_dim, n_voxels] -> [b, feat_dim, d, h, w]
        """        
        # define depth bin
        # [b, feat_dim, n_voxels] -> [b, feat_dim, d, h, w]
        b, feat_dim, _ = voxel_feat.size()
        voxel_feat = voxel_feat.view(b, feat_dim, self.z_dim, self.y_dim, self.x_dim) 
        
        proj_feats = []
        for cam in range(self.num_cams):
            # construct 3D point grid for each view
            cam_points = jt.matmul(inv_K[:, cam, :3, :3], self.pixel_grid)
            cam_points = self.depth_grid * cam_points.view(self.batch_size, 3, 1, self.num_pix)
            cam_points = jt.contrib.concat([cam_points, self.pixel_ones], dim=1) # [b, 4, n_depthbins, n_pixels]
            cam_points = cam_points.view(self.batch_size, 4, -1) # [b, 4, n_depthbins * n_pixels]
            
            # apply extrinsic: local 3D point -> global coordinate, [b, 3, n_depthbins * n_pixels]
            points = jt.matmul(extrinsics[:, cam, :3, :], cam_points)

            # 3D grid_sample [b, n_voxels, 3], value: (x, y, z) point
            grid = points.permute(0, 2, 1) 
            
            for i in range(3):
                v_length = self.voxel_end_p[i] - self.voxel_str_p[i]
                grid[:, :, i] = (grid[:, :, i] - self.voxel_str_p[i]) / v_length * 2. - 1.
                
            grid = grid.view(self.batch_size, self.proj_d_bins, self.img_h, self.img_w, 3)            
            proj_feat = nn.grid_sample(voxel_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            proj_feat = proj_feat.view(b, self.proj_d_bins * self.v_dim_o[-1], self.img_h, self.img_w)
            
            # conv, reduce dimension
            proj_feat = self.reduce_dim(proj_feat)
            proj_feats.append(proj_feat)
        return proj_feats

    def augment_extrinsics(self, ext):
        """
        This function augments depth estimation results using augmented extrinsics [batch, cam, 4, 4]  
        """
        with jt.no_grad():
            b, cam, _, _ = ext.size()
            ext_aug = ext.clone()

            # rotation augmentation
            angle = jt.rand(b, cam, 3)
            self.aug_angle = [15, 15, 40]
            for i in range(3):
                angle[:, :, i] = (angle[:, :, i] - 0.5) * self.aug_angle[i]
            angle_mat = axis_angle_to_matrix(angle) # 3x3
            tform_mat = jt.init.eye(4).repeat(b, cam, 1, 1)
            tform_mat[:, :, :3, :3] = angle_mat
            tform_mat = tform_mat.to(device=ext.device, dtype=ext.dtype)

            ext_aug = tform_mat @ ext_aug
        return ext_aug
    
    def execute(self, inputs, feats_agg):
        mask = inputs['mask']
        K = inputs['K', self.fusion_level+1]
        inv_K = inputs['inv_K', self.fusion_level+1]
        extrinsics = inputs['extrinsics']
        extrinsics_inv = inputs['extrinsics_inv']
        
        fusion_dict = {}
        for cam in range(self.num_cams):
            fusion_dict[('cam', cam)] = {}
        
        # device, dtype check, match dtype and device
        sample_tensor = feats_agg[0, 0, ...] # B, n_cam, c, h, w
        self.type_check(sample_tensor)
            
        # backproject each per-pixel feature into 3D space (or sample per-pixel features for each voxel)
        voxel_feat = self.backproject_into_voxel(feats_agg, mask, K, extrinsics_inv) # [batch_size, feat_dim, n_voxels]
            
        if self.model == 'depth':
            # for each pixel, collect voxel features -> output image feature     
            proj_feats = self.project_voxel_into_image(voxel_feat, inv_K, extrinsics)
            fusion_dict['proj_feat'] = pack_cam_feat(jt.stack(proj_feats, 1))
            return fusion_dict      
        
        elif self.model == 'pose':
            b, c, _ = voxel_feat.shape      
            voxel_feat = voxel_feat.view(b, c*self.z_dim, 
                                         self.y_dim, self.x_dim)            
            bev_feat= self.reduce_dim(voxel_feat)
            return bev_feat
