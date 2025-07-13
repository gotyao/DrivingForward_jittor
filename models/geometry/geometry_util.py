import numpy as np
import jittor as jt
from jittor import nn
from utils import axis_angle_to_matrix

        
def vec_to_matrix(rot_angle, trans_vec, invert=False):
    """
    This function transforms rotation angle and translation vector into 4x4 matrix.
    """
    # initialize matrices
    b, _, _ = rot_angle.shape
    R_mat = jt.init.eye(4).repeat([b, 1, 1])
    T_mat = jt.init.eye(4).repeat([b, 1, 1])

    R_mat[:, :3, :3] = axis_angle_to_matrix(rot_angle).squeeze(1)
    t_vec = trans_vec.clone().contiguous().view(-1, 3, 1)

    if invert == True:
        R_mat = R_mat.transpose(1,2)
        t_vec = -1 * t_vec

    T_mat[:, :3,  3:] = t_vec

    if invert == True:
        P_mat = jt.nn.matmul(R_mat, T_mat)
    else :
        P_mat = jt.nn.matmul(T_mat, R_mat)
    return P_mat


class Projection(nn.Module):
    """
    This class computes projection and reprojection function. 
    """
    def __init__(self, batch_size, height, width, device):
        super().__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        
        # initialize img point grid
        img_points = np.meshgrid(range(width), range(height), indexing='xy')
        img_points = jt.Var(np.stack(img_points, 0)).float32()
        img_points = jt.stack([img_points[0].view(-1), img_points[1].view(-1)], 0).repeat(batch_size, 1, 1)
        img_points = img_points
        
        self.to_homo = jt.ones([batch_size, 1, width*height]).to(device)
        self.homo_points = jt.contrib.concat([img_points, self.to_homo], 1)

    def backproject(self, invK, depth):
        """
        This function back-projects 2D image points to 3D.
        """
        depth = depth.view(self.batch_size, 1, -1)

        points3D = jt.matmul(invK[:, :3, :3], self.homo_points)
        points3D = depth*points3D
        return jt.contrib.concat([points3D, self.to_homo], 1)
    
    def reproject(self, K, points3D, T):
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        """
        # project points 
        points2D = (K @ T)[:,:3, :] @ points3D

        # normalize projected points for grid sample function
        norm_points2D = points2D[:, :2, :]/(points2D[:, 2:, :] + 1e-7)
        norm_points2D = norm_points2D.view(self.batch_size, 2, self.height, self.width)
        norm_points2D = norm_points2D.permute(0, 2, 3, 1)

        norm_points2D[..., 0 ] /= self.width - 1
        norm_points2D[..., 1 ] /= self.height - 1
        norm_points2D = (norm_points2D-0.5)*2
        return norm_points2D        

    def execute(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)
        pix_coords = self.reproject(rp_K, cam_points, T)
        return pix_coords

if __name__ == '__main__':
    rot_angle_input = jt.array([
        [[0.0, 0.0, 1.0]],
        [[1.5, 0.0, 0.0]]
    ])
    trans_vec_input = jt.array([
        [1.0, 2.0, 3.0],
        [-1.0, 0.0, 0.5]
    ])
    print(vec_to_matrix(rot_angle_input, trans_vec_input, True))