from jittor import nn

class BaseLoss(nn.Module):
    """
    Base class loss calculation
    """
    def __init__(self, cfg, rank):
        super().__init__()
        self.rank = rank
        self.init_weights(cfg)
        self.init_attrib(cfg) 

    def init_weights(self, cfg):
        for attr in cfg.keys(): 
            if attr == 'loss' or attr == 'training':
                for k, v in cfg[attr].items():
                    setattr(self, k, v)
                    
    def init_attrib(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                if k == 'scales' or k=='frame_ids' or k == 'novel_view_mode':
                    setattr(self, k, v)
        
    def get_logs(self, loss_dict, output, cam):
        """
        This function logs depth and pose information for monitoring training process. 
        """
        # log statistics
        depth_log = output[('depth', 0, 0)].clone().detach()
        loss_dict['depth/mean'] = depth_log.mean().item()
        loss_dict['depth/max'] = depth_log.max().item()
        loss_dict['depth/min'] = depth_log.min().item()

        if cam == 0:
            pose_t = output[('cam_T_cam', 0, -1)].clone().detach()
            loss_dict['pose/tx'] = pose_t[:, 0, 3].abs().mean().item()
            loss_dict['pose/ty'] = pose_t[:, 1, 3].abs().mean().item()
            loss_dict['pose/tz'] = pose_t[:, 2, 3].abs().mean().item()
            
        return loss_dict
    
    def execute(self, *args, **kwargs):
        raise NotImplementedError('Not implemented for BaseLoss')   