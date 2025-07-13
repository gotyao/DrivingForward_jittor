import jittor as jt
from jittor import nn


def pack_cam_feat(x):
    if isinstance(x, dict):
        for k, v in x.items():
            b, n_cam = v.shape[:2]
            x[k] = v.view(b*n_cam, *v.shape[2:])
        return x
    else:
        b, n_cam = x.shape[:2]
        x = x.view(b*n_cam, *x.shape[2:])
    return x


def unpack_cam_feat(x, b, n_cam):
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = v.view(b, n_cam, *v.shape[1:])
        return x
    else:
        x = x.view(b, n_cam, *x.shape[1:])
    return x


def upsample(x):
    return nn.interpolate(x, scale_factor=2, mode='nearest')


class ReflectConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, padding=0, bias=False, padding_mode='reflect'):
        super().__init__()
        assert padding_mode == 'reflect'

        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(int(in_planes), int(out_planes), kernel_size=int(kernel_size),
            stride=int(stride), dilation=int(dilation), padding=int(0), bias=bias)

    def execute(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

# enc_feat_dim 896
# self.fusion_feat_in_dim 256
# kernel_size=1
def conv2d(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, nonlin = 'LRU', padding_mode = 'reflect', norm = False):
    if nonlin== 'LRU':
        act = nn.LeakyReLU(0.1)
    elif nonlin == 'ELU':
        act = nn.ELU()
    else:
        act = nn.Identity()
        
    if norm:
        conv = ReflectConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=False, padding_mode=padding_mode)        
        bnorm = nn.BatchNorm2d(out_planes)
    else:
        conv = ReflectConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)        
        bnorm = nn.Identity()
    return nn.Sequential(conv, bnorm, act)

class ReflectConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, padding=0, bias=False, padding_mode='reflect'):
        super().__init__()
        assert padding_mode == 'reflect'
        self.padding = (padding, padding)
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                            stride=stride, dilation=dilation, padding=0, bias=bias)

    def execute(self, x):
        pad_left, pad_right = self.padding
        x = nn.pad(x, [pad_left, pad_right], mode='reflect')
        x = self.conv(x)
        return x

def conv1d(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, nonlin='LRU', padding_mode='reflect', norm = False):
    if nonlin== 'LRU':
        act = nn.LeakyReLU(0.1)
    elif nonlin == 'ELU':
        act = nn.ELU()
    else:
        act = nn.Identity()
        
    if norm:
        conv = ReflectConv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=False, padding_mode=padding_mode)        
        bnorm = nn.BatchNorm1d(out_planes)
    else:
        conv = ReflectConv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)        
        bnorm = nn.Identity()        
        
    return nn.Sequential(conv, bnorm, act)        