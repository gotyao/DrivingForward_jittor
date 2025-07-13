# from jaxtyping import Float
from math import isqrt
import math
import numpy as np
import jittor as jt
from utils import matrix_to_angles, wigner_D

def depth2pc(depth, extrinsic, intrinsic):
    B, C, H, W = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = jt.meshgrid(jt.linspace(0.5, H-0.5, H), jt.linspace(0.5, W-0.5, W))
    pts_2d = jt.stack([x, y, jt.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B H W 3

    pts_2d[..., 2] = depth
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = jt.contrib.concat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)

    rot_t = rot.permute(0, 2, 1)
    pts = jt.nn.bmm(rot_t, pts_2d) - jt.nn.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)

def rotate_sh(
    sh_coefficients: jt.Var,  # *#batch n
    rotations: jt.Var,  # *#batch 3 3
) -> jt.Var:  # *batch n
    sh_coefficients = sh_coefficients.float32()
    rotations = rotations.float32()

    *_, n = sh_coefficients.shape
    alpha, beta, gamma = matrix_to_angles(rotations)
    result = []
    # TODO: disable cpu switch once jt.matmul is repaired
    jt.flags.use_cuda = 0
    for degree in range(isqrt(n)):
        sh_rotations = wigner_D(degree, alpha, beta, gamma)
        sh_rotations = jt.float32(sh_rotations)
        b = sh_coefficients[..., degree ** 2: (degree + 1) ** 2]
        b_temp = b.unsqueeze(-1)

        s = sh_rotations @ b_temp
        sh_rotated = s.squeeze(-1)
        result.append(sh_rotated)

    jt.flags.use_cuda = 1
    return jt.contrib.concat(result, dim=-1)

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getProjectionMatrix(znear, zfar, K, h, w):
    near_fx = znear / K[0, 0]
    near_fy = znear / K[1, 1]
    left = - (w - K[0, 2]) * near_fx
    right = K[0, 2] * near_fx
    bottom = (K[1, 2] - h) * near_fy
    top = K[1, 2] * near_fy

    P = jt.zeros((4, 4))
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)