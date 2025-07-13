import jittor as jt
from scipy.linalg import expm
import numpy as np

def axis_angle_to_matrix(axis_angle: jt.Var) -> jt.Var:
    alpha, beta, gamma = jt.split(axis_angle, 1, dim=-1)

    ca = jt.cos(alpha)
    cb = jt.cos(beta)
    cc = jt.cos(gamma)
    sa = jt.sin(alpha)
    sb = jt.sin(beta)
    sc = jt.sin(gamma)

    # 构造旋转矩阵的每一行
    row0 = jt.stack([cb*cc, -cb*sc, sb], dim=-1).squeeze(-2)
    row1 = jt.stack([ca*sc + sa*sb*cc, ca*cc - sa*sb*sc, -sa*cb], dim=-1).squeeze(-2)
    row2 = jt.stack([sa*sc - ca*sb*cc, sa*cc + ca*sb*sc, ca*cb], dim=-1).squeeze(-2)

    # 将三行堆叠成一个矩阵
    R = jt.stack([row0, row1, row2], dim=-2)

    return R

def matrix_to_euler_angles(R: jt.Var) -> jt.Var:
    eps = 1e-6

    R20 = R[..., 2, 0]
    R21 = R[..., 2, 1]
    R22 = R[..., 2, 2]
    R10 = R[..., 1, 0]
    R00 = R[..., 0, 0]
    R01 = R[..., 0, 1]
    R02 = R[..., 0, 2]

    beta = jt.arcsin(-R20)
    cos_beta = jt.cos(beta)

    nonzero = (jt.abs(cos_beta) > eps).astype(R.dtype)

    alpha = jt.where(
        nonzero,
        jt.arctan2(R21, R22),
        jt.Var(0.0)
    )

    gamma = jt.where(
        nonzero,
        jt.arctan2(R10, R00),
        jt.arctan2(-R01, R02)
    )

    return jt.stack([alpha, beta, gamma], dim=-1)

# e3nn

def _mat_x(a):
    c, s = a.cos(), a.sin()
    o, z = jt.ones_like(a), jt.zeros_like(a)
    return jt.stack(
        [jt.stack([o,z,z],-1),
            jt.stack([z,c,-s],-1),
            jt.stack([z,s,c],-1)
           ],
        -2
    )

def _mat_y(a):
    c, s = a.cos(), a.sin()
    o, z = jt.ones_like(a), jt.zeros_like(a)
    return jt.stack(
        [jt.stack([c,z,s],-1),
            jt.stack([z,o,z],-1),
            jt.stack([-s,z,c],-1)
         ],
        -2
    )

def _ang_to_mat(a, b, g):
    return _mat_y(a) @ _mat_x(b) @ _mat_y(g)

def _xyz_to_ang(xyz):
    n_xyz = jt.normalize(xyz, p=2.0, dim=-1)
    c_xyz = n_xyz.clamp(-1, 1).float64()
    beta = jt.acos(c_xyz[..., 1])
    alpha = jt.atan2(c_xyz[..., 0], c_xyz[..., 2])
    return alpha, beta

def matrix_to_angles(R):
    y_vec = R[..., :, 1]
    a, b = _xyz_to_ang(y_vec)

    R_ab0 = _ang_to_mat(a, b, jt.zeros_like(a))
    R_prime = R_ab0.transpose(-1, -2) @ R

    c = jt.atan2(R_prime[..., 0, 2], R_prime[..., 0, 0])

    return a, b, c

def su2_generators(j: int) -> np.ndarray:
    m = np.arange(-j, j)
    raising = np.diag(-np.sqrt(j * (j + 1) - m * (m + 1)), k=-1)

    m = np.arange(-j + 1, j + 1)
    lowering = np.diag(np.sqrt(j * (j + 1) - m * (m - 1)), k=1)

    m = np.arange(-j, j + 1)
    return np.stack([0.5 * (raising + lowering),
                            np.diag(1j * m),
                            -0.5j * (raising - lowering)], axis=0)

def change_basis_real_to_complex(l: int, dtype=np.complex128) -> np.ndarray:
    size = 2 * l + 1
    q = np.zeros((size, size), dtype=np.complex128)

    for m in range(-l, 0):
        row = l + m
        q[row, l + abs(m)] = 1 / np.sqrt(2)
        q[row, l - abs(m)] = -1j / np.sqrt(2)

    q[l, l] = 1

    for m in range(1, l + 1):
        row = l + m
        factor = (-1) ** m
        q[row, l + abs(m)] = factor / np.sqrt(2)
        q[row, l - abs(m)] = 1j * factor / np.sqrt(2)

    q *= (-1j) ** l

    if dtype == np.float32:
        q = q.astype(np.complex64)
    else:
        q = q.astype(np.complex128)
    return q

def so3_generators(l: int) -> np.ndarray:
    X = su2_generators(l)
    Q = change_basis_real_to_complex(l)
    Q_conj_T = np.conj(Q.T) @ X @ Q

    return Q_conj_T.real.copy()

def wigner_D(l: int, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    alpha = np.mod(alpha, 2 * np.pi)
    beta = np.mod(beta, 2 * np.pi)
    gamma = np.mod(gamma, 2 * np.pi)

    X = so3_generators(l)

    def apply_exp(arr, generator):
        shape = arr.shape + (2 * l + 1, 2 * l + 1)
        result = np.empty(shape, dtype=np.float64)
        it = np.nditer(arr, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            angle = arr[idx]
            result[idx] = expm(angle * generator)
        return result

    exp_alpha = apply_exp(alpha, X[1])
    exp_beta = apply_exp(beta, X[0])
    exp_gamma = apply_exp(gamma, X[1])

    re = exp_alpha @ exp_beta @ exp_gamma
    re = re.real
    return re

# if __name__ == '__main__':
#     R = jt.Var([[[[[[[0.01026021, 0.00843345, 0.9999118 ],
#                      [-0.99987257, 0.01231626, 0.01015593],
#                      [-0.01222952, -0.9998886, 0.00855874]]]]]]])
#     R1 = jt.Var([[[[[[[0.82254606, 0.00647832, 0.56866163],
#                       [-0.56868434, 0.01643407, 0.8223917],
#                       [-0.00401771, -0.99984396, 0.01720189]]]]]]])
#
#     print(matrix_to_angles(R))
#     print(matrix_to_angles(R1))


if __name__ == '__main__':
    print(wigner_D(1, np.array([[[[[3.1351]]]]]), np.array([[[[[1.5544]]]]]), np.array([[[[[-2.5366]]]]])))
    print(wigner_D(1, np.array([[[[[3.1332]]]]]), np.array([[[[[1.5585]]]]]), np.array([[[[[-1.5810]]]]])))


