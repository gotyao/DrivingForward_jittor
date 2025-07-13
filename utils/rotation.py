import jittor as jt
from scipy.linalg import expm
import numpy as np

# pytorch3d

def axis_angle_to_matrix(axis_angle: jt.Var) -> jt.Var:
    shape = axis_angle.shape

    angles = jt.norm(axis_angle, p=2, dim=-1, keepdims=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = jt.zeros(shape[:-1])
    cross_product_matrix = jt.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = jt.init.eye(3)
    angles_sqrd = angles * angles
    angles_sqrd = jt.where(angles_sqrd == 0, 1, angles_sqrd)
    ret = (
        identity.expand(cross_product_matrix.shape)
        + jt.where(angles == 0, 1, jt.sin(angles) / angles) * cross_product_matrix
        + ((1 - jt.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )

    return ret

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool
) -> jt.Var:
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return jt.atan2(data[..., i1], data[..., i2])
    return jt.atan2(-data[..., i2], data[..., i1])

def matrix_to_euler_angles(matrix: jt.Var) -> jt.Var:
    central_angle = jt.asin(
        matrix[..., 0, 2] * (-1.0 if 0 - 2 in [-1, 2] else 1.0)
    )

    o = (
        _angle_from_tan(
            'X', 'Y', matrix[..., 2], False
        ),
        central_angle,
        _angle_from_tan(
            'Z', 'Y', matrix[..., 0, :], True
        ),
    )
    return jt.stack(o, -1)

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
    xyz = jt.normalize(xyz, p=2.0, dim=-1)
    xyz = xyz.clamp(-1, 1)
    beta = np.arccos(xyz[..., 1].numpy())
    alpha = np.arctan2(xyz[..., 0].numpy(), xyz[..., 2].numpy())
    return alpha, beta

def matrix_to_angles(R):
    x = R @ jt.float32([0.0, 1.0, 0.0])
    a, b = _xyz_to_ang(x)
    R = _ang_to_mat(jt.float32(a), jt.float32(b), jt.zeros_like(jt.float32(a))).transpose(-1, -2) @ R
    c = np.arctan2(R[..., 0, 2].numpy(), R[..., 0, 0].numpy())

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

if __name__ == '__main__':
    R = jt.Var([[[[[[[0.01026021, 0.00843345, 0.9999118 ],
                     [-0.99987257, 0.01231626, 0.01015593],
                     [-0.01222952, -0.9998886, 0.00855874]]]]]]])
    R1 = jt.Var([[[[[[[0.82254606, 0.00647832, 0.56866163],
                      [-0.56868434, 0.01643407, 0.8223917],
                      [-0.00401771, -0.99984396, 0.01720189]]]]]]])

    x = R @ jt.Var([0.0,1.0,0.0])
    print(x)

    print(matrix_to_angles(R))
    print(matrix_to_angles(R1))

# if __name__ == '__main__':
#     for i in range(6):
#         a = wigner_D(i, [[[[[3.1321907]]]]], [[[[[1.5584798]]]]], [[[[[-1.5799736]]]]])
#         b = jt.Var(a)
