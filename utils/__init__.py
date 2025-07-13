from .logger import Logger
from .misc import get_config
from .rotation import axis_angle_to_matrix, matrix_to_euler_angles, matrix_to_angles, wigner_D

__all__ = ['Logger', 'get_config', 'axis_angle_to_matrix', 'matrix_to_euler_angles', 'matrix_to_angles', 'wigner_D']


import sys


_LIBS = ['./external/packnet_sfm', './external/dgp', './external/monodepth2']    

def setup_env():       
    if not _LIBS[0] in sys.path:        
        for lib in _LIBS:
            sys.path.append(lib)

setup_env()