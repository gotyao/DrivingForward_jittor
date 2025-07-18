# Copyright 2020 Toyota Research Institute.  All rights reserved.

import yacs
import numpy as np
import jittor as jt

########################################################################################################################

def is_numpy(data):
    """Checks if data is a numpy array."""
    return isinstance(data, np.ndarray)

def is_Var(data):
    """Checks if data is a jittor Var."""
    return jt.is_var(data)
# def is_tensor(data):
#     """Checks if data is a torch tensor."""
#     return type(data) == torch.Tensor

def is_tuple(data):
    """Checks if data is a tuple."""
    return isinstance(data, tuple)

def is_list(data):
    """Checks if data is a list."""
    return isinstance(data, list)

def is_dict(data):
    """Checks if data is a dictionary."""
    return isinstance(data, dict)

def is_str(data):
    """Checks if data is a string."""
    return isinstance(data, str)

def is_int(data):
    """Checks if data is an integer."""
    return isinstance(data, int)

def is_seq(data):
    """Checks if data is a list or tuple."""
    return is_tuple(data) or is_list(data)

def is_cfg(data):
    """Checks if data is a configuration node"""
    return type(data) == yacs.config.CfgNode

########################################################################################################################