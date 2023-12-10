import numpy as np

from optimus3d.types import *
import optimus3d.utils as utils

def union_stopper(*args):
    def stopper(**kwargs):
        return any((stopper_i(**kwargs) for stopper_i in args))
    return stopper

def iterations_stopper(num: int):
    def stopper(**kwargs):
        iteration, = utils._unpack('iteration', **kwargs)
        return iteration == num
    return stopper

def grad_norm_stopper(epsilon: Numeric):
    def stopper(**kwargs):
        diff_val, = utils._unpack('diff_val', **kwargs)
        return np.linalg.norm(diff_val) <= epsilon
    return stopper

def step_norm_stopper(epsilon: Numeric):
    def stopper(**kwargs):
        trajectory, = utils._unpack('trajectory', **kwargs)
        if len(trajectory) == 1:
            return False
        trajectory = np.matrix(trajectory)[:, [0, 1]]
        x_k, x_k_prew = trajectory[-2:]
        return np.linalg.norm(x_k - x_k_prew) <= epsilon
    return stopper
