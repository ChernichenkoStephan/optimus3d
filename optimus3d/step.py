import numpy as np
from collections.abc import Callable

from optimus3d.types import *
from optimus3d.functions import *
import optimus3d.utils as utils

def const_stepper(step_size: Numeric):
    def step(**kwargs) -> tuple[Numeric, Numeric]:
        return step_size, 0
    return step

def _armijo_rule(
        f: function,
        x_k: np.ndarray,
        a_k: Numeric,
        d_k: np.ndarray,
        epsilon: Numeric,
        diff_val: np.ndarray,
    ) -> bool:
    return f(x_k + a_k * d_k) <= f(x_k) + epsilon * a_k * np.dot(diff_val, d_k)

def armijo_stepper(alpha: Numeric, epsilon: Numeric):
    def step(**kwargs) -> tuple[Numeric, Numeric]:
        args = utils._unpack('func', 'x_k', 'diff_val', 'd_k', **kwargs)
        func, x_k, diff_val, d_k = args
        k = 0
        a_k = alpha
        while not _armijo_rule(func, x_k, a_k, d_k, epsilon, diff_val):
            a_k *= epsilon
            k += 1
        return a_k, k
    return step

def iteration_stepper(iteration_step_func: Callable = None):
    def default_func(iteration):
        return 2 / (iteration + 2)

    if iteration_step_func is None:
        iteration_step_func = default_func

    def step(**kwargs) -> tuple[Numeric, Numeric]:
        iteration, = utils._unpack('iteration', **kwargs)
        return iteration_step_func(iteration), 0
    return step
