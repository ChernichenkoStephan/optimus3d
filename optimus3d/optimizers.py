from optimus3d.types import *
from optimus3d.functions import *
import optimus3d.utils as utils
from numpy.linalg import inv

#  optimization

class Optimizer:
    def __init__(
            self, 
            stopper: Callable, 
            verbose=False,
            ):
        self._print = utils._optional_printer(verbose)
        self._stopper = stopper

    def _diffx(self, func: function, x: np.ndarray) -> np.ndarray:
        diff_func = func.diff()
        self._print(f'{diff_func=}')

        diff_val = np.array([diff(x) for diff in diff_func])
        self._print(f'{diff_val=}')

        return diff_func, diff_val

    def _hessianx(self, func: function, x: np.ndarray) -> np.ndarray:
        hessian_func = func.hessian()
        self._print(f'{hessian_func=}')

        hessian_val = np.array([[diff(x) for diff in row] for row in hessian_func])
        self._print(f'{hessian_val=}')

        return hessian_func,hessian_val 
    
    def _prep(
            self,
            func: function,
            x_0: np.ndarray,
            ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, list]:
        k = 0
        x_k = x_0
        if x_k is None:
            x_k = mp.array([0, 0])
        self._print(f'{x_k=}')

        diff_val = np.array([np.inf, np.inf])
        self._print(f'{diff_val=}')

        f_start = np.append(x_0, func(x_0))
        self._print(f'{f_start=}')

        trajectory = [f_start,]
        self._print('-----------------------------------------')
        return k, x_k, diff_val, f_start, trajectory

    def _after_step(
            self,
            iteration: int,
            func: function,
            x_k: np.ndarray,
            trajectory: list,
            ):
        f_x_k = func(x_k)
        trajectory.append(np.append(x_k, f_x_k))
        self._print(f'x_k+1={x_k} k = {iteration}')
        self._print('-----------------------------------------')


class UnconstrainedOptimizer(Optimizer):
    def __init__(
            self, 
            stopper: Callable, 
            verbose=False,
            ):
        super().__init__(stopper, verbose)

    def min(
            self,
            func: function,
            x_0: np.ndarray = None,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k, x_k, diff_val, f_start, trajectory = self._prep(func, x_0)
        while not self._stopper(
                iteration=k,
                diff_val=diff_val,
                trajectory=trajectory,
                ):
            k += 1
            self._print(f'{x_k=}')
            x_k, diff_val, hessian_val = self._step(func, x_k)
            self._after_step(k,func, x_k, trajectory)

        x_min = x_k
        f_min = np.append(x_min, func(x_min))
        return x_min, f_min, np.array(trajectory)


class IntervalConstrainedOptimizer(Optimizer):
    def __init__(
            self, 
            stopper: Callable, 
            verbose=False,
            ):
        super().__init__(stopper, verbose)

    def min(
            self,
            func: function,
            intervals: tuple[tuple, tuple],
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class FunctionConstrainedOptimizer(Optimizer):
    def __init__(
            self, 
            stopper: Callable, 
            verbose=False,
            ):
        super().__init__(stopper, verbose)

    def min(
            self,
            func: function,
            constrain: function,
            x_0: np.ndarray = None,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class GradientOptimizer(UnconstrainedOptimizer):
    def __init__(
            self, 
            stopper: Callable, 
            stepper: Callable, 
            verbose=False,
            ):
        self._stepper = stepper 
        super().__init__(stopper, verbose)

    def _step(
            self,
            func: function,
            x_k: Sequence | np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        diff_func, diff_val = self._diffx(func, x_k)

        d_k = -1 * diff_val 
        self._print(f'{d_k=}')

        a_k, iterations = self._stepper(
                    func=func, 
                    x_k=x_k, 
                    diff_val=diff_val, 
                    d_k=d_k,
                )
        self._print(f'got {a_k=} with {iterations=}')

        print(f'x_k + a_k * d_k = {x_k + a_k * d_k}')
        return x_k + a_k * d_k, diff_val, None


class NewtonOptimizer(UnconstrainedOptimizer):
    def __init__(
            self, 
            stopper: Callable, 
            verbose=False,
            ):
        super().__init__(stopper, verbose)

    def _step(
            self,
            func: function,
            x_k: Sequence | np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        diff_func, diff_val = self._diffx(func, x_k)
        hessian_func, hessian_val = self._hessianx(func, x_k)

        step_size = np.dot(inv(hessian_val), diff_val)
        self._print(f'{step_size=}')

        return x_k - step_size, diff_val, hessian_val


class LineOptimizer:
    def __init__(
            self, 
            epsilon=0.001, 
            max_iterations=10_000, 
            verbose=False,
            ):
        self._print = utils._optional_printer(verbose)
        self._max_iterations = max_iterations
        self._epsilon = epsilon 

    def min(self, func: function) -> tuple[Numeric, Numeric]:
        return 0, 0


class DichotomyLineOptimizer(LineOptimizer):
    def __init__(
            self, 
            interval=(-1000, 1000), 
            epsilon=0.001, 
            max_iterations=10_000, 
            verbose=False,
            ):
        self._interval = interval 
        if interval[0] >= interval[1]:
            raise Exception('bad interval')
        super().__init__(epsilon, max_iterations, verbose)

    def min(self, func: function) -> tuple[Numeric, Numeric]:
        k = 0
        left, right = self._interval
        self._print(f'{self._interval=}')

        epsilon = self._epsilon
        self._print(f'{epsilon=}')

        while abs(right - left) >= epsilon or k > self._max_iterations:
            k += 1
            self._print(f'{left}, {right}')

            left_third = left + (right - left) / 3
            right_third = right - (right - left) / 3

            if func(left_third) > func(right_third):
                left = left_third
            else:
                right = right_third

        return (left + right) / 2, k


class NCGradientOptimizer(UnconstrainedOptimizer):
    """
    Nonlinear conjugate gradient method
    """

    def __init__(
            self, 
            stopper: Callable, 
            line_optimizer: LineOptimizer = DichotomyLineOptimizer(),
            verbose=False,
            ):

        self.line_optimizer = line_optimizer
        self.betta_func = self._fletcher_reeves
        super().__init__(stopper, verbose)

    def _line_func(
        self,
        func: function,
        x_k: np.ndarray, 
        s_k: np.ndarray,
        ) -> Callable:
        def f(a):
            return func(x_k + a * s_k)
        return f

    def _fletcher_reeves(
            self,
            d_k: np.ndarray,
            d_k_prew: np.ndarray,
            s_k_prew: np.ndarray,
            ):
        numerator   = np.dot(d_k.T, d_k)
        denominator = np.dot(d_k_prew.T, d_k_prew)
        return numerator / denominator

    def _step(
            self,
            func: function,
            x_k: Sequence | np.ndarray,
            d_k_prew: np.ndarray,
            s_k_prew: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        diff_func, diff_val = self._diffx(func, x_k)
        d_k = -1 * diff_val
        self._print(f'{d_k=}')

        b_k = self.betta_func(d_k, d_k_prew, None)
        self._print(f'{b_k=}')

        s_k = d_k + b_k * s_k_prew
        self._print(f'{s_k=}')

        line_func = self._line_func(func, x_k, s_k)
        a_k, iterations = self.line_optimizer.min(line_func)
        self._print(f'got {a_k=} with {iterations=}')

        x_k = x_k + a_k * s_k
        return x_k, diff_val, None, d_k, s_k

    def min(
            self,
            func: function,
            x_0: np.ndarray = None,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k, x_k, diff_val, f_start, trajectory = self._prep(func, x_0)

        diff_func, diff_val = self._diffx(func, x_k)
        d_0 = -1 * diff_val
        self._print(f'{d_0=}')

        line_func = self._line_func(func, x_k, d_0)
        a_0, iterations = self.line_optimizer.min(line_func)
        self._print(f'got {a_0=} with {iterations=}')

        x_k = x_0 + a_0 * d_0
        self._print(f'x_0={x_k}')

        s_k = d_k = d_0
        self._print('-----------------------------------------')
        while not self._stopper(
                iteration=k,
                diff_val=diff_val,
                trajectory=trajectory,
                ):
            k += 1
            x_k, diff_val, _, d_k, s_k = self._step(func, x_k, d_k, s_k)
            self._after_step(k,func, x_k, trajectory)

        x_min = x_k
        f_min = np.append(x_min, func(x_min))
        return x_min, f_min, np.array(trajectory)


class CGradientOptimizer(IntervalConstrainedOptimizer):
    """
    Conditional gradient method (Frank–Wolfe algorithm) for intervals
    """

    def __init__(
            self, 
            stopper: Callable, 
            stepper: Callable, 
            verbose=False,
            ):
        self._stepper = stepper
        super().__init__(stopper, verbose)

    def _find_coordinate_parallelepiped_xi(
            self,
            diff_val: np.ndarray,
            interval: tuple,
            i: int,
            ):
        if diff_val[i] > 0:
            return interval[0]
        if diff_val[i] < 0:
            return interval[1]
        else:
            return (interval[1] - interval[0]) / 2

    def _find_mid_step(
            self,
            diff_val: np.ndarray,
            intervals: tuple[tuple, tuple],
            ):
        return (
            self._find_coordinate_parallelepiped_xi(diff_val, intervals[0], 0),
            self._find_coordinate_parallelepiped_xi(diff_val, intervals[1], 1),
        )

    def _step(
            self,
            func: function,
            x_k: Sequence | np.ndarray,
            intervals: tuple[tuple, tuple],
            iteration: int,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        diff_func, diff_val = self._diffx(func, x_k)

        x_k_ = self._find_mid_step(diff_val, intervals) 
        self._print(f'{x_k_=}')

        x_d = x_k_ - x_k
        self._print(f'{x_d=}')
        
        v_k = np.dot(diff_val, x_d)
        self._print(f'{v_k=}')
        if v_k == 0:
            return x_k, diff_val, None, v_k

        d_k = x_k_ - x_k
        self._print(f'{d_k=}')

        a_k, iterations = self._stepper(
                    func=func, 
                    x_k=x_k, 
                    diff_val=diff_val, 
                    d_k=d_k,
                    iteration=iteration,
                )
        self._print(f'got {a_k=} with {iterations=}')

        return x_k + a_k * d_k, diff_val, None, v_k

    def min(
            self,
            func: function,
            intervals: tuple[tuple, tuple],
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_0 = np.array([0, 0])
        k, x_k, diff_val, f_start, trajectory = self._prep(func, x_0)
        
        f_x_0 = func(x_0)
        self._print(f'{f_x_0=}')

        v_k = 0
        while not self._stopper(
                iteration=k,
                diff_val=diff_val,
                trajectory=trajectory,
                ) or v_k == 0:
            k += 1
            x_k, diff_val, _, v_k = self._step(func, x_k, intervals, k)
            self._after_step(k,func, x_k, trajectory)

        x_min = x_k
        f_min = np.append(x_min, func(x_min))
        return x_min, f_min, np.array(trajectory)


class QuadraticPenaltyOptimizer(FunctionConstrainedOptimizer, GradientOptimizer):
    def __init__(
            self, 
            stopper: Callable, 
            c_stepper: Callable, 
            a_stepper: Callable, 
            verbose=False,
            ):
        self._c_stepper = c_stepper
        FunctionConstrainedOptimizer.__init__(self, stopper, verbose)
        GradientOptimizer.__init__(self, stopper, a_stepper, verbose)

    def min(
            self,
            func: function,
            constrain: function,
            x_0: np.ndarray = None,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k, x_k, diff_val, f_start, trajectory = self._prep(func, x_0)
        while not self._stopper(
                iteration=k,
                diff_val=diff_val,
                trajectory=trajectory,
                ):
            self._print(f'{k=}')
            
            c_k, iterations = self._c_stepper(iteration=k)
            self._print(f'got {c_k=} with {iterations=}')

            aprox_func = func + (constrain ** 2) * (c_k / 2)
            self._print(f'{aprox_func=}')
            
            x_k, diff_val, hessian_val = self._step(aprox_func, x_k)
            self._after_step(k,func, x_k, trajectory)
            k += 1

        x_min = x_k
        f_min = np.append(x_min, func(x_min))
        return x_min, f_min, np.array(trajectory)

