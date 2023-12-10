from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify, symbols
from sympy.core.expr import Expr
from functools import reduce
from typing import Self
from collections.abc import Sequence, Callable

from optimus3d.types import *
import optimus3d.utils as utils

class function(str):

    @staticmethod
    def _replace(txt: str, pairs: dict[str, str]) -> str:
        for src, dst in pairs.items():
            txt = txt.replace(src, dst)
        return txt

    @staticmethod
    def _concat(*args: str | Self) -> Self:
        return ''.join((repr(a).replace("'", "") for a in args))

    @staticmethod
    def _make_nupy_valid_repr(expr_str: str) -> str:
        expr_str = function._replace(expr_str, {
            'sin':'np.sin',
            'cos':'np.cos',
            'tan':'np.tan',
            'exp':'np.exp',
            'log':'np.log',
            'log2':'np.log2',
            'log10':'np.log10',
        })
        return expr_str

    @staticmethod
    def _exptract_names(expr_str: str) -> tuple[str, str]:
        vals = function._replace(expr_str, {
            '+': ' ',
            '-': ' ',
            '*': ' ',
            '^': ' ',
            '**': ' ',
            'sin':' ',
            'cos':' ',
            'exp':' ',
            'log':' ',
            'log2':' ',
            'log10':' ',
            '(': ' ',
            ')': ' ',
            '.': ' ',
        }).split(' ')
        names = filter(lambda v: not v.isnumeric() and v != '', vals)
        names = list(set(names))
        if len(names) != 2:
            raise Exception(f'wrong variables amount, got "{len(names)}" {names}, want "2"')
        return tuple(sorted(names))

    def __new__(cls, value, *args, **kwargs):
        return super(function, cls).__new__(cls, value)
        
    def __init__(
            self,
            expr_str: str = 'x1 + x2',
            var_names: tuple[str, str] = None,
            ):

        # define expression representations
        self._str = expr_str
        self._repr = expr_str
        self.sympy_valid_repr = self._replace(expr_str, {'^':'**'})
        self.np_valid_repr = function._make_nupy_valid_repr(self.sympy_valid_repr)
        
        # define variables representations
        if var_names is None:
            var_names = function._exptract_names(expr_str)
        self._var_names = var_names

        x1, x2 = var_names
        self._x = symbols(f'{x1} {x2}')
        self._x1, self._x2 = self._x
        self._expr = sympify(self.sympy_valid_repr)
        
        # diffs
        self._cached_diff = None
        self._cached_hessian = None

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return repr(self._repr)

    def __add__(self, other: Self | str | int):
        self._validate_dunders_input_type(other)
        expr_str = self._concat(self, ' + ', other)
        return function(expr_str, self._var_names) 
    
    def __sub__(self, other: Self | str | int):
        self._validate_dunders_input_type(other)
        expr_str = self._concat(self, ' - ', other)
        return function(expr_str, self._var_names) 

    def __mul__(self, other: Self | str | int):
        self._validate_dunders_input_type(other)
        expr_str = self._concat('(', self, ')', ' * ', '(', other, ')')
        return function(expr_str, self._var_names) 

    def __truediv__(self, other: Self | str | int):
        self._validate_dunders_input_type(other)
        expr_str = self._concat('(', self, ')', ' / ', '(', other, ')')
        return function(expr_str, self._var_names) 

    def __pow__(self, other: Self | str | int):
        self._validate_dunders_input_type(other)
        expr_str = self._concat('(', self, ')', ' ** ', '(', other, ')')
        return function(expr_str, self._var_names) 

    def _validate_dunders_input_type(self, other):
        if not (other.__class__ in (self.__class__, str, int, float)):
            raise Exception(f'wrong input type, want "function", got "{other.__class__}"')

    def _get_grid(self, grid_step: float, radius: int) -> tuple[np.array, np.array, np.array]:
        samples = np.arange(-radius, radius, grid_step)
        x_1, x_2 = np.meshgrid(samples, samples)
        return x_1, x_2, self(x_1, x_2)
    
    def show(
        self, 
        ax: mplot3d.axes3d.Axes3D = None,
        grid_step: float = 0.05, 
        radius: int = 8, 
        point: tuple = None, 
        trajectory: tuple = None, 
        name: str = None,
    ) -> None:

        grid = self._get_grid(grid_step, radius)
        
        ax = self.figure(
            ax=ax,
            grid_step=grid_step,
            radius=radius,
            point=point,
            trajectory=trajectory,
            name=name,
        )
        plt.ion()
        plt.show()

    def figure(
        self, 
        ax: mplot3d.axes3d.Axes3D = None,
        grid_step: float = 0.05, 
        radius: int = 8, 
        point: tuple = None, 
        trajectory: tuple = None, 
        name: str = None,
    ) -> mplot3d.axes3d.Axes3D:
        grid = self._get_grid(grid_step, radius)
        
        if ax is None:
            plt.ioff()
            
            x_1_name, x_2_name = self._var_names
            x_3_name = 'x3'

            if self._var_names != ('x1', 'x2'):
                x_3_name = 'z'

            ax = utils._setup_subplot(
                    x_1_name,
                    x_2_name,
                    x_3_name,
                    self._str,
            )

        if point is not None:
            point_x1, point_x2, point_x3 = point
            ax.scatter(point_x1, point_x2, point_x3, color='red')

        if trajectory is not None:
            x1 = trajectory[:, 0]
            x2 = trajectory[:, 1]
            x3 = trajectory[:, 2]

            ax.plot3D(x1, x2, x3, 'b')

            for trajectory_point in trajectory:
                point_x1, point_x2, point_x3 = trajectory_point
                ax.scatter(point_x1, point_x2, point_x3, color='red', s=3)

        grid_x1, grid_x2, grid_x3 = grid
        ax.plot_surface(grid_x1, grid_x2, grid_x3, rstride=5, cstride=5, alpha=0.7)

        return ax

    
    def _repr_latex_(self) -> str:
        return self._expr._repr_latex_()
    
    def __call__(self, *args) -> np.ndarray:
        def raise_args(): raise FuncArgsException
        x1, x2 = 0.0, 0.0
        match len(args):
            case 0:
                raise_args()
            case 1:
                if not issubclass(args[0].__class__, (Sequence, np.ndarray)):
                    raise_args()
                if len(args[0]) != 2:
                    raise_args()
                x1, x2 = args[0]
            case 2:
                x1, x2 = args[0], args[1]
            case _:
                raise_args()
        return np.array(eval(self.np_valid_repr))

    def _calc_diff(self) -> tuple[Self, Self]:
        dx_f = [self._expr.diff(x) for x in self._x]
        dx_f_functions = (function(repr(dxi_f), self._var_names) for dxi_f in dx_f)
        return tuple(dx_f_functions)
    
    def _diff(self) -> tuple[Self, Self]:
        if self._cached_diff is None:
            self._cached_diff = self._calc_diff()
        return self._cached_diff
    
    def diff(self) -> tuple[Self, Self]:
        return self._diff()
    
    def _calc_hessian(self) -> tuple[tuple[Self, Self], tuple[Self, Self]]:
        mtx = list()
        for xi in self._x:
            row = list()
            for xj in self._x:
                dxij_f = self._expr.diff(xj).diff(xi)
                row.append(function(repr(dxij_f), self._var_names))
            mtx.append(row)

        return tuple(mtx)
                
    
    def _hessian(self) -> tuple[tuple[Self, Self], tuple[Self, Self]]:
        if self._cached_hessian is None:
            self._cached_hessian = self._calc_hessian()
        return self._cached_hessian
        
    def hessian(self) -> tuple[tuple[Self, Self], tuple[Self, Self]]:
        return self._hessian()


