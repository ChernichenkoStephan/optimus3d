from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

import optimus3d.utils as utils

def _get_surface_grid(
        grid_step: float,
        intervals: tuple[tuple, tuple],
        slide: int = 0,
        ) -> tuple[np.array, np.array, np.array]:

    x_1 = np.arange(intervals[0][0], intervals[0][1], grid_step)
    x_2 = np.arange(intervals[1][0], intervals[1][1], grid_step)
    x_1, x_2 = np.meshgrid(x_1, x_2)
    x_3 = np.ones((len(x_1),len(x_1[0])), dtype=float) * slide
    return x_1, x_2, x_3

def get_surface_grid(
        grid_step: float,
        intervals: tuple[tuple, tuple],
        slide: int = 0,
        ) -> tuple[np.array, np.array, np.array]:
    return _get_surface_grid(grid_step, intervals, slide)


def surface_figure(
    ax: mplot3d.axes3d.Axes3D = None,
    grid_step: float = 0.05, 
    intervals: tuple[tuple, tuple] = ((0, 1), (0, 1)),
    slide: int = 0,
) -> mplot3d.axes3d.Axes3D:
    print(intervals)

    grid = _get_surface_grid(grid_step, intervals, slide)
    
    if ax is None:
        plt.ioff()
        ax = utils._setup_subplot()

    grid_x1, grid_x2, grid_x3 = grid
    ax.plot_surface(grid_x1, grid_x2, grid_x3, rstride=5, cstride=5, alpha=0.7)

    return ax

def surface_show(
    ax: mplot3d.axes3d.Axes3D = None,
    grid_step: float = 0.05, 
    intervals: tuple[tuple, tuple] = ((0, 1), (0, 1)),
    slide: int = 0,
) -> None:
    ax = surface_figure(
        ax=ax,
        grid_step=grid_step,
        intervals=intervals,
        slide=slide,
    )
    plt.ion()
    plt.show()
