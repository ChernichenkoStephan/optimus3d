from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def _unpack(*names, **kwargs) -> tuple:
    vals = []
    for n in names:
        if n not in kwargs:
            raise Exception(f'missing: "{n}"')
        vals.append(kwargs[n])
    return tuple(vals)

def _optional_printer(shuld_print: bool):
    def print_func(val):
        if shuld_print:
            print(val)
    return print_func

def _setup_subplot(
            x1_name: str = 'x1',
            x2_name: str = 'x2',
            x3_name: str = 'x3',
            title: str = None,
        ) -> mplot3d.axes3d.Axes3D:
    plt.rcParams.update({
        'figure.figsize': (4, 4),
        'figure.dpi': 200,
        'xtick.labelsize': 4,
        'ytick.labelsize': 4
    })
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.set_xlabel(x1_name)
    ax.set_ylabel(x2_name)
    ax.set_zlabel(x3_name)
    if title is not None:
        plt.title(title)
    return ax


