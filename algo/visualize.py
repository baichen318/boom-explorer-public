# Author: baichen318@gmail.com


from utils import info
import matplotlib.pyplot as plt
from dataset import load_dataset


markers = [
    '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3',
    '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
    'X', 'D', 'd', '|', '_'
]
colors = [
    'c', 'b', 'g', 'r', 'm', 'y', 'k', # 'w'
]


def plot_pareto_set(data, **kwargs):
    """
        see its usage in `boom_explorer.py`
    """
    plt.figure()
    design_space = kwargs["design_space"]
    x, y = load_dataset(design_space, preprocess=False)

    plt.scatter(
        y[:, 0],
        y[:, 1],
        s=3,
        marker=markers[19],
        c=colors[-1],
        alpha=0.2,
        label="Design space"
    )
    plt.scatter(
        kwargs["gt"][:, 0],
        kwargs["gt"][:, 1],
        s=3,
        marker=markers[12],
        c=colors[1],
        label="GT"
    )
    plt.scatter(
        data[:, 0],
        data[:, 1],
        s=3,
        marker=markers[15],
        c=colors[3],
        label="Pareto set"
    )
    plt.legend()
    plt.xlabel("C.C.")
    plt.ylabel("Power")
    plt.title("C.C. vs. Power (BOOM-Explorer)")
    info("save the figure: {}.".format(kwargs["output"]))
    plt.savefig(kwargs["output"])
    plt.show()
