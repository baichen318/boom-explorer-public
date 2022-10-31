# Author: baichen318@gmail.com


import torch
import numpy as np
from botorch.utils.multi_objective.pareto import is_non_dominated


def get_pareto_frontier(y: torch.Tensor, reverse=True):
    """
        NOTICE: `is_non_dominated` assumes maximization
        reverse is set to True when you call `rescale_dataset`
        before `get_pareto_set`,
        reverse is set to False when you call `get_pareto_set`
        before `rescale_dataset`.
        refer to `scale_dataset` and `rescale_dataset` in util.py
    """
    if reverse:
        return y[is_non_dominated(-y)]
    else:
        return y[is_non_dominated(y)]


def get_pareto_optimal_solutions(x: torch.Tensor, y: torch.Tensor):
    return x[is_non_dominated(y)]


def calc_adrs(reference, learned_pareto_set):
    """
        reference: <torch.Tensor>
        learned_pareto_set: <torch.Tensor>
    """
    # calculate average distance to the reference set
    ADRS = 0
    try:
        reference = reference.cpu()
        learned_pareto_set = learned_pareto_set.cpu()
    except:
        pass
    for omega in reference:
        mini = float('inf')
        for gama in learned_pareto_set:
            mini = min(mini, np.linalg.norm(omega - gama))
        ADRS += mini
    ADRS = ADRS / len(reference)
    return ADRS


