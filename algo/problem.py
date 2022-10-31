# Author: baichen318@gmail.com


import torch
import numpy as np
from torch import Tensor
from utils import assert_error
from abc import ABC, abstractmethod
from vlsi_flow.manager import vlsi_flow
from vlsi_flow.vlsi_report import get_report
from typing import List, Optional, Tuple, NoReturn
from dataset import load_dataset, ndarray_to_tensor
from design_space.boom_design_space import parse_boom_design_space


class BaseProblem(torch.nn.Module, ABC):
    """
        base class for construction a problem.
    """

    dim: int
    _bounds: List[Tuple[float, float]]
    _check_grad_at_opt: bool = True

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        """
            base class for construction a problem.

        args:
            noise_std: standard deviation of the observation noise.
            negate: if True, negate the function.
        """
        super().__init__()
        self.noise_std = noise_std
        self.negate = negate
        self.register_buffer(
            "bounds", torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)
        )

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        """
            evaluate the function on a set of points.

        args:
            X: a `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: if `True`, add observation noise as specified by `noise_std`.

        returns:
            a `batch_shape`-dim tensor of function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)

    @abstractmethod
    def evaluate_true(self, X: Tensor) -> Tensor:
        """
            evaluate the function (w/o observation noise) on a set of points.
        """
        raise NotImplementedError


class MultiObjectiveProblem(BaseProblem):
    """
        base class for a multi-objective problem.
    """

    num_objectives: int
    _ref_point: List[float]
    _max_hv: float

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        """
            base constructor for multi-objective test functions.

        args:
            noise_std: standard deviation of the observation noise.
            negate: if True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)
        ref_point = torch.tensor(self._ref_point, dtype=torch.float)
        if negate:
            ref_point *= -1
        self.register_buffer("ref_point", ref_point)

    @property
    def max_hv(self) -> float:
        try:
            return self._max_hv
        except AttributeError:
            raise NotImplementedError(
                error_message("problem {} does not specify maximal hypervolume".format(
                    self.__class__.__name__)
                )
            )

    def gen_pareto_front(self, n: int) -> Tensor:
        """
            generate `n` pareto optimal points.
        """
        raise NotImplementedError


class DesignSpaceProblem(MultiObjectiveProblem):
    def __init__(self, configs: dict):
        self.configs = configs
        if configs["mode"] == "offline":
            self.load_dataset()
        else:
            assert configs["mode"] == "online", \
                assert_error("working mode should set online.")
            self.design_space = parse_boom_design_space(self.configs)
            self.generate_design_space()

        self._ref_point = torch.tensor([0.0, 0.0])
        self._bounds = torch.tensor([(2.0, 2.0)])
        super().__init__()

    def load_dataset(self) -> NoReturn:
        x, y = load_dataset(self.configs["dataset"]["path"])
        self.total_x = ndarray_to_tensor(x)
        self.total_y = ndarray_to_tensor(y[:, :-1])
        self.time = ndarray_to_tensor(y[:, -1])
        self.x = self.total_x.clone()
        self.y = self.total_y.clone()
        self.n_dim = self.x.shape[-1]
        self.n_sample = self.x.shape[0]

    def generate_design_space(self) -> NoReturn:
        x = []
        for i in range(len(self.design_space)):
            x.append(self.design_space.idx_to_vec(i))
        self.total_x = ndarray_to_tensor(np.array(self.x))
        self.x = self.total_x.clone()
        self.n_dim = self.x.shape[-1]
        self.n_sample = self.x.shape[0]

    def evaluate_true(self, x: torch.Tensor) -> torch.Tensor:
        if self.configs["mode"] == "offline":
            _, indices = torch.topk(
                ((self.x.t() == x.unsqueeze(-1)).all(dim=1)).int(),
                1,
                1
            )
            return self.y[indices].to(torch.float32).squeeze()
        else:
            idx = [self.design_space.vec_to_idx(_x.numpy().tolist()) for _x in x]
            self.design_space.generate_chisel_codes(idx)
            vlsi_flow(self.design_space, idx)
            perf, power, _ = get_report(self.design_space)
            y = torch.cat(
                (ndarray_to_tensor(perf).unsqueeze(1), ndarray_to_tensor(power).unsqueeze(1)),
                dim=1
            )
            return y

    def remove_sampled_data(self, x: torch.Tensor) -> NoReturn:
        if self.configs["mode"] == "offline":
            sampled = torch.zeros(
                self.x.size()[0],
                dtype=torch.bool
            )[:, np.newaxis]
            _, indices = torch.topk(
                ((self.x.t() == x.unsqueeze(-1)).all(dim=1)).int(),
                1,
                1
            )
            mask = sampled.index_fill_(0, indices.squeeze(), True).squeeze()
            self.x = self.x[mask[:] == False]
            self.y = self.y[mask[:] == False]


def create_problem(configs: dict) -> DesignSpaceProblem:
    return DesignSpaceProblem(configs)
