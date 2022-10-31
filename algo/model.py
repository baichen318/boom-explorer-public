# Author: baichen318@gmail.com


import os
import sys
import tqdm
import torch
import gpytorch
import numpy as np
import torch.nn as nn
from time import time
from torch.nn import Module
from utils import info, assert_error
from gpytorch.priors.prior import Prior
from gpytorch.models.exact_gp import ExactGP
from gpytorch.means.linear_mean import LinearMean
from botorch.utils.containers import TrainingData
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from botorch.models.gpytorch import MultiTaskGPyTorchModel
from botorch.models.transforms.input import InputTransform
from typing import Optional, Dict, Tuple, NoReturn, List, Any
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class MOGP(ExactGP, MultiTaskGPyTorchModel):
    """
        Multi-objective GP model with ICM kernel, inferring observation noise.
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        task_feature: int,
        mean_type: str = "constant",
        covar_module: Optional[Module] = None,
        task_covar_prior: Optional[Prior] = None,
        output_tasks: Optional[List[int]] = None,
        rank: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
    ) -> None:
        """
        Args:
            train_X: A `n x (d + 1)` or `b x n x (d + 1)` (batch mode) tensor
                of training data. One of the columns should contain the task
                features (see `task_feature` argument).
            train_Y: A `n` or `b x n` (batch mode) tensor of training observations.
            task_feature: The index of the task feature (`-d <= task_feature <= d`).
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            rank: The rank to be used for the index kernel. If omitted, use a
                full rank (i.e. number of tasks) kernel.
            task_covar_prior : A Prior on the task covariance matrix. Must operate
                on p.s.d. matrices. A common prior for this is the `LKJ` prior.
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        all_tasks, task_feature, d = self.get_all_tasks(
            transformed_X, task_feature, output_tasks
        )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)

        # squeeze output dim
        train_Y = train_Y.squeeze(-1)
        if output_tasks is None:
            output_tasks = all_tasks
        else:
            if set(output_tasks) - set(all_tasks):
                raise RuntimeError(assert_error(
                    "All output tasks must be present in input data.")
                )
        self._output_tasks = output_tasks
        self._num_outputs = len(output_tasks)

        likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05))

        # construct indexer to be used in forward
        self._task_feature = task_feature
        self._base_idxr = torch.arange(d)
        # exclude task feature
        self._base_idxr[task_feature:] += 1

        super().__init__(
            train_inputs=train_X,
            train_targets=train_Y,
            likelihood=likelihood
        )
        if mean_type == "linear":
            self.mean_module = LinearMean(input_size=d)
        else:
            self.mean_module = ConstantMean()
        if covar_module is None:
            self.covar_module = ScaleKernel(
                base_kernel=MaternKernel(
                    nu=2.5, ard_num_dims=d, lengthscale_prior=GammaPrior(3.0, 6.0)
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        else:
            self.covar_module = covar_module

        num_tasks = len(all_tasks)
        self._rank = rank if rank is not None else num_tasks

        self.task_covar_module = IndexKernel(
            num_tasks=num_tasks, rank=self._rank, prior=task_covar_prior
        )
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)

    def _split_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts base features and task indices from input data.

        Args:
            x: The full input tensor with trailing dimension of size `d + 1`.
                Should be of float/double data type.

        Returns:
            2-element tuple containing

            - A `q x d` or `b x q x d` (batch mode) tensor with trailing
            dimension made up of the `d` non-task-index columns of `x`, arranged
            in the order as specified by the indexer generated during model
            instantiation.
            - A `q` or `b x q` (batch mode) tensor of long data type containing
            the task indices.
        """
        batch_shape, d = x.shape[:-2], x.shape[-1]
        x_basic = x[..., self._base_idxr].view(batch_shape + torch.Size([-1, d - 1]))
        task_idcs = (
            x[..., self._task_feature]
            .view(batch_shape + torch.Size([-1, 1]))
            .to(dtype=torch.long)
        )
        return x_basic, task_idcs

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        x_basic, task_idcs = self._split_inputs(x)
        # Compute base mean and covariance
        mean_x = self.mean_module(x_basic)
        covar_x = self.covar_module(x_basic)
        # Compute task covariances
        covar_i = self.task_covar_module(task_idcs)
        # Combine the two in an ICM fashion
        covar = covar_x.mul(covar_i)
        return MultivariateNormal(mean_x, covar)

    @classmethod
    def get_all_tasks(
        cls,
        train_X: torch.Tensor,
        task_feature: int,
        output_tasks: Optional[List[int]] = None,
    ) -> Tuple[List[int], int, int]:
        if train_X.ndim != 2:
            # batch mode MTGPs are blocked upstream in GPyTorch
            raise ValueError(assert_error(
                    "Unsupported shape {} for train_X.".format(train_X.shape)
                )
            )
        d = train_X.shape[-1] - 1
        if not (-d <= task_feature <= d):
            raise ValueError(assert_error(
                    "Must have that -{} <= task_feature <= {}".format(d, d)
                )
            )
        task_feature = task_feature % (d + 1)
        all_tasks = train_X[:, task_feature].unique().to(dtype=torch.long).tolist()
        return all_tasks, task_feature, d

    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs) -> Dict[str, Any]:
        """
        Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: Additional options for the model that pertain to the
                training data, including:

                - `task_features`: Indices of the input columns containing the task
                  features (expected list of length 1),
                - `task_covar_prior`: A GPyTorch `Prior` object to use as prior on
                  the cross-task covariance matrix,
                - `prior_config`: A dict representing a prior config, should only be
                  used if `prior` is not passed directly. Should contain:
                  `use_LKJ_prior` (whether to use LKJ prior) and `eta` (eta value,
                  float),
                - `rank`: The rank of the cross-task covariance matrix.
        """
        task_features = kwargs.pop("task_features", None)
        if task_features is None:
            raise ValueError(f"`task_features` required for {cls.__name__}.")
        task_feature = task_features[0]
        inputs = {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "task_feature": task_feature,
            "rank": kwargs.get("rank"),
        }

        prior = kwargs.get("task_covar_prior")
        prior_config = kwargs.get("prior_config")
        if prior and prior_config:
            raise ValueError(
                assert_error("Only one of `prior` and `prior_config` arguments expected.")
            )

        if prior_config:
            if not prior_config.get("use_LKJ_prior"):
                raise ValueError(assert_error("Currently only config for LKJ prior is supported."))
            all_tasks, _, _ = MOGP.get_all_tasks(training_data.X, task_feature)
            num_tasks = len(all_tasks)
            sd_prior = GammaPrior(1.0, 0.15)
            sd_prior._event_shape = torch.Size([num_tasks])
            eta = prior_config.get("eta", 0.5)
            if not isinstance(eta, float) and not isinstance(eta, int):
                raise ValueError(assert_error(
                        "eta must be a real number, your eta was {}.".format(eta)
                    )
                )
            prior = LKJCovariancePrior(num_tasks, eta, sd_prior)

        inputs["task_covar_prior"] = prior
        return inputs


class MLP(nn.Sequential):
    """
        MLP as preprocessor of DKLGP
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(MLP, self).__init__()
        # NOTICE: we do not add spectral normalization
        self.add_module("linear-1", nn.Linear(input_dim, 1000))
        self.add_module("relu-1", nn.ReLU())
        self.add_module("linear-2", nn.Linear(1000, 500))
        self.add_module("relu-2", nn.ReLU())
        self.add_module("linear-3", nn.Linear(500, 50))
        self.add_module("relu-3", nn.ReLU())
        self.add_module("linear-4", nn.Linear(50, output_dim))


class DKLGP(object):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, **kwargs: Dict):
        self.n_dim = x.shape[-1]
        self.n_target = y.shape[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = MLP(self.n_dim, kwargs["mlp_output_dim"])
        self.mlp.apply(weights_init)
        x = self.forward_mlp(x)
        x = self.transform_xlayout(x)
        y = self.transform_ylayout(y)
        self.gp = MOGP(x, y, task_feature=-1, mean_type="linear")

    def set_train(self) -> NoReturn:
        self.gp.train()
        self.gp.likelihood.train()
        self.mlp.train()
        self.gp = self.gp.to(self.device)
        self.likelihood = self.gp.likelihood.to(self.device)
        self.mlp = self.mlp.to(self.device)

    def set_eval(self) -> NoReturn:
        self.mlp.eval()
        self.gp.eval()
        self.gp.likelihood.eval()

    def transform_xlayout(self, x: torch.Tensor) -> torch.Tensor:
        """
            [x1; x2; x3]  <-->  [y11, y12; y21, y22; y31, y33]
                =>
            [x1, 0; x2, 0; x3, 0; x1, 1; x2, 1; x3, 1]  <-->  [y11; y21; y31; y12; y22; y32]
        """
        x = x.to(self.device)
        nsample = x.shape[0]
        x = torch.cat([x for i in range(self.n_target)], dim=0)
        task_index = torch.zeros(nsample, 1).to(x.device)
        for i in range(1, self.n_target):
            task_index = torch.cat([task_index, i * torch.ones(nsample, 1).to(x.device)], dim=0)
        x = torch.cat([x, task_index], dim=1)
        return x

    def transform_ylayout(self, y: torch.Tensor) -> torch.Tensor:
        """
            [x1; x2; x3]  <-->  [y11, y12; y21, y22; y31, y33]
                =>
            [x1, 0; x2, 0; x3, 0; x1, 1; x2, 1; x3, 1]  <-->  [y11; y21; y31; y12; y22; y32]
        """
        y = y.to(self.device)
        y = y.chunk(self.n_target, dim=1)
        y = torch.cat([y[i] for i in range(len(y))], dim=0)
        return y

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.mlp(x)
        # normalization
        x = x - x.min(0)[0]
        x = 2 * (x / x.max(0)[0]) - 1
        return x

    def train(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.forward_mlp(x)
        x = self.transform_xlayout(x)
        self.gp.set_train_data(x)
        y = self.gp(x)

        return y

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        def _transform_ylayout(y: torch.Tensor) -> torch.Tensor:
            y = y.chunk(self.n_target, dim=0)
            return torch.cat([y[i].unsqueeze(1) for i in range(2)], dim=1)

        self.set_eval()
        with torch.no_grad(), \
            gpytorch.settings.use_toeplitz(False),\
            gpytorch.settings.fast_pred_var():
            x = self.forward_mlp(x)
            x = self.transform_xlayout(x)
            pred = self.gp(x)
        pred = _transform_ylayout(pred.mean)

        return pred

    def sample(self, x: torch.Tensor, y: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        acq_func = ExpectedHypervolumeImprovement(
            model=self.gp,
            ref_point=torch.tensor([1, 1]).to(self.device),
            partitioning=NondominatedPartitioning(
                ref_point=torch.tensor([1, 1]).to(self.device),
                Y=y.to(self.device)
            )
        ).to(self.device)
        _x = self.mlp(x.to(self.device).float())
        acqv = acq_func(_x.unsqueeze(1).to(self.device))
        top_k, idx = torch.topk(acqv, k=5)
        new_x = x[idx]
        new_y = y[idx]

        return new_x.reshape(-1, self.n_dim), new_y.reshape(-1, 2), torch.mean(top_k)

    def save(self, path: str) -> NoReturn:
        state_dict = {
            "mlp": self.mlp.state_dict(),
            "gp": self.gp.state_dict()
        }
        torch.save(state_dict, path)
        info("saving model to {}.".format(path))

    def load(self, mdl: str) -> NoReturn:
        state_dict = torch.load(mdl)
        self.mlp.load_state_dict(state_dict["mlp"])
        self.gp.load_state_dict(state_dict["gp"])
        self.set_eval()


def initialize_dkl_gp(
    x: torch.Tensor, y: torch.Tensor, mlp_output_dim: int
) -> DKLGP:
    return DKLGP(x, y, mlp_output_dim=mlp_output_dim)
