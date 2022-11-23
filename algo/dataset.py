# Author: baichen318@gmail.com


import csv
import torch
import numpy as np
from utils import if_exist
from typing import Union, Tuple, List


def load_dataset(path: str, preprocess=True) -> Tuple[np.ndarray, np.ndarray]:
    def _read_csv() -> Tuple[List[List[int]], List[str]]:
        dataset = []
        if_exist(path, strict=True)
        with open(path, 'r') as f:
            reader = csv.reader(f)
            title = next(reader)
            for row in reader:
                dataset.append(row)
        return dataset, title

    def validate(dataset: List[List[int]]) -> np.ndarray:
        """
            `dataset`: <tuple>
        """
        data = []
        for item in dataset:
            _data = []
            f = item[0].split(' ')
            for i in f:
                _data.append(int(i))
            for i in item[1:]:
                _data.append(float(i))
            data.append(_data)
        data = np.array(data)

        return data

    dataset, _ = _read_csv()
    dataset = validate(dataset)
    if preprocess:
        dataset = scale_dataset(dataset)
    # split to two matrices
    x = []
    y = []
    for data in dataset:
        x.append(data[:-3])
        y.append(np.array([data[-3], data[-2], data[-1]]))
    return np.array(x), np.array(y)


def scale_dataset(
    dataset: Union[torch.Tensor, np.ndarray],
    perf_idx: int = -3,
    power_idx: int = -2
) -> Union[torch.Tensor, np.ndarray]:
    """
        NOTICE: scale the data by `max - x / \alpha`
        max clock cycles: 84103
        min clock cycles: 63539
        max power: 0.14650000000000002
        min power: 0.045950000000000005
        after scaling, the clock cycles are [0.2948, 1.32305]
        the power values are [0.5349999999999999, 1.5405000000000002]
    """
    if isinstance(dataset, torch.Tensor):
        _dataset = dataset.clone()
    else:
        _dataset = dataset.copy()
    _dataset[:, perf_idx] = (90000 - _dataset[:, perf_idx]) / 20000
    _dataset[:, power_idx] = (0.2 - _dataset[:, power_idx]) * 10
    return _dataset


def rescale_dataset(
    dataset: Union[torch.Tensor, np.ndarray],
    perf_idx: int = -3,
    power_idx: int = -2
) -> Union[torch.Tensor, np.ndarray]:
    """
        NOTICE: please see `scale_dataset`
    """
    if isinstance(dataset, torch.Tensor):
        _dataset = dataset.clone()
    else:
        _dataset = dataset.copy()
    if _dataset.shape[1] == 3:
        perf_idx = -3
        power_idx = -2
    else:
        assert _dataset.shape[1] == 2
        perf_idx = -2
        power_idx = -1
    _dataset[:, perf_idx] = 90000 - 20000 * _dataset[:, perf_idx]
    _dataset[:, power_idx] = 0.2 - _dataset[:, power_idx] / 10

    return _dataset


def tensor_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy()


def ndarray_to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.Tensor(array)
