# Author: baichen318@gmail.com


import math
import random
import numpy as np
from typing import List, NoReturn
from dataset import ndarray_to_tensor
from problem import DesignSpaceProblem


class RandomizedTED(object):

    def __init__(self, kwargs: dict):
        super(RandomizedTED, self).__init__()
        self.Nrted = kwargs["Nrted"]
        self.mu = kwargs["mu"]
        self.sig = kwargs["sig"]

    def f(self, u, v):
        u = u[:-2]
        v = v[:-2]
        return pow(
            math.e,
            -np.linalg.norm(
                np.array(u, dtype=np.float64) - np.array(v, dtype=np.float64)
            ) ** 2 / (2 * self.sig ** 2)
        )

    def f_same(self, K: List[List[int]]) -> np.ndarray:
        n = len(K)
        F = []
        for i in range(n):
            t = []
            for j in range(n):
                t.append(self.f(K[i], K[j]))
            F.append(t)
        return np.array(F)

    def update_f(self, F: List[List[int]], K: List[int]) -> NoReturn:
        n = F.shape[0]
        for i in range(len(K)):
            denom = self.f(K[i], K[i]) + self.mu
            for j in range(n):
                for k in range(n):
                    F[j][k] -= (F[j][i] * F[k][i]) / denom

    def select_mi(self, K: List[List[int]], F: List[List[int]]) -> List[List[int]]:
        return K[
            np.argmax(
                [np.linalg.norm(F[i]) ** 2 / (self.f(K[i], K[i]) + self.mu) \
                    for i in range(len(K))]
            )
        ]

    def rted(self, vec: np.ndarray, m: int) -> List[List[int]]:
        """
            vec: the dataset
            m: the number of samples initialized from `vec`
        """
        K_ = []
        for i in range(m):
            M_ = random.sample(list(vec), self.Nrted)
            M_ = M_ + K_
            M_ = [tuple(M_[j]) for j in range(len(M_))]
            M_ = list(set(M_))
            F = self.f_same(M_)
            self.update_f(F, M_)
            K_.append(self.select_mi(M_, F))
        return K_


class MicroAL(RandomizedTED):

    # dataset constructed after cluster w.r.t. DecodeWidth
    _cluster_dataset = None
   
    def __init__(self, problem: DesignSpaceProblem):
        self.problem = problem
        self.configs = problem.configs["initialize"]
        self.num_per_cluster = self.configs["batch"] // self.configs["cluster"]
        self.decoder_threshold = self.configs["decoder-threshold"]
        # feature dimension
        self.n_dim = problem.n_dim
        super(MicroAL, self).__init__(self.configs)

    @property
    def cluster_dataset(self):
        return self._cluster_dataset

    @cluster_dataset.setter
    def cluster_dataset(self, dataset):
        self._cluster_dataset = dataset

    def set_weight(self, pre_v=None):
        # if `pre_v` is specified, then `weights` will be assigned accordingly
        if pre_v:
            assert isinstance(pre_v, list) and len(pre_v) == self.n_dim, \
                assert_error("unsupported pre_v {}".format(pre_v))
            weights = pre_v
        else:
            # NOTICE: `decodeWidth` should be assignd with larger weights
            weights = [1 for i in range(self.n_dim)]
            weights[1] *= self.decoder_threshold
        return weights

    def distance(self, x, y, l=2, pre_v=None):
        """calculates distance between two points"""
        weights = self.set_weight(pre_v=pre_v)
        return np.sum((x - y)**l * weights).astype(float)

    def kmeans(self, points, k, max_iter, pre_v=None):
        """k-means clustering algorithm"""
        centroids = [points[i] for i in np.random.randint(len(points), size=k)]
        new_assignment = [0] * len(points)
        old_assignment = [-1] * len(points)

        i = 0
        split = False
        while i < max_iter or split == True and new_assignment != old_assignment:
            old_assignment = list(new_assignment)
            split = False
            i += 1

            for p in range(len(points)):
                distances = [self.distance(points[p], centroids[c], pre_v=pre_v) \
                    for c in range(len(centroids))]
                new_assignment[p] = np.argmin(distances)

            for c in range(len(centroids)):
                members = [points[p] for p in range(len(points)) if new_assignment[p] == c]
                if members:
                    centroids[c] = np.mean(members, axis=0).astype(int)
                else:
                    centroids[c] = points[np.random.choice(len(points))]
                    split = True

        loss = np.sum([self.distance(points[p], centroids[new_assignment[p]], pre_v=pre_v) \
            for p in range(len(points))])

        return centroids, new_assignment, loss

    def gather_groups(self, dataset, cluster):
        new_dataset = [[] for i in range(self.configs["cluster"])]

        for i in range(len(dataset)):
            new_dataset[cluster[i]].append(dataset[i])
        for i in range(len(new_dataset)):
            new_dataset[i] = np.array(new_dataset[i])
        if self.configs["vis-micro-al"]:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE

            vis_micro_al(new_dataset)
        return new_dataset

    def initialize(self, dataset: np.ndarray) -> List[List[int]]:
        # NOTICE: `rted` may select duplicated points,
        # in order to avoid this problem, we delete
        # duplicated points randomly
        def _delete_duplicate(vec):
            """
                `vec`: <list>
            """
            return [list(v) for v in set([tuple(v) for v in vec])]

        centroids, new_assignment, loss = self.kmeans(
            dataset,
            self.configs["cluster"],
            max_iter=self.configs["clustering-iteration"]
        )
        self.cluster_dataset = self.gather_groups(dataset, new_assignment)

        sampled_data = []
        for c in self.cluster_dataset:
            x = []
            while len(x) < min(self.num_per_cluster, len(c)):
                if len(c) > (self.num_per_cluster - len(x)) and \
                    len(c) > self.configs["Nrted"]:
                    candidates = self.rted(
                        c,
                        self.num_per_cluster - len(x)
                    )
                else:
                    candidates = c
                for _c in candidates:
                    x.append(_c)
                l = len(x)
                x = _delete_duplicate(x)
                if len(x) == l:
                    """
                        A bug fix, please refer it to:
                        https://github.com/baichen318/boom-explorer-public/issues/2
                    """
                    break
            for _x in x:
                sampled_data.append(_x)
        return sampled_data


def micro_al(problem: DesignSpaceProblem):
    initializer = MicroAL(problem)
    x = ndarray_to_tensor(initializer.initialize(problem.x.numpy()))
    y = problem.evaluate_true(x)
    problem.remove_sampled_data(x)
    return x, y
