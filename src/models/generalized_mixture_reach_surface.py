# Copyright 2021 The Private Cardinality Estimation Framework Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Class for modeling Generalized mixture reach surface."""

import copy
from typing import List
import numpy as np
from cvxopt import matrix
from heapq import heappushpop, heapify
from cvxopt import solvers
from typing import Iterable
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve

solvers.options["show_progress"] = False

MIN_IMPROVEMENT_PER_ROUND = 1e-3
MAX_NUM_ROUNDS = 1000
DISTANCE_TO_THE_WALL = 0.01
NUM_NEAR_THE_WALL = 10
MIN_NUM_INIT = 30
MAX_NUM_INIT = 10


class GeneralizedMixtureReachSurface(ReachSurface):
    """Models reach with the generalized mixture overlap model.

    GeneralizedMixtureReachSurface models the combined reach with the formula

    ExpectedUnionReach = N * sum_{j=1..c} [1 - prod_{i=1..p} (1-a_i_j r_i / N)]

    where :
          all a_i_j >= 0
          sum_{j=1..c} a_i_j=1 for all i.

    Here p is the number of publishers, c is the number of clusters,
    and N is explained in the docstring.

    reference doc:
    https://drive.google.com/file/d/16vlwlB6e21cIFOpH0V7-ys6UPjc1wAMy/

    """

    def __init__(
        self,
        reach_curves: Iterable[ReachCurve],
        reach_points: Iterable[ReachPoint],
        num_clusters: int,
    ):
        """Constructor for GeneralizedMixtureReachSurface.

        Args:
          reach_curves: A list of ReachCurves to be used in model fitting and
            reach prediction.
          reach_points: A list of ReachPoints to which the model is to be fit.
            This list is of arbitrary length, but it should contain a minimum of
            c*(p-1)/2 points, where p is the number of publishers. Otherwise,
            there is no guarantee of a unique solution. The reach points
            represent arbitrary points on the reach surface, involving spend at
            multiple advertisers
          num_clusters: number of clusters.
          N : A non-negative value that is at least as large as the maximum
            reach of any of the per-publisher reach functions, e.g., N >=
            reach_curves[i].max_reach for i=1,2,...,p.
        """

        self._reach_curves = copy.deepcopy(reach_curves)
        self._n = len(reach_points)
        self._p = len(reach_points[0].impressions)
        self._c = num_clusters
        # TODO(uakyol) : Optimize N later.
        self._N = max([reach_curve.max_reach for reach_curve in reach_curves]) * 2
        self._initial_a = np.random.dirichlet(
            np.ones(self._c) / 5, size=self._p
        ).flatten()
        super().__init__(data=reach_points)

    def _define_criteria(
        self,
        min_improvement_per_round: float = MIN_IMPROVEMENT_PER_ROUND,
        max_num_rounds: int = MAX_NUM_ROUNDS,
        distance_to_the_wall: float = DISTANCE_TO_THE_WALL,
        num_near_the_wall: int = NUM_NEAR_THE_WALL,
        min_num_init: int = MIN_NUM_INIT,
        max_num_init: int = MAX_NUM_INIT,
    ):
        self._min_improvement = min_improvement_per_round
        self._max_num_rounds = max_num_rounds
        self._distance_to_the_wall = distance_to_the_wall
        self._num_near_the_wall = num_near_the_wall
        self._min_num_init = min_num_init
        self._max_num_init = max_num_init

    def get_theta(self, x):
        return (1 - (1 - x).prod(axis=1)) / x.sum(axis=1)

    def get_x(self, a, r):
        return np.array(
            [
                a[[j + i * self._c for i in range(self._p)]] * r / self._N
                for j in range(self._c)
            ]
        )

    def update_theta_from_a(self, a, r):
        """Udpate theta using equation:

        theta_j := theta(a_1_j r_1 / N, ..., a_p_j r_p / N)

        With theta_j, the union reach can be represented as a linear
        combination of theta_j.
        Thus if we know all theta_j, we can solve the parameters a by linear
        regression. Now theta_j also depends on a, and we propose iteratively
        solving a from theta and updating theta from a.
        The idea is similar to the classical Expectation-maximization algorithm.

        Args:
          a: a length (p * c) vector. Current best guess of a.
          r: a length n list of length p vectors. Single publisher reaches for
            each training point.

        Returns:
          n x c matrix th, where
          th[i][j] = theta(a[1][j] r[i][1] / N, a[2][j] r[i][2] / N, ...,
          a[p][j] r[i][p] / N)
        """
        return [self.get_theta(self.get_x(a, r[k])) for k in range(self._n)]

    def solve_a_given_z(self, z, show_status=False):
        """A key intermediate step to fit GM model.

        Args:
          y: an n x 1 matrix of total reach values
          r: an n x p matrix of per-publisher reach values
          th: an n x c matrix of theta values
          N: total universe size

        Returns:
          a: a c x p matrix that minimizes
          sum_{k=1..n} [ y^k - sum_{i=1..p} sum_{j=1..c} a_i_j z_i_j^k} ]^2
          subject to the constraints that a[i][j] >= 0 and row sums are one.
        """

        self._P = matrix(
            sum(
                [
                    np.matmul(
                        z[k].reshape(self._p * self._c, 1),
                        z[k].reshape(1, self._p * self._c),
                    )
                    for k in range(self._n)
                ]
            )
        )
        self._q = -matrix(
            sum((self._data[k].reach() / self._N) * z[k] for k in range(self._n))
        )
        G = -matrix(np.eye(self._p * self._c))
        h = matrix(np.zeros(self._p * self._c))
        A = matrix(np.kron(np.eye(self._p), np.ones(self._c)))
        b = matrix(np.ones(self._p))
        res = solvers.qp(self._P, self._q, G, h, A, b)
        if show_status:
            print(res["status"])
        return np.array(res["x"]).flatten()

    def get_z_from_theta(self, theta, r):
        """Calculate intermidiate value z.

        Args:
          theta: a length n list of length c vectors. Current best guess of
            theta.
          r: a length n list of length p vectors. Single publisher reaches for
            each training point.

        Returns:
          a length-n list of length-(p * c) vectors indicating each r_i *
          theta_j / N for each training point.
        """
        return [
            np.matmul(r[k].reshape(self._p, 1), theta[k].reshape(1, self._c)).flatten()
            / self._N
            for k in range(self._n)
        ]

    def update_a_from_theta(self, theta, r):
        """Udpate theta.

        Args:
          theta: a length n list of length c vectors. Current best guess of
            theta.
          r: a length n list of length p vectors. Single publisher reaches for
            each training point.

        Returns:
          A length (p * c) vector indicating each $a_{ij}$.
        """
        z = self.get_z_from_theta(theta, r)
        return self.solve_a_given_z(z)

    def _loss(self, a):
        return (
            np.matmul(np.matmul(np.transpose(a), self._P), a)
            + np.matmul(np.transpose(self._q), a)
        )[0]

    def init_a_sampler(self):
        return np.random.dirichlet(np.ones(self._c) / 5, size=self._p).flatten()

    def _fit_multi_init_a(self, random_seed: int = 0):
        np.random.seed(random_seed)
        max_heap = [-np.Inf] * self._num_near_the_wall
        heapify(max_heap)

        # A List following the max heap order to save the negative losses to
        # save the smallest k locally optimum losses.
        # We take negative loss simply because we need a min heap to save the
        # smallest k values but python only (conveniently) supports a max heap.
        num_init = 0
        self._fitted_loss = np.Inf

        def _close_enough(heap: List[float]) -> bool:
            smallest_loss, kth_smallest_loss = -max(heap), -min(heap)
            return kth_smallest_loss < (1 + self._distance_to_the_wall) * smallest_loss

        while (
            num_init < self._min_num_init or not _close_enough(max_heap)
        ) and num_init < self._max_num_init:
            init_a = self.init_a_sampler()
            local_fit, local_loss, local_converge = self._fit_one_init_a(init_a)
            heappushpop(max_heap, -local_loss)
            # update the smallest k locally optimum losses
            if local_loss < self._fitted_loss:
                self._fitted_loss = local_loss
                self.a = local_fit
                self._model_success = local_converge
            num_init += 1
        self._model_success = (local_converge, _close_enough(max_heap))
        self._k_smallest_losses = sorted([-l for l in max_heap])

    def _fit_one_init_a(self, initial_a):
        prev_loss = 10000000000  # np.Inf #self._loss(initial_a)
        cur_loss = prev_loss - 1
        num_rounds = 0

        reach_vectors = np.array(
            [
                self.get_reach_vector(reach_point.impressions)
                for reach_point in self._data
            ]
        )

        cur_a = initial_a.copy()
        cur_theta = self.update_theta_from_a(cur_a, reach_vectors)

        while (
            num_rounds < self._max_num_rounds
            and cur_loss < prev_loss - self._min_improvement
        ):
            # lbd = self._round(lbd)
            cur_a = self.update_a_from_theta(cur_theta, reach_vectors)
            prev_loss = cur_loss
            cur_loss = self._loss(cur_a)
            # print("cur_loss",cur_loss)
            num_rounds += 1
            cur_theta = self.update_theta_from_a(cur_a, reach_vectors)
        return cur_a, cur_loss, (cur_loss >= prev_loss - self._min_improvement)

    def _fit(self) -> None:
        self._define_criteria()
        self._fit_multi_init_a()

    def by_impressions(
        self, impressions: Iterable[int], max_frequency: int = 1
    ) -> ReachPoint:

        reach_vector = self.get_reach_vector(impressions)
        reach = 0
        for j in range(self._c):
            w = 1
            for i in range(self._p):
                w *= 1 - ((self.a[j + i * self._c] * reach_vector[i]) / self._N)
            reach += 1 - w
        reach *= self._N
        return ReachPoint(impressions, [reach])
