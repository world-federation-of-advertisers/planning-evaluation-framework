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
"""Class for modeling Pairwise union reach surface."""

import copy
from typing import List
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from typing import Iterable
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve

solvers.options["show_progress"] = False


class GeneralizedMixtureReachSurface(ReachSurface):
    """Models reach with the generalized mixture overlap model.

    GeneralizedMixtureReachSurface models the combined reach with the formula

    ...

    where:
          p is the number of publishers
          a_i_j are the model parameters we want to estimate
          imp_i is the numnber of impressions for ith publisher
          r_j(imp_j) is the reach of the jth publisher for imp_j impressions.

    reference doc:  ###
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
            p(p-1)/2 points, where p is the number of publishers. Otherwise,
            there is no guarantee of a unique solution. The reach points
            represent arbitrary points on the reach surface, involving spend at
            multiple advertisers
        """

        self._reach_curves = copy.deepcopy(reach_curves)
        self._n = len(reach_points)
        self._p = len(reach_points[0].impressions)
        self._c = num_clusters
        self._n_iter = 1000
        super().__init__(data=reach_points)

    def update_theta_from_a(self, a, r):
        """Udpate theta using equation (TODO: add formula somewhere).

        Args:
          a: a length (p * c) vector. Current best guess of a.
          r: a length n list of length p vectors. Single publisher reaches for
            each training point.

        Returns:
          A length n list of length c vectors that indicate each
          $theta_j^{(k)}$.
        """
        theta = []
        for k in range(n):
            theta_k = np.zeros(c)
            for j in range(c):
                x = a[[j + i * c for i in range(p)]] * r[k] / N
                theta_k[j] = (1 - np.prod(1 - x)) / np.sum(x)
            theta.append(theta_k)
        return theta

    def update_a_from_theta(self, theta, r, y):
        """Udpate theta using equation (TODO: add formula somewhere).

        Args:
          theta: a length n list of length c vectors. Current best guess of
            theta.
          r: a length n list of length p vectors. Single publisher reaches for
            each training point.
          y: a length n list of responses.

        Returns:
          A length (p * c) vector indicating each $a_{ij}$.
        """
        z = self.get_z_from_theta(theta, r)
        return solve_a_given_z(z, y)

    def get_z_from_theta(self, theta, r):
        z = []
        for k in range(n):
            z_k = np.zeros(p * c)
            for i in range(p):
                for j in range(c):
                    z_k[j + i * c] = r[k][i] * theta[k][j] / N
            z.append(z_k)
        return z

    def _fit(self) -> None:
        initial_a = np.random.dirichlet(np.ones(self._c) / 5, size=self._p).flatten()
        cur_a = initial_a.copy()
        cur_theta = self.update_theta_from_a(cur_a, r)
        for _ in range(self._n_iter):
            z = self.get_z_from_theta(cur_theta, r)
            cur_a = self.update_a_from_theta(cur_theta, r, true_y)
            cur_theta = self.update_theta_from_a(cur_a, r)

    def by_spend(self, spend: Iterable[float], max_frequency: int = 1) -> ReachPoint:
        return

    def get_reach_vector(self, impressions: Iterable[int]) -> Iterable[int]:
        """Calculates single publisher reaches for a given impression vector.

        Args:
          impressions: A list of impressions per publisher.

        Returns:
          A list R of length p. The value R[i] is the reach achieved on
          publisher i when impressions[i] impressions are shown.
        """

        return

    def by_impressions(
        self, impressions: Iterable[int], max_frequency: int = 1
    ) -> ReachPoint:
        y = []
        for k in range(n):
            y_k = 0
            for j in range(c):
                w = 1
                for i in range(p):
                    w *= 1 - self.a[j + i * c] * r[k][i] / N
                y_k += 1 - w
            y.append(y_k)
        return y
