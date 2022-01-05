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


    N * sum_{j=1..c} [1 - prod_{i=1..p} (1-a_i_j r_i / N)] +epsilon

    where :
          all a_i_j >= 0
          sum_{j=1..c} a_i_j=1 for all i.

    """

    def __init__(
        self,
        reach_curves: Iterable[ReachCurve],
        reach_points: Iterable[ReachPoint],
        num_clusters: int,
        N: int,
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
          num_clusters: number of clusters. Different users in the same cluster
            have the same probability to receive a new impression.
          N : A non-negative value that is at least as large as the maximum
            reach of any of the per-publisher reach functions,
            e.g., N >= reach_curves[i].max_reach for i=1,2,...,p.
        """

        self._reach_curves = copy.deepcopy(reach_curves)
        self._n = len(reach_points)
        self._p = len(reach_points[0].impressions)
        self._c = num_clusters
        self._N = N
        self._n_iter = 10
        super().__init__(data=reach_points)

    def update_theta_from_a(self, a, r):
        """Udpate theta using equation:
        theta(x_1, ... ,x_p) = (1 - prod_{i=1..p} (1- x_i)) / sum_{i=1..p} x_i}

        Args:
          a: a length (p * c) vector. Current best guess of a.
          r: a length n list of length p vectors. Single publisher reaches for
            each training point.

        Returns:
          A length n list of length c vectors that indicate each
          theta_j^{(k)}.
        """
        theta = []
        for k in range(self._n):
            theta_k = np.zeros(self._c)
            for j in range(self._c):
                x = a[[j + i * self._c for i in range(self._p)]] * r[k] / self._N
                theta_k[j] = (1 - np.prod(1 - x)) / np.sum(x)
            theta.append(theta_k)
        return theta

    def solve_a_given_z(self, z, show_status=False):
        """A key intermediate step to fit GM model.

        Args:
          z: a length-n list of length-(p * c) vectors.

        Returns:
          a length-(p * c) vector of the optimization problem above

          𝑃=∑𝑛𝑘=1𝑧(𝑘)[𝑧(𝑘)]⊤ ,  𝑞=−∑𝑛𝑘=1𝑦(𝑘)𝑧(𝑘) ,  𝐺=− np.eye( 𝑝×𝑐 ),
          ℎ= np.zeros(𝑝×𝑐), 𝐴=np.kron(np.eye(c), np.ones(p)), 𝑏=np.ones(c),
          𝑎=(𝑎11,…,𝑎1𝑐,…,𝑎𝑝1,…,𝑎𝑝𝑐), 𝑧=(𝑧11,…,𝑧1𝑐,…,𝑧𝑝1,…,𝑧𝑝𝑐).
        """
        P = matrix(
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
        q = -matrix(sum(self._data[k].reach() * z[k] for k in range(self._n)))
        G = -matrix(np.eye(self._p * self._c))
        h = matrix(np.zeros(self._p * self._c))
        A = matrix(np.kron(np.eye(self._p), np.ones(self._c)))
        b = matrix(np.ones(self._p))
        res = solvers.qp(P, q, G, h, A, b)
        if show_status:
            print(res["status"])
        return np.array(res["x"]).flatten()

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

    def get_z_from_theta(self, theta, r):
        z = []
        for k in range(self._n):
            z_k = np.zeros(self._p * self._c)
            for i in range(self._p):
                for j in range(self._c):
                    z_k[j + i * self._c] = r[k][i] * theta[k][j] / self._N
            z.append(z_k)
        return z

    def _fit(self) -> None:
        reach_vectors = np.array(
            [
                self.get_reach_vector(reach_point.impressions)
                for reach_point in self._data
            ]
        )
        cur_a = np.random.dirichlet(np.ones(self._c) / 5, size=self._p).flatten()
        cur_theta = self.update_theta_from_a(cur_a, reach_vectors)
        for _ in range(self._n_iter):
            z = self.get_z_from_theta(cur_theta, reach_vectors)
            cur_a = self.update_a_from_theta(cur_theta, reach_vectors)
            cur_theta = self.update_theta_from_a(cur_a, reach_vectors)
        self.a = cur_a

    def by_spend(self, spend: Iterable[float], max_frequency: int = 1) -> ReachPoint:
        return

    def by_impressions(
        self, impressions: Iterable[int], max_frequency: int = 1
    ) -> ReachPoint:

        reach_vector = self.get_reach_vector(impressions)
        reach = 0
        for j in range(self._c):
            w = 1
            for i in range(self._p):
                w *= 1 - self.a[j + i * self._c] * reach_vector[i] / self._N
            reach += 1 - w
        return ReachPoint(impressions, [reach])
