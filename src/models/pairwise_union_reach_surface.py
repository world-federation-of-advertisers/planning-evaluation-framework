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


class PairwiseUnionReachSurface(ReachSurface):
    """Models reach with the pairwise union overlap model.

    PairwiseUnionReachSurface models the combined reach with the formula

    r_1(imp_1)+r_2(imp_2)+...+r_p(imp_p) -
      1/2 sum_{j=1..p} sum_{k=1..p}
      (a_{j,k} * {r_j(imp_j)r_k(imp_k)} / {max\{m_j,m_k}})

    where:
          p is the number of publishers
          a_i_j are the model parameters we want to estimate
          imp_i is the numnber of impressions for ith publisher
          r_j(imp_j) is the reach of the jth publisher for imp_j impressions.

    reference doc:
    https://docs.google.com/document/d/1YEEXv8xLBOZ69dWCdyVgyRkmCKu5Wa3AwFfFSFqKFBg/edit?resourcekey=0-NpbIK7OYXgWQCJgG3YAgYA#heading=h.cg5r5tr0crku


    """

    def __init__(
        self, reach_curves: Iterable[ReachCurve], reach_points: Iterable[ReachPoint]
    ):
        """Constructor for PaiwiseUnionReachSurface.

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
        super().__init__(data=reach_points)

    def _fit(self) -> None:
        z, alpha = self._get_z_and_alpha()
        self._a = self._solve_a_given_z_and_alpha(z, alpha)

    def by_spend(self, spend: Iterable[float], max_frequency: int = 1) -> ReachPoint:
        return self.by_impressions(
            [
                curve.impressions_for_spend(pub_spend)
                for curve, pub_spend in zip(self._reach_curves, spend)
            ],
            max_frequency,
        )

    def get_reach_vector(self, impressions: Iterable[int]) -> Iterable[int]:
        """Calculates single publisher reaches for a given impression vector.

        Args:
          impressions: A list of impressions per publisher.

        Returns:
          A list R of length p. The value R[i] is the reach achieved on
          publisher i when impressions[i] impressions are shown.
        """

        return [
            reach_curve.by_impressions([impression]).reach()
            for reach_curve, impression in zip(self._reach_curves, impressions)
        ]

    def by_impressions(
        self, impressions: Iterable[int], max_frequency: int = 1
    ) -> ReachPoint:

        reach_vector = self.get_reach_vector(impressions)
        reach_sum = sum(reach_vector)
        overlap = sum(
            [
                (self._a[i * self._p + j] * reach_vector[i] * reach_vector[j])
                / (
                    max(
                        self._reach_curves[i].max_reach, self._reach_curves[j].max_reach
                    )
                    * 2
                )
                for i in range(self._p)
                for j in range(self._p)
            ]
        )

        return ReachPoint(impressions, [reach_sum - overlap])

    def _get_inequality_constraints(self):
        """Returns the inequality constraint for the Quadratic Programming solver.

        3 types of inequality constraints are stacked in matrix G and vector h:

          G1: a_i_j >= 0 for all i,j -> all parameters are non negative
          G2: sum_{j=1..p} a_i_j <= 1 for all i -> row sums are no more than one
          G3: sum_{i=1..p} a_i_j <= 1 for all j -> col sums are no more than one

        Returns:
          (matrix G, vector h) cvxopt.solvers recognizes the constraint
          Gx <= h, where x is the parameter vector.
        """
        G1 = -matrix(np.eye(self._p * self._p))
        G2 = np.kron(np.eye(self._p), np.ones(self._p))
        G3 = np.kron(np.ones(self._p), np.eye(self._p))
        h = matrix(
            np.concatenate(
                (np.zeros(self._p * self._p), np.ones(2 * self._p)), axis=None
            )
        )
        return matrix(np.vstack((G1, G2, G3))), h

    def _get_equality_constraints(self):
        """Returns the equality constraint for the Quadratic Programming solver.

        2 types of equality constraint is described by matrix A and vector b:

          A1: a_i_i = 0 for all i -> publishers don't overlap with themselves.
          A2: a_i_j = a_j_i -> pairwise parameters are symteric.

        Returns:
            (matrix A, vector b) cvxopt.solvers recognizes the constraint
            Ax = b, where x is the parameter vector.
        """

        A1 = np.zeros(shape=(self._p, self._p * self._p))
        for i in range(self._p):
            A1[i, i * self._p + i] = 1

        num_symmetry_constraints = int(((self._p * self._p) - self._p) / 2)
        A2 = np.zeros(shape=(num_symmetry_constraints, self._p * self._p))
        constraint_row = 0
        for i in range(self._p):
            for j in range(i + 1, self._p):
                A2[constraint_row, i * self._p + j] = 1
                A2[constraint_row, j * self._p + i] = -1
                constraint_row += 1
        b = matrix(
            np.concatenate(
                (np.zeros(self._p), np.zeros(num_symmetry_constraints)),
                axis=None,
            )
        )
        return matrix(np.vstack((A1, A2))), matrix(b)

    def _solve_a_given_z_and_alpha(self, z: List[List[int]], alpha: List[List[int]]):
        """Optimize for matrix a given z and alpha using Quadratic Programming.

        argmin_a sum_{k=1..n} [y^{k} - alpha^{k} - a^T z^{k}]^2 =
                 sum (a^T z^{k})^2 - 2 \sum [y^{k} - z^{k}] a^T z^{k} + ...

        expanded,
          P = sum_{k=1..n} z^{k} [z^{k}]^T,
          and
          q = - sum_{k=1..n} [y^{k} - alpha^{k}] z^{k},

        and we solve

          argmin_a (1/2) x^T P a + q^T x

          s.t. Gx <= h
               and
               Ax=b,

        Args:
          z: a length-n list of length-(p * p) vectors.
          alpha: a length-n list of scalars.

        Returns:
          a length-(p * p) vector of the optimization problem above
        """

        P = matrix(
            sum(
                [
                    np.matmul(
                        z[k].reshape(self._p * self._p, 1),
                        z[k].reshape(1, self._p * self._p),
                    )
                    for k in range(self._n)
                ]
            )
        )
        q = -matrix(
            sum((self._data[k].reach() - alpha[k]) * z[k] for k in range(self._n))
        )

        G, h = self._get_inequality_constraints()
        A, b = self._get_equality_constraints()
        res = solvers.qp(P, q, G, h, A, b)

        return np.array(res["x"]).flatten()

    def _get_z_and_alpha(self):
        """Get intermidiate quantities z and alpha.

        Returns:
          Tuple of z and alpha.
          z: Length-n list of length-(p * p) vectors. Each vector is flattened
          version of the matrix that denotes pairwise non-overlapped regions.
          alpha: Length-n list of scalars. Each scalar denotes the overlapped
          reach
          for that reach point.
        """
        reach_vectors = np.array(
            [
                self.get_reach_vector(reach_point.impressions)
                for reach_point in self._data
            ]
        )

        z = [
            -np.array(
                [
                    (reach_vectors[k][i] * reach_vectors[k][j])
                    / (
                        max(
                            self._reach_curves[i].max_reach,
                            self._reach_curves[j].max_reach,
                        )
                        * 2
                    )
                    for i in range(self._p)
                    for j in range(self._p)
                ]
            )
            for k in range(self._n)
        ]
        alpha = [sum(reach_vector) for reach_vector in reach_vectors]
        return z, alpha
