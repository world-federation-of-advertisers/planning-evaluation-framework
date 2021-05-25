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
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from typing import Iterable
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve


class PairwiseUnionReachSurface(ReachSurface):
  """Models reach with the pairwise union overlap model."""

  def __init__(self, reach_curves: Iterable[ReachCurve],
               reach_points: Iterable[ReachPoint]):
    """Constructor for PaiwiseUnionReachSurface.

    Args:
      reach_curves: A list of ReachCurves to be used in model fitting and reach
        prediction.
      reach_points: A list of ReachPoints to which the model is to be fit. This
        is parallel to the reach_curves list. The reach point at ith poisition
        is drawn from ith reach curve.
    """

    self._reach_curves = copy.deepcopy(reach_curves)
    self._n = len(reach_points)
    self._p = len(reach_points[0].impressions)
    super().__init__(data=reach_points, max_reach=0)

  def _fit(self) -> None:
    z, alpha = self.get_z_and_alpha()
    self._a = self.solve_a_given_z_and_alpha(z, alpha)

  def get_reach_vector(self, impressions: Iterable[int]) -> Iterable[int]:
    return [
        reach_curve.by_impressions(impression).reach()
        for reach_curve, impression in zip(self._reach_curves, impressions)
    ]

  def by_impressions(self,
                     impressions: Iterable[int],
                     max_frequency: int = 1) -> ReachPoint:

    reach_vector = self.get_reach_vector(impressions)
    reach_sum = sum(reach_vector)
    overlap = sum([
        (self._a[i * self._p + j] * reach_vector[i] * reach_vector[j]) /
        (max(self._reach_curves[i].max_reach, self._reach_curves[j].max_reach) *
         2) for i in range(self._p) for j in range(self._p)
    ])

    return ReachPoint(impressions, [reach_sum - overlap])

  def get_constraints(self):
    G1 = -matrix(np.eye(self._p * self._p))
    G2 = np.kron(np.eye(self._p), np.ones(self._p))
    G3 = np.kron(np.ones(self._p), np.eye(self._p))
    h = matrix(
        np.concatenate((np.zeros(self._p * self._p), np.ones(2 * self._p)),
                       axis=None))
    return matrix(np.vstack((G1, G2, G3))), h

  def construct_matrix_a(self):
    A = np.zeros(shape=(self._p, self._p * self._p))
    for i in range(self._p):
      A[i, i * self._p + i] = 1
    return matrix(A)

  def solve_a_given_z_and_alpha(self, z, alpha):
    """Optimize for matrix a given z and alpha.

    Args:
      z: a length-n list of length-(p * p) vectors.
      alpha: a length-n list of scalars.

    Returns:
      a length-(p * p) vector of the optimization problem above
    """

    P = matrix(
        sum([
            np.matmul(z[k].reshape(self._p * self._p, 1),
                      z[k].reshape(1, self._p * self._p))
            for k in range(self._n)
        ]))
    q = -matrix(
        sum((self._data[k].reach() - alpha[k]) * z[k] for k in range(self._n)))

    G, h = self.get_constraints()
    A = self.construct_matrix_a()
    b = matrix(np.zeros(self._p))
    res = solvers.qp(P, q, G, h, A, b)

    return np.array(res['x']).flatten()

  def get_z_and_alpha(self):
    """Get intermidiate quantities z and alpha.

    Returns:
      Tuple of z and alpha.
      z: Length-n list of length-(p * p) vectors. Each vector is flattened
      version of the matrix that denotes pairwise overlap regions.
      alpha: Length-n list of scalars. Each scalar denotes the overlapped reach
      for that reach point.
    """
    reach_vectors = np.array([
        self.get_reach_vector(reach_point.impressions)
        for reach_point in self._data
    ])

    z = [
        -np.array([(reach_vectors[k][i] * reach_vectors[k][j]) /
                   (max(self._reach_curves[i].max_reach,
                        self._reach_curves[j].max_reach) * 2)
                   for i in range(self._p)
                   for j in range(self._p)])
        for k in range(self._n)
    ]
    alpha = [sum(reach_vector) for reach_vector in reach_vectors]
    return z, alpha