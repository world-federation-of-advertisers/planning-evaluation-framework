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


class RestrictedPairwiseUnionReachSurface(PairwiseUnionReachSurface):
  """Models reach with the pairwise union overlap model."""

  def __init__(self, reach_curves: Iterable[ReachCurve],
               reach_points: Iterable[ReachPoint]):
    """Constructor for RestrictedPaiwiseUnionReachSurface.

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
    res = minimize(
        fun=lambda x: self.loss(x),
        x0=np.array([0] * self._p),
        constraints=self.get_constraints())
    self.construct_a_from_lambda(res['x'])

  def construct_a_from_lambda(self, lbd):
    """Get value of flattened a matrix from lamdas.

    Args:
      lbd: a length p vector indicating lambda_i for each pub.

    Returns:
      the value of flattened a matrix.
    """
    self._a = np.ones(self._p * self._p)
    for i in range(self._p):
      for j in range(self._p):
        self._a[i * self._p + j] = lbd[i] * lbd[j]

  def get_constraints(self):
   """Get constraints to be used in optimization.

    Returns:
      the list of constraint functions
    """
    cons = []
    for i in range(self._p):
      # All lambdas are non negative : lbd[i] >= 0
      cons.append({'type': 'ineq', 'fun': lambda x: x[i]})
      # Lambda j sum times Lambda i is less than 1 : 1 - lbd[i] * sum(lbd) >= 0
      cons.append({'type': 'ineq', 'fun': lambda x: 1 - x[i] * sum(x)})
    return cons

  def fitted(self, lbd, reach_vector):
    """Get value of fitted union reach.

    Args:
      lbd: a length p vector indicating lambda_i for each pub.
      reach_vector: a length p vector indicating the single-pub reach of each
        pub at a single data point.
      u: a length p vector indicating the universe size of each pub.

    Returns:
      the value of fitted union reach.
    """
    reach_sum = sum(reach_vector)
    overlap = sum([
        (lbd[i] * lbd[j] * reach_vector[i] * reach_vector[j]) /
        (max(self._reach_curves[i].max_reach, self._reach_curves[j].max_reach) *
         2) for i in range(self._p - 1) for j in range(i, self._p)
    ])
    return reach_sum - overlap

  def loss(self, lbd):
    """Get value of loss function.

    Args:
      lbd: a length p vector indicating lambda_i for each pub.

    Returns:
      the value of fitted union reach.
    """
    reach_vectors = np.array([
        self.get_reach_vector(reach_point.impressions)
        for reach_point in self._data
    ])
    return sum([(self._data[k] - fitted(lbd, reach_vectors[k]))**2
               for k in range(self._n)])
