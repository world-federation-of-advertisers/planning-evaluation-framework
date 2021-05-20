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


class RestrictedPairwiseUnionReachSurface(ReachSurface):
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
