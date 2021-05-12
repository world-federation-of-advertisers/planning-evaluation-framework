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
"""Base class for modeling a reach surface.

A reach surface is a mapping from a spend or impression vector to reach.
"""

import copy
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from typing import Iterable
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface



class PaiwiseUnionReachSurface(ReachSurface):
  """Models reach with the pairwise union overlap model."""

  def _fit(self) -> None:
    raise NotImplementedError()

  def by_impressions(self,
                     impressions: Iterable[int],
                     max_frequency: int = 1) -> ReachPoint:
    
    raise NotImplementedError()
