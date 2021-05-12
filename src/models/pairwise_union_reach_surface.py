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
from typing import Iterable
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface


class PaiwiseUnionReachSurface(ReachSurface):
    """Models reach as a function of impressions or spend."""

    def __init__(self, data: Iterable[ReachPoint], max_reach: int = None):
        """Constructor

        Args:
          data:  A list of ReachPoints to which the model is to be fit.
          max_reach:  Optional.  If specified, the maximum possible reach that can
            be achieved.
        """
        pass

    def _fit(self) -> None:
        """Fits a model to the data that was provided in the constructor."""
        raise NotImplementedError()

    def by_impressions(
        self, impressions: Iterable[int], max_frequency: int = 1
    ) -> ReachPoint:
        """Returns the estimated reach for a given impression vector."""
        raise NotImplementedError()

    def by_spend(self, spend: Iterable[float], max_frequency: int = 1) -> ReachPoint:
        """Returns the estimated reach for a given spend vector."""
        raise NotImplementedError()

    @property
    def max_reach(self) -> int:
        """Returns the max number of people that can potentially be reached."""
        return self._max_reach
