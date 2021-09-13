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


class ReachSurface:
    """Models reach as a function of impressions or spend."""

    def __init__(self, data: Iterable[ReachPoint], max_reach: int = None):
        """Constructor

        Args:
          data:  A list of ReachPoints to which the model is to be fit.
          max_reach:  Optional.  If specified, the maximum possible reach that
            can be achieved.
        """
        self._data = copy.deepcopy(data)
        if not self._data:
            raise ValueError("At least one ReachPoint must be specified")
        dim = len(self._data[0].impressions)
        if not all([len(point.impressions) == dim for point in data]):
            raise ValueError("Not all input points have the same dimensionality.")
        self._max_reach = max_reach
        self._fit()

    def _fit(self) -> None:
        """Fits a model to the data that was provided in the constructor."""
        raise NotImplementedError()

    def by_impressions(
        self, impressions: Iterable[int], max_frequency: int = 1
    ) -> ReachPoint:
        """Returns the estimated reach for a given impression vector."""
        raise NotImplementedError()

    def by_spend(self, spend: Iterable[float], max_frequency: int = 1) -> ReachPoint:
        return self.by_impressions(
            [
                curve.impressions_for_spend(pub_spend)
                for curve, pub_spend in zip(self._reach_curves, spend)
            ],
            max_frequency,
        )

    @property
    def max_reach(self) -> int:
        """Returns the max number of people that can potentially be reached."""
        return self._max_reach

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
