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
"""Base class for modeling a reach curve.

A reach curve is a mapping from scalar spend or impression value to reach.
"""

import copy
from typing import Callable

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface


class ReachCurve(ReachSurface):
    """Models reach as a function of scalar spend or impressions."""

    def __init__(self, data: [ReachPoint], max_reach: int = None):
        """Constructor

        Args:
          data:  A list of ReachPoints to which the model is to be fit.
          max_reach:  Optional.  If specified, the maximum possible reach that can
            be achieved.
        """
        super().__init__(data, max_reach)
        if len(data[0].impressions) != 1:
            raise ValueError("Attempt to create reach curve from multidimensional data")

    def _by_quantile(
        self, quantile: float, reach_function: Callable[[int], ReachPoint]
    ) -> int:
        """Returns a value needed to achieve a given quantile of 1+ reach.

        Args:
          quantile: float, a number in (0,1).
            For many reach models, the maximum reach is an asymptote and hence
            can never be exactly achieved.
          reach_function: A function that maps some quantity, typically either
            impressions or spend, to a corresponding ReachPoint.
        Returns:
          The number of impressions that would have to be delivered in order
          to achieve that fraction of the maximum possible reach.
        """
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        lower, upper = 0, 1
        target_reach = self.max_reach * quantile
        # Establishes invariant that reach(lower) <= target_reach < reach(upper)
        niter = 0
        while reach_function([upper]).reach() <= target_reach:
            lower, upper = upper, 2 * upper
            niter += 1
            if niter > 60:
                # After 60 iterations, upper will equal 2**60, which seems far
                # larger than any value that we would ever encounter in practice.
                raise OverflowError("Maximum reach is not achievable")
        # Maintains invariant that reach(lower) <= target_reach < reach(upper)
        while upper - lower > 1:
            mid = (upper + lower) // 2
            if reach_function([mid]).reach() <= target_reach:
                lower = mid
            else:
                upper = mid
        return lower

    def impressions_at_reach_quantile(self, quantile: float) -> int:
        """Returns the number of impressions for a given quantile of 1+ reach.

        Args:
          quantile: float, a number in (0, 1).
            For many reach models, the maximum reach is an asymptote and hence
            can never be exactly achieved.
        Returns:
          The number of impressions that would have to be delivered in order
          to achieve that fraction of the maximum possible reach.
        """
        return self._by_quantile(quantile, self.by_impressions)

    def spend_at_reach_quantile(self, quantile: float) -> int:
        """Returns the required spend to achieve a given quantile of 1+ reach.

        Args:
          quantile: float, a number between 0 and 1, excluding the endpoints.
            For many reach models, the maximum reach is an asymptote and hence
            can never be exactly achieved.
        Returns:
          The required spend in order to achieve that fraction of the maximum
          possible reach.
        """
        return self._by_quantile(quantile, self.by_spend)

    def impressions_for_spend(self, spend: float) -> int:
        """Converts spend to impressions."""
        raise NotImplementedError()
