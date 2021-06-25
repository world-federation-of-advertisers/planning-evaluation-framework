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
"""George Goerg's one-point reach curve model.

This class fits the model described in George Goerg's paper:

  Goerg, Georg M. "Estimating reach curves from one data point." (2014).

Goerg assumes the underlying reach curve is determined by an exponential-
Poisson distribution with unknown mixing parameter, and shows how the
reach curve can be extrapolated from a single point on it.
"""

import warnings

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve


class GoergModel(ReachCurve):
    """Goerg single-point reach curve model."""

    def __init__(self, data: [ReachPoint]):
        """Constructs an Goerg single point reach model.

        Args:
          data:  A list of ReachPoints to which the model is to be fit.
          max_reach:  Optional.  If specified, the maximum possible reach that can
            be achieved.
        """
        if len(data) != 1:
            raise ValueError("Exactly one ReachPoint must be specified")
        self._impressions = data[0].impressions[0]
        self._reach = data[0].reach(1)
        self._fit()
        self._max_reach = self._rho
        if data[0].spends:
            self._cpi = data[0].spends[0] / data[0].impressions[0]
        else:
            self._cpi = None

    def _fit(self) -> None:
        """Fits a model to the data that was provided in the constructor."""
        if self._impressions == self._reach:
            # In this corner case, there will be a division by zero error if
            # we estimate rho using the formula. This error will block the rest
            # of evaluation. To avoid blocking the rest of evaluation,
            # we will assign rho a hard-coded maximum value, which is 100 here.
            # TODO(jiayu): Think about the choice of max value or alternatives.
            warnings.warn(
                "impression = reach. rho is assigned a hard-coded maximum value."
            )
            self._rho = 100
        else:
            self._rho = (self._impressions * self._reach) / (
                self._impressions - self._reach
            )
        self._beta = self._rho

    def by_impressions(self, impressions: [int], max_frequency: int = 1) -> ReachPoint:
        """Returns the estimated reach as a function of impressions.

        Args:
          impressions: list of ints of length 1, specifying the hypothetical number
            of impressions that are shown.
          max_frequency: int, specifies the number of frequencies for which reach
            will be reported.
        Returns:
          A ReachPoint specifying the estimated reach for this number of impressions.
        """
        if len(impressions) != 1:
            raise ValueError("Impressions vector must have a length of 1.")
        kplus_reach_list = []
        for k in range(1, max_frequency + 1):
            kplus_reach = (
                self._rho * (impressions[0] / (impressions[0] + self._beta)) ** k
            )
            kplus_reach_list.append(kplus_reach)
        if self._cpi:
            spend = impressions[0] * self._cpi
            return ReachPoint(impressions, kplus_reach_list, [spend])
        else:
            return ReachPoint(impressions, kplus_reach_list)

    def by_spend(self, spends: [int], max_frequency: int = 1) -> ReachPoint:
        """Returns the estimated reach as a function of spend assuming constant CPM

        Args:
          spend: list of floats of length 1, specifying the hypothetical spend.
          max_frequency: int, specifies the number of frequencies for which reach
            will be reported.
        Returns:
          A ReachPoint specifying the estimated reach for this number of impressions.
        """
        if not self._cpi:
            raise ValueError("Impression cost is not known for this ReachPoint.")
        if len(spends) != 1:
            raise ValueError("Spend vector must have a length of 1.")
        return self.by_impressions([spends[0] / self._cpi], max_frequency)
