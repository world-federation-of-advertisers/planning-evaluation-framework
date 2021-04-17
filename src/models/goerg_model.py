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

from wfa_planning_evaluation_framework.models.single_publisher_impression_model import ReachPoint
from wfa_planning_evaluation_framework.models.single_publisher_impression_model import ImpressionsToReachModel


class GoergModel(ImpressionsToReachModel):
  """Goerg single-point reach curve model."""

  def __init__(self, data: [ReachPoint]):
    """Constructs an ImpressionsToReachModel.

    Args:
      data:  A list of ReachPoints to which the model is to be fit.
      max_reach:  Optional.  If specified, the maximum possible reach that can
        be achieved.
    """
    if len(data) != 1:
      raise ValueError("Exactly one ReachPoint must be specified")
    self._impressions = data[0].impressions
    self._reach = data[0].reach_at_frequency[0]
    self._fit()
    self._max_reach = self._rho

  def _fit(self) -> None:
    """Fits a model to the data that was provided in the constructor."""
    self._rho = ((self._impressions * self._reach) /
                 (self._impressions - self._reach))
    self._beta = self._rho

  def frequencies(self, impressions: int, max_frequency: int) -> int:
    """Returns the estimated reach for frequencies 1..max_frequency.

    Args:
      impressions: int, specifies the hypothetical number of impressions that
        are shown.
      max_frequency: int, specifies the number of frequencies for which reach
        will be reported.
    Returns:
      An array reach[], where reach[i] is the number of people who would be
      reached AT LEAST i+1 times.  The length of the array is equal to
      max_frequency.
    """
    kplus_reach_list = []
    for k in range(1,max_frequency+1):
      kplus_reach = self._rho * (impressions / (impressions + self._beta))**k
      kplus_reach_list.append(kplus_reach)
    return kplus_reach_list
