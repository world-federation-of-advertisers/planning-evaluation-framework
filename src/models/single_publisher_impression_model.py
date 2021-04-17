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
"""Base class for modeling single pub reach from impression counts.

A single publisher impression model is a function that maps impression
counts to number of users reached.  To fit the model, one or more points
on the reach curve are provided.
"""

from typing import NamedTuple


class ReachPoint(NamedTuple):
  """A single point on a reach curve.

  impressions:  The number of impressions that were shown.
  reach_at_frequency: A tuple of values representing the number of people
    reached at each frequency.  reach_at_frequency[i] is the number of
    people who were reached AT LEAST i+1 times.
  """
  impressions: int
  reach_at_frequency: list[int]


class ImpressionsToReachModel:
  """Models single-publisher reach as a function of impressions."""

  def __init__(self, data: [ReachPoint], max_reach: int = None):
    """Constructs an ImpressionsToReachModel.

    Args:
      data:  A list of ReachPoints to which the model is to be fit.
      max_reach:  Optional.  If specified, the maximum possible reach that can
        be achieved.
    """
    if not data:
      raise ValueError("At least one ReachPoint must be specified")
    self.data = data.copy()
    self._max_reach = max_reach
    self._fit()

  def _fit(self) -> None:
    """Fits a model to the data that was provided in the constructor."""
    raise NotImplementedError()

  def reach(self, impressions: int) -> int:
    """Returns the estimated reach for a given number of impressions."""
    return self.frequencies(impressions, 1)[0]

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
    raise NotImplementedError()

  @property
  def max_reach(self) -> int:
    """Returns the max number of people that can potentially be reached."""
    return self._max_reach

  def impressions_at_reach_quantile(self, quantile: float) -> int:
    """Returns the number of impressions for a given quantile of reach.

    Args:
      quantile: float, a number between 0 and 1.
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
    while self.reach(upper) <= target_reach:
      lower, upper = upper, 2 * upper
      niter += 1
      if niter > 60:
        raise OverflowError("Maximum reach is not achievable")
    # Maintains invariant that reach(lower) <= target_reach < reach(upper)
    while upper - lower > 1:
      mid = (upper + lower) // 2
      if self.reach(mid) <= target_reach:
        lower = mid
      else:
        upper = mid
    return lower
