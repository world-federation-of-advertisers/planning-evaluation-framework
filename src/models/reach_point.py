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
"""Base class for modeling a reach point.

Represents a single point on either a reach curve or a reach surface.
"""

from typing import Iterable


class ReachPoint:
  """A single point on a reach curve."""

  def __init__(self,
               impressions: Iterable[int],
               kplus_frequencies: Iterable[int],
               spend: Iterable[float] = None):
    """Represents a single point on a reach curve.

    Args:
      impressions:  The number of impressions that were served by
        each publisher.  This should be a list
      kplus_frequencies:  A list of values representing the number
        of people reached at various frequencies.  kplus_frequencies[k]
        is the number of people who were reached AT LEAST k+1 times.
      spend:  If given, the amount that was spent at this point on each
        publisher.
    """
    if spend and len(impressions) != len(spend):
      raise ValueError("impressions and spend must have same length")
    self._impressions = tuple(impressions)
    self._kplus_frequencies = tuple(kplus_frequencies)
    if spend:
      self._spend = tuple(spend)
    else:
      self._spend = None

  @property
  def impressions(self):
    """Returns the number of impressions associated with this point."""
    return self._impressions

  def reach(self, k=1):
    """Returns the k+ reach associated with this point."""
    if not 0 < k <= len(self._kplus_frequencies):
      raise ValueError("k {} is out of range. max allowed is {}".format(
          k, len(self._kplus_frequencies)))
    return self._kplus_frequencies[k - 1]

  def frequency(self, k=1):
    """Returns the number of people reached with frequency exactly k."""
    if not 0 < k < len(self._kplus_frequencies):
      raise ValueError("k {} is out of range. max allowed is {}".format(
          k, len(self._kplus_frequencies)))
    return self._kplus_frequencies[k - 1] - self._kplus_frequencies[k]

  @property
  def spend(self):
    """Returns the spend associated with this point."""
    return self._spend
