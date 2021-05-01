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
"""A class for tracking privacy budget usage."""

from typing import Dict
from typing import List
from typing import NamedTuple

DP_NOISE_MECHANISM_LAPLACE = "Laplace"
DP_NOISE_MECHANISM_DISCRETE_LAPLACE = "Discrete Laplace"
DP_NOISE_MECHANISM_GAUSSIAN = "Gaussian"
DP_NOISE_MECHANISM_DISCRETE_GAUSSIAN = "Discrete Gaussian"


class NoisingEvent(NamedTuple):
    """Records the addition of differentially private noise

    This object records the addition of differentially private noise
    to data.  It can represent adding noise to a single quantity,
    such as adding Laplace noise to the total cardinality of a set,
    or it can represent the addition of noise in parallel to a collection
    of closely related but non-overlapping quantities, such as adding
    discrete Laplace noise to the buckets of a histogram.
    """

    epsilon: float  # Amount of budget used in this event.
    delta: float  # Amount of budget used in this event.
    mechanism: str  # See DP_NOISE_MECHANISM above.
    params: Dict  # Mechanism-specific parameters.


class PrivacyTracker:
    """A class for tracking and reporting privacy budget usage.

    NOTE:  This implementation does not report privacy consumption when
    using the advanced composition rule or when using privacy loss
    distributions.  That would be an important addition to this code
    when performing workflow evaluations, but is probably not necessary
    for model evaluation.

    A second way in which this code could be extended is that it
    does not currently allow for tracking of per-EDP privacy budgets.

    A third way in which this code could be extended would be track
    privacy usage by partition.  In other words, if one noising event
    applied to the M25-34 demo group and a second noising event
    applied to the F35-44 demo group, then by using the parallel
    composition rule, the total noise applied would be the maximum of
    the two events.  Again, this is something that would be useful for
    workflow evaluation but probably not for model evaluation.
    """

    def __init__(self):
        """Returns an object for recording and tracking privacy budget usage."""
        self._epsilon_sum = 0
        self._delta_sum = 0
        self._noising_events = []  # A list of NoisingEvents

    @property
    def epsilon(self) -> float:
        """Total epsilon that has been consumed so far."""
        return self._epsilon_sum

    @property
    def delta(self) -> float:
        """Total delta that has been consumed so far."""
        return self._delta_sum

    @property
    def mechanisms(self) -> List[str]:
        """List of mechanisms that have been applied."""
        return [event.mechanism for event in self._noising_events]

    def append(self, event: NoisingEvent) -> None:
        """Records an application of differentially private noise."""
        self._epsilon_sum += event.epsilon
        self._delta_sum += event.delta
        self._noising_events.append(event)
