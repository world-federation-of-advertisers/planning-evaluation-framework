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


class PrivacyBudget(NamedTuple):
    """Specifies the amount of privacy budget an operation is allowed to use.

    Equivalently, indirectly specifies the amount of noise that will be added
    to the results of an operation.

    An algorithm A is (epsilon, delta) differentially private, if when
    databases D_1 and D_2 differ by only a single row, the probability
    that A outputs x on D_1 is related to the probability that A outputs
    x on D_2 by the relation:

      Pr(x | D_1) \leq exp(epsilon) Pr(x | D_2) + delta.

    """

    epsilon: float
    delta: float


class NoisingEvent(NamedTuple):
    """Records the addition of differentially private noise

    This object records the addition of differentially private noise
    to data.  It can represent adding noise to a single quantity,
    such as adding Laplace noise to the total cardinality of a set,
    or it can represent the addition of noise in parallel to a collection
    of closely related but non-overlapping quantities, such as adding
    discrete Laplace noise to the buckets of a histogram.
    """

    budget: PrivacyBudget  # Privacy budget associated to this event.
    mechanism: str  # See DP_NOISE_MECHANISM above.
    params: Dict  # Mechanism-specific parameters.


class PrivacyTracker:
    """A class for tracking and reporting privacy budget usage.

    TODO #1: This implementation does not report privacy consumption when
    using the advanced composition rule or when using privacy loss
    distributions.

    TODO #2: Extend this code to allow for tracking of per-EDP privacy budgets.

    TODO #3: Extend this code to track privacy usage by partition.  
    In other words, if one noising event applied to the M25-34 demo group 
    and a second noising event applied to the F35-44 demo group, then by 
    using the parallel composition rule, the total noise applied would be 
    the maximum of the two events.  Again, this is something that would be 
    useful for workflow evaluation but probably not for model evaluation.
    """

    def __init__(self):
        """Returns an object for recording and tracking privacy budget usage."""
        self._epsilon_sum = 0
        self._delta_sum = 0
        self._noising_events = []  # A list of NoisingEvents

    @property
    def privacy_consumption(self) -> PrivacyBudget:
        """Returns the total privacy budget consumed so far."""
        return PrivacyBudget(self._epsilon_sum, self._delta_sum)

    @property
    def mechanisms(self) -> List[str]:
        """List of mechanisms that have been applied."""
        return [event.mechanism for event in self._noising_events]

    def append(self, event: NoisingEvent) -> None:
        """Records an application of differentially private noise."""
        self._epsilon_sum += event.budget.epsilon
        self._delta_sum += event.budget.delta
        self._noising_events.append(event)
