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


class SamplingBucketIndices(NamedTuple):
    """Records the indices of sampling buckets.

    For simplicity, we represent buckets as an interval [0, 1). A user is
    mapped to a real number in this interval. A sampling bucket represents an
    interval [smallest_index, smallest_index + sampling_rate). This means that
    all users that get mapped to this interval is used for computing
    differentially private estimates.
    """

    smallest_index: float
    sampling_rate: float

    def contains(self, sampling_bucket: float) -> bool:
        """Returns whether the sampling bucket is contained in the interval."""
        return (
            self.smallest_index <= sampling_bucket
            and sampling_bucket < self.smallest_index + self.sampling_rate
        )


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
    # See SamplingBucketIndices above.
    sampling_buckets: SamplingBucketIndices = SamplingBucketIndices(0, 1)


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
        self._noising_events = []  # A list of NoisingEvents
        # A list of starting points of buckets
        self._sampling_buckets_starting_points = set([])

    @property
    def privacy_consumption(self) -> PrivacyBudget:
        """Returns the total privacy budget consumed so far.

        Total privacy consumption is currently computed using the
        basic composition rule.  This will be expanded in the future
        to support advanced composition (see TODO #1 above).
        """
        # Take maximum privacy budget spent across all buckets.
        max_epsilon = 0
        max_delta = 0
        for sampling_bucket_ in self._sampling_buckets_starting_points:
            consumed_budget = self.privacy_consumption_for_sampling_bucket(
                sampling_bucket_
            )
            max_epsilon = max(max_epsilon, consumed_budget.epsilon)
            max_delta = max(max_delta, consumed_budget.delta)
        return PrivacyBudget(max_epsilon, max_delta)

    def privacy_consumption_for_sampling_bucket(self, sampling_bucket) -> PrivacyBudget:
        """Returns the total privacy budget consumed so far for a given sampling bucket."""
        epsilon_sum = 0
        delta_sum = 0
        for event in self._noising_events:
            if event.sampling_buckets.contains(sampling_bucket):
                epsilon_sum += event.budget.epsilon
                delta_sum += event.budget.delta
        return PrivacyBudget(epsilon_sum, delta_sum)

    @property
    def mechanisms(self) -> List[str]:
        """List of mechanisms that have been applied."""
        return [event.mechanism for event in self._noising_events]

    def append(self, event: NoisingEvent) -> None:
        """Records an application of differentially private noise."""
        self._noising_events.append(event)
        self._sampling_buckets_starting_points.add(
            event.sampling_buckets.smallest_index
        )
