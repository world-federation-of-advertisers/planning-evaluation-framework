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
"""Models a single publiser. """

from typing import Dict
from typing import List
from typing import NamedTuple

from wfa_cardinality_estimation_evaluation_framework.estimators.same_key_aggregator import (
    ExponentialSameKeyAggregator,
)
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import (
    VectorOfCounts,
)

from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyTracker
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    SystemParameters,
)


class Publisher:
    def __init__(
        self,
        publisher_data: PublisherData,
        publisher_index: int,
        params: SystemParameters,
        privacy_tracker: PrivacyTracker,
    ):
        """Constructs a model for a single publisher and campaign.

        Args:
          publisher_data:  The source of impression data for
            this publisher.
          publisher_index:  The index of this publisher within
            the list of all publishers known to Halo.
          params:  Configuration parameters for this halo instance.
          privacy_tracker:  PrivacyTracker object that will be used
            to track the consumption of differential privacy by this
            publisher.
        """
        self._publisher_data = publisher_data
        self._publisher_index = publisher_index
        self._campaign_spend = (
            params.campaign_spend_fractions[publisher_index] * publisher_data.max_spend
        )
        self._params = params

    @property
    def campaign_spend(self) -> float:
        """Returns the amount spent on this campaign."""
        return self._campaign_spend

    @property
    def max_spend(self) -> float:
        """Returns the maximum amount that can be spent with this publisher."""
        return self._publisher_data.max_spend

    def true_reach_by_spend(self, spend: float, max_frequency: int = 1) -> ReachPoint:
        """Returns the true reach obtained for a given spend vector.

        Args:
            spend:  The hypothetical amount spent.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            A ReachPoint representing the true reach that would have been
            obtained for this spend.
        """
        user_counts = self._publisher_data.user_counts_by_spend(spend)
        impressions = sum(user_counts.values())
        reach = ReachPoint.user_counts_to_kplus_reaches(user_counts, max_frequency)
        return ReachPoint([impressions], reach, [spend])

    def liquid_legions_sketch(self, spend: float) -> ExponentialSameKeyAggregator:
        """Returns the LiquidLegions sketch associated with a given spend.
        Note that the returned data structure is not differentially private.
        """
        sketch = ExponentialSameKeyAggregator(
            length=int(self._params.liquid_legions.sketch_size),
            decay_rate=self._params.liquid_legions.decay_rate,
            random_seed=self._params.liquid_legions.random_seed,
        )
        spend = min(spend, self._campaign_spend)
        for id, freq in self._publisher_data.user_counts_by_spend(spend).items():
            sketch.add_ids([id] * freq)
        return sketch

    def dp_reach_curve(self, epsilon: float, delta: float) -> ReachCurve:
        """Returns a differentially private reach curve model.

        Args:
          epsilon:  The amount of privacy budget that can be used.
          delta:    The addition portion of the amount of privacy budget
             that can be used.
        """
        raise NotImplementedError()

    def dp_vector_of_counts(self, epsilon: float, delta: float) -> VectorOfCounts:
        """Returns a differentially private VectorOfCounts for the campaign.

        Args:
          epsilon:  The amount of privacy budget that can be used.
          delta:    The addition portion of the amount of privacy budget
             that can be used.
        """
        raise NotImplementedError()
