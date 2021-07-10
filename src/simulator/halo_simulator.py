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
"""Simulates the Halo system.

This code simulates the part of the Halo system from the Measurement
Coordinator downwards.  This includes the SMPC and the publishers.
The Halo Simulator represents the functionality and interface
that is assumed to exist by the modeling strategies that are examined as
part of this evaluation effort.
"""

from typing import List
from typing import Set
from typing import Tuple
from typing import Dict
import copy
import collections
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.same_key_aggregator import (
    StandardizedHistogramEstimator,
)
from wfa_cardinality_estimation_evaluation_framework.common.noisers import (
    LaplaceMechanism,
)
from wfa_cardinality_estimation_evaluation_framework.estimators.vector_of_counts import (
    VectorOfCounts,
)
from wfa_cardinality_estimation_evaluation_framework.estimators.estimator_noisers import (
    GeometricEstimateNoiser,
)

from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    DP_NOISE_MECHANISM_DISCRETE_GAUSSIAN,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import NoisingEvent
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyBudget
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyTracker
from wfa_planning_evaluation_framework.simulator.publisher import Publisher
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    SystemParameters,
)


MAX_ACTIVE_PUBLISHERS = 20


class HaloSimulator:
    """Simulator for the Halo System.

    A Halo Simulator simulates the Halo System from the Measurement Coordinator
    downwards.  In particular, it simulates the SMPC and the behavior of the
    publishers.  The Halo Simulator represents the functionality and interface
    that is assumed to exist by the modeling strategies that are examined as
    part of this evaluation effort.  Explicitly, HaloSimulator simulates
    (i) the observable, DP data points and (ii) the DP single-pub reach curves
    that will be later used for fitting the multi-pub reach surface in
    PlannerSimulator.
    """

    def __init__(
        self,
        data_set: DataSet,
        params: SystemParameters,
        privacy_tracker: PrivacyTracker,
    ):
        """Halo simulator.

        Args:
          data_set:  The data that will be used as ground truth for this
            halo instance.
          params:  Configuration parameters for this halo instance.
          privacy_tracker: A PrivacyTracker object that will be used to track
            privacy budget consumption for this simulation.
        """
        self._data_set = data_set
        if params.campaign_spend_fractions_generator(1) is not None:
            params = params._replace(
                campaign_spend_fractions=params.campaign_spend_fractions_generator(
                    data_set.publisher_count
                )
            )
        self._params = params
        self._privacy_tracker = privacy_tracker
        self._publishers = []
        campaign_spends = []
        max_spends = []
        for i in range(data_set.publisher_count):
            self._publishers.append(
                Publisher(data_set._data[i], i, params, privacy_tracker)
            )
            campaign_spends.append(self._publishers[i].campaign_spend)
            max_spends.append(self._publishers[i].max_spend)
        self._campaign_spends = tuple(campaign_spends)
        self._max_spends = tuple(max_spends)

    @property
    def publisher_count(self):
        """Returns the number of publishers in this Halo instance."""
        return len(self._publishers)

    @property
    def campaign_spends(self):
        """Returns the vector of campaign spends."""
        return self._campaign_spends

    @property
    def max_spends(self):
        """Returns the vector of per publisher max spends."""
        return self._max_spends

    def true_reach_by_spend(
        self, spends: List[float], max_frequency: int = 1
    ) -> ReachPoint:
        """Returns the true reach obtained for a given spend vector.

        The true reach is the reach that would have actually been obtained
        for a given spend vector.  This is in contrast to the simulated
        reach, which is the noised reach estimated by SMPC.

        Args:
            spends:  The hypothetical amount spend vector, equal in length to
              the number of publishers.  spends[i] is the amount that is
              spent with publisher i.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            A ReachPoint representing the true reach that would have been
            obtained for this spend allocation.
        """
        return self._data_set.reach_by_spend(spends, max_frequency)

    def simulated_reach_by_spend(
        self,
        spends: List[float],
        budget: PrivacyBudget,
        privacy_budget_split: float = 0.5,
        max_frequency: int = 1,
    ) -> ReachPoint:
        """Returns a simulated differentially private reach estimate.

        Args:
            spends:  The hypothetical amount spend vector, equal in length to
              the number of publishers.  spends[i] is the amount that is
              spent with publisher i.
            budget:  The amount of privacy budget that can be consumed while
              satisfying the request.
            privacy_budget_split:  Specifies the proportion of the privacy budget
              that should be allocated to reach estimation.  The remainder is
              allocated to frequency estimation.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            A ReachPoint representing the differentially private estimate of
            the reach that would have been obtained for this spend allocation.
            This estimate is obtained by simulating the construction of
            Liquid Legions sketches, one per publisher, combining them, and
            adding differentially private noise to the result.
        """
        combined_sketch = self._publishers[0].liquid_legions_sketch(spends[0])
        estimator = StandardizedHistogramEstimator(
            max_freq=max_frequency,
            reach_noiser_class=GeometricEstimateNoiser,
            frequency_noiser_class=GeometricEstimateNoiser,
            reach_epsilon=budget.epsilon * privacy_budget_split,
            frequency_epsilon=budget.epsilon * (1 - privacy_budget_split),
            reach_delta=budget.delta * privacy_budget_split,
            frequency_delta=budget.delta * (1 - privacy_budget_split),
            reach_noiser_kwargs={
                "random_state": np.random.RandomState(
                    seed=self._params.generator.integers(low=0, high=1e9)
                )
            },
            frequency_noiser_kwargs={
                "random_state": np.random.RandomState(
                    seed=self._params.generator.integers(low=0, high=1e9)
                )
            },
        )
        for i in range(1, len(spends)):
            sketch = self._publishers[i].liquid_legions_sketch(spends[i])
            combined_sketch = StandardizedHistogramEstimator.merge_two_sketches(
                combined_sketch, sketch
            )
        frequencies = [
            round(x) for x in estimator.estimate_cardinality(combined_sketch)
        ]

        # TODO(jiayu,pasin): Does this look right?
        for noiser_class, epsilon, delta in estimator.output_privacy_parameters():
            self._privacy_tracker.append(
                NoisingEvent(
                    PrivacyBudget(epsilon, delta),
                    DP_NOISE_MECHANISM_DISCRETE_GAUSSIAN,
                    {"privacy_budget_split": privacy_budget_split},
                )
            )

        # convert result to a ReachPoint
        impressions = self._data_set.impressions_by_spend(spends)
        kplus_reaches = ReachPoint.frequencies_to_kplus_reaches(frequencies)
        return ReachPoint(impressions, kplus_reaches, spends)

    def simulated_venn_diagram_reach_by_spend(
        self,
        spends: List[float],
        budget: PrivacyBudget,
        max_frequency: int = 1,
    ) -> List[ReachPoint]:
        """Returns a simulated differentially private Venn diagram reach estimate.

        For each subset of publishers, computes a differentially private
        reach and frequency estimate for those users who are reached by
        all and only the publishers in that subset.

        Args:
            spends:  The hypothetical spend vector, equal in length to
              the number of publishers.  spends[i] is the amount that is
              spent with publisher i. Note that publishers with 0 spends will
              not be included in the Venn diagram reach.
            budget:  The amount of privacy budget that can be consumed while
              satisfying the request.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            A list of ReachPoint. Each reach point represents the mapping from
            the spends of a subset of publishers to the differentially private
            estimate of the number of people reached in this subset.
        """
        raise NotImplementedError()

    def _form_venn_diagram_regions(
        self, spends: List[float], max_frequency: int = 1
    ) -> Tuple[Dict[int, Set], Dict[int, List]]:
        """Form Venn diagram regions that contain k+ reaches

        For each subset of publishers, computes k+ reaches for those users
        who are reached by the publishers in that subset.

        Args:
            spends:  The hypothetical spend vector, equal in length to
              the number of publishers.  spends[i] is the amount that is
              spent with publisher i. Note that publishers with 0 spends will
              not be included in the Venn diagram reach.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            pub_set_by_region:  Contains the mapping of a binary representation
              to a set of publisher ids that are in the corresponding region.
            regions:  Contains k+ frequencies indexed by their binary
              representation.
        """
        # Get user counts by spend for each active publisher
        user_counts_by_pub_id = {}
        for pub_id, spend in enumerate(spends):
            if not spend:
                continue
            user_counts_by_pub_id[pub_id] = self._publishers[
                pub_id
            ]._publisher_data.user_counts_by_spend(spend)

        if len(user_counts_by_pub_id) > MAX_ACTIVE_PUBLISHERS:
            raise ValueError(
                f"There are {len(user_counts_by_pub_id)} publishers"
                f"for Venn diagram algorithm. The maximum limit is {MAX_ACTIVE_PUBLISHERS}"
            )

        # Locate user's region and sum the impressions. Also, map the region
        # representation to publisher ids.
        UserInfo = collections.namedtuple(
            typename="Userinfo",
            field_names="located_region impressions",
            defaults=[0, 0],
        )

        users_info = collections.defaultdict(UserInfo)
        pub_set_by_region = collections.defaultdict(set)

        for pub_id, user_counts in user_counts_by_pub_id.items():
            for user_id, impressions in user_counts.items():
                lr = users_info[user_id].located_region
                new_lr = lr | (1 << pub_id)

                if not new_lr in pub_set_by_region:
                    pub_set_by_region[new_lr] = copy.deepcopy(pub_set_by_region[lr])
                    pub_set_by_region[new_lr].add(pub_id)

                users_info[user_id] = UserInfo(
                    new_lr, users_info[user_id].impressions + impressions
                )

        # Remove the null region
        pub_set_by_region.pop(0)

        # Fill in the occupied regions of the Venn diagram with capped user counts.
        region_user_counts = collections.defaultdict(dict)
        for user_id, info in users_info.items():
            lr = info.located_region
            region_user_counts[lr][user_id] = (
                max_frequency if info.impressions > max_frequency else info.impressions
            )

        # Compute k+ reaches in each region except the first one
        regions = dict()
        for lr, user_counts in region_user_counts.items():
            counts = list(user_counts.values())
            regions[lr] = list(
                np.bincount(counts, minlength=max_frequency + 1)[1:]  # ignore 0 count
            )

        return dict(pub_set_by_region), regions

    def simulated_reach_curve(
        self, publisher_index: int, budget: PrivacyBudget
    ) -> ReachCurve:
        """Returns a simulated differentially private reach curve model.

        Args:
          publisher_index: The index of the publisher for which the reach
            curve is to be returned.
          budget:  The amount of privacy budget that can be consumed while
            satisfying the request.
        """
        raise NotImplementedError()

    def simulated_vector_of_counts(
        self, publisher_index: int, budget: PrivacyBudget
    ) -> VectorOfCounts:
        """Returns a simulated differentially private VectorOfCounts for a campaign.

        Args:
          publisher_index: The index of the publisher for which the reach
            curve is to be returned.
          budget:  The amount of privacy budget that can be consumed while
            satisfying the request.
        """
        raise NotImplementedError()

    @property
    def privacy_tracker(self) -> PrivacyTracker:
        return self._privacy_tracker
