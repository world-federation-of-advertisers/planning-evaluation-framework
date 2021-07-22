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
from itertools import chain, combinations
import numpy as np
from scipy.special import expi

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

    def _cardinality_estimate_variance(self, n: int) -> float:
        """Variance of cardinality estimate by unnoised LiquidLegions.

        Args:
          n:  true cardinality.

        Returns:
          var(n-hat) where n-hat is the cardinality estimate from a
          LiquidLegions which contains n distinct items and has the system
          parameters (decay rate and sketch size) in this halo.
        """
        m = self._params.liquid_legions.sketch_size
        a = self._params.liquid_legions.decay_rate
        c = a * n / m / (1 - np.exp(-a))
        variance = (
            expi(-c) - expi(-c * np.exp(-a)) - expi(-2 * c) + expi(-2 * c * np.exp(-a))
        )
        variance *= a * n ** 2 / m / (np.exp(-c) - np.exp(-c * np.exp(-a))) ** 2
        variance -= n
        return variance

    def _num_active_registers(
        self, n: int, random_generator: np.random.Generator = np.random.default_rng()
    ) -> int:
        """Simulate an observation of the number of active registers."""
        m = self._params.liquid_legions.sketch_size
        a = self._params.liquid_legions.decay_rate
        register_probs = np.exp(-a * np.arange(m) / m)
        register_probs /= sum(register_probs)
        return sum(random_generator.multinomial(n, register_probs) == 1)

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
              spent with publisher i.
            budget:  The amount of privacy budget that can be consumed while
              satisfying the request.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            A list of ReachPoint. Each reach point represents the mapping from
            the spends of a subset of publishers to the differentially private
            estimate of the number of people reached in this subset.
        """
        raise NotImplementedError()

    def _sample_venn_diagram(
        self,
        primitive_regions: Dict[int, List],
        sample_size: int,
        random_generator: np.random.Generator = np.random.default_rng(),
    ) -> Dict[int, List]:
        """Return primitive regions with sampled reaches.

        Args:
            primitive_regions:  Contains k+ reaches in the regions. The k+
              reaches for a given region is given as a list r[] where r[k] is
              the number of people who were reached AT LEAST k+1 times.
            sample_size:  The total number of sampled reach from the primitive
              regions.
            random_generator:  An instance of numpy.random.Generator that is
              used for generating samples from a multivariate hypergeometric
              distribution.
        Returns:
            A dictionary in which each key are the binary representations of
              each primitive region of the Venn diagram, and each value is a
              list of the reach in the corresponding region.
        """

        region_repr_and_reach_pairs = [
            (region_repr, kplus_reaches[0])
            for region_repr, kplus_reaches in primitive_regions.items()
        ]
        region_repr_seq, reach_population = list(zip(*region_repr_and_reach_pairs))

        sampled_reach = random_generator.multivariate_hypergeometric(
            reach_population, sample_size
        )

        return {
            region_repr: [r] for region_repr, r in zip(region_repr_seq, sampled_reach)
        }

    def _generate_reach_points_from_venn_diagram(
        self, spends: List[float], primitive_regions: Dict[int, List]
    ) -> List[ReachPoint]:
        """Return the reach points of the powerset of active publishers.

        For each subset of active publishers, compute reach and frequency
        estimate for those users who are reached by at least one of the
        publishers in the subset.

        Args:
            spends:  The hypothetical spend vector, equal in length to
              the number of publishers.  spends[i] is the amount that is
              spent with publisher i.
            primitive_regions:  Contains k+ reaches in the regions. The k+
              reaches for a given region is given as a list r[] where r[k] is
              the number of people who were reached AT LEAST k+1 times.
        Returns:
            A list of ReachPoint. Each reach point represents the mapping from
            the spends of a subset of publishers to the number of people reached
            in this subset.
        """
        active_pub_set = [i for i in range(len(spends)) if spends[i]]
        active_pub_powerset = chain.from_iterable(
            combinations(active_pub_set, r) for r in range(1, len(active_pub_set) + 1)
        )
        impressions = self._data_set.impressions_by_spend(spends)

        reach_points = []
        for sub_pub_ids in active_pub_powerset:
            sub_reach = self._aggregate_reach_in_primitive_venn_diagram_regions(
                sub_pub_ids, primitive_regions
            )
            pub_subset = set(sub_pub_ids)
            sub_imps = [
                imps if pub_id in pub_subset else 0
                for pub_id, imps in enumerate(impressions)
            ]
            sub_spends = [
                s if pub_id in pub_subset else 0 for pub_id, s in enumerate(spends)
            ]
            reach_points.append(ReachPoint(sub_imps, [sub_reach], sub_spends))

        return reach_points

    def _aggregate_reach_in_primitive_venn_diagram_regions(
        self, pub_ids: List[int], primitive_regions: Dict[int, List]
    ) -> int:
        """Returns aggregated reach from Venn diagram primitive regions.

        To obtain the union reach of the given subset of publishers, we sum up
        the reaches from the primitive regions which belong to at least one of
        the given publisher. Note that the binary representation of the index
        of a primitive region represents the formation of publisher IDs in that
        primitive region.

        For example, given a subset of publisher ids, {0}, out of the whole set
        {0, 1, 2}, the reaches in the following primitive regions will be summed
        up:

            region with index = 1 = bin('001'): belongs to pub_id-0
            region with index = 3 = bin('011'): belongs to pub_id-0 and 1
            region with index = 5 = bin('101'): belongs to pub_id-0 and 2
            region with index = 7 = bin('111'): belongs to pub_id-0, 1, and 2

        Args:
            pub_ids:  The list of target publisher IDs for computing aggregated
              reach.
            primitive_regions:  Contains k+ reaches in the regions. The k+
              reaches for a given region is given as a list r[] where r[k] is
              the number of people who were reached AT LEAST k+1 times.
        Returns:
            aggregated_reach:  The total reach from the given publishers.
        """
        targeted_pub_repr = sum(1 << pub_id for pub_id in pub_ids)
        aggregated_reach = sum(
            primitive_regions[r][0]
            for r in primitive_regions.keys()
            if r & targeted_pub_repr
        )

        return aggregated_reach

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
