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
from collections import defaultdict
from itertools import chain, combinations
import numpy as np
from scipy.special import expi

from wfa_cardinality_estimation_evaluation_framework.estimators.base import (
    EstimateNoiserBase,
)
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
    DP_NOISE_MECHANISM_DISCRETE_LAPLACE,
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
        kplus_reaches = [
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
        return ReachPoint(
            impressions=impressions, kplus_reaches=kplus_reaches, spends=spends
        )

    def _liquid_legions_cardinality_estimate_variance(self, n: int) -> float:
        """Variance of cardinality estimate by unnoised LiquidLegions.

        TODO(jiayu): add a link of the paper to explain the variance formula,
        once the paper is published.

        Args:
          n:  Carinality of the items contained in a LiquidLegions.

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

    def _liquid_legions_num_active_regions(self, n: int) -> int:
        """Simulate an observation of the number of active registers.

        Args:
          n:  Carinality of the items contained in a LiquidLegions.
          random_generator:  An instance of numpy.random.Generator that is
            used for assigning different items into different registers.

        Returns:
          An observation of the number of active registers in the LiquidLegions.
        """
        m = self._params.liquid_legions.sketch_size
        a = self._params.liquid_legions.decay_rate
        register_probs = np.exp(-a * np.arange(m) / m)
        register_probs /= sum(register_probs)
        num_active_regs = sum(
            self._params.generator.multinomial(n, register_probs) == 1
        )
        if num_active_regs == 0:
            raise RuntimeError(
                "Zero active registers. Please change the LiquidLegions configuration."
            )
        return num_active_regs

    def simulated_venn_diagram_reach_by_spend(
        self,
        spends: List[float],
        budget: PrivacyBudget,
        privacy_budget_split: float = 0.5,
        max_frequency: int = 1,
    ) -> List[ReachPoint]:
        """Returns a simulated differentially private Venn diagram reach estimate.

        For each subset of publishers, computes a differentially private reach
        and frequency estimate for those users who are reached by all and only
        only the publishers in that subset. If user X is included, then X is
        reached by at least one of the publishers in the set.  And, every user
        that is reached by at least one of the publishers is included in the
        count.

        Args:
            spends:  The hypothetical spend vector, equal in length to
              the number of publishers.  spends[i] is the amount that is
              spent with publisher i.
            budget:  The amount of privacy budget that can be consumed while
              satisfying the request.
            privacy_budget_split:  Specifies the proportion of the privacy budget
              that should be allocated to reach estimation.  The remainder is
              allocated to cardinality estimation.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            A list of ReachPoint. Each reach point represents the mapping from
            the spends of a subset of publishers to the differentially private
            estimate of the number of people reached in this subset.
        """
        if max_frequency != 1:
            raise ValueError("Max frequency has to be 1.")

        venn_diagram_regions = self._form_venn_diagram_regions(spends, max_frequency)

        if not venn_diagram_regions:
            return []

        true_cardinality = self.true_reach_by_spend(spends).reach()
        sample_size = self._liquid_legions_num_active_regions(true_cardinality)

        sampled_venn_diagram_regions = self._sample_venn_diagram(
            venn_diagram_regions, sample_size
        )

        noised_sampled_venn_diagram_regions = self._add_dp_noise_to_primitive_regions(
            sampled_venn_diagram_regions,
            budget,
            privacy_budget_split,
        )

        scaled_venn_diagram_regions = self._scale_up_venn_diagram_regions(
            noised_sampled_venn_diagram_regions,
            true_cardinality,
            np.sqrt(
                self._liquid_legions_cardinality_estimate_variance(true_cardinality)
            ),
            budget,
            1 - privacy_budget_split,
        )

        reach_points = self._generate_reach_points_from_venn_diagram(
            spends, scaled_venn_diagram_regions
        )

        return reach_points

    def _form_venn_diagram_regions(
        self, spends: List[float], max_frequency: int = 1
    ) -> Dict[int, List]:
        """Form primitive Venn diagram regions that contain k+ reaches

        For each subset in the powerset of publishers with nonzero spend,
        computes k+ reaches for those users who are reached by the publishers
        in that subset.

        Args:
            spends:  The hypothetical spend vector, equal in length to the
              number of publishers.  spends[i] is the amount that is spent with
              publisher i. Note that the publishers with 0 spends, i.e. inactive
              publishers, will not be included in the Venn diagram regions.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            regions:  A dictionary in which each key is the binary
              representation of each primitive region of the Venn diagram, and
              each value is a list of the k+ reaches in the corresponding
              region.
              Note that the binary representation of a key represents the
              formation of publisher IDs in that primitive region. For example,
              primitive_regions[key] with key = 5 = bin('101') is the region
              which belongs to pub_id-0 and id-2.
              The k+ reaches for a given region is given as a list r[] where
              r[k] is the number of people who were reached AT LEAST k+1 times.
        """
        # Get user counts by spend for each active publisher
        user_counts_by_pub_id = {
            pub_id: self._publishers[pub_id]._publisher_data.user_counts_by_spend(spend)
            for pub_id, spend in enumerate(spends)
            if spend
        }

        if len(user_counts_by_pub_id) > MAX_ACTIVE_PUBLISHERS:
            raise ValueError(
                f"There are {len(user_counts_by_pub_id)} publishers for the Venn "
                f"diagram algorithm. The maximum limit is {MAX_ACTIVE_PUBLISHERS}."
            )

        # Generate the representations of all primitive regions from the
        # powerset of the active publishers, excluding the empty set.
        # Ex: if active_pubs = [0, 2] among all publishers [0, 1, 2], then
        # active_pub_powerset is [[0], [2], [0, 2]]. For the regions from the
        # the powerset of the active publishers, they are: [2^0, 2^2, 2^0 + 2^2]
        active_pubs = list(user_counts_by_pub_id.keys())
        active_pub_powerset = chain.from_iterable(
            combinations(active_pubs, r) for r in range(1, len(active_pubs) + 1)
        )
        regions_with_active_pubs = [
            sum(1 << pub_id for pub_id in pub_ids) for pub_ids in active_pub_powerset
        ]

        # Locate user's region which is represented by the binary representation
        # of a number, and sum the user's impressions.
        user_region = defaultdict(int)
        user_impressions = defaultdict(int)

        for pub_id, user_counts in user_counts_by_pub_id.items():
            for user_id, impressions in user_counts.items():
                # To update the user's located region, we use bit operation here.
                # Ex: For a user reached by publisher id-0 and id-2, it's located
                # at the region with the binary representation = bin('101') = 5.
                # If the user is also reached by publisher id-1, then the updated
                # representation will be bin('111') = 7.
                user_region[user_id] |= 1 << pub_id
                user_impressions[user_id] += impressions

        # Compute the frequencies in the occupied primitive Venn diagram regions
        # with the user counts capped by max_frequency.
        frequencies_by_region = {
            r: [0] * (max_frequency + 1) for r in regions_with_active_pubs
        }
        for user_id, region in user_region.items():
            impressions = min(max_frequency, user_impressions[user_id])
            frequencies_by_region[region][impressions] += 1

        # Compute k+ reaches in the primitive regions. Ignore 0-frequency.
        occupied_regions = set(user_region.values())
        regions = {
            r: ReachPoint.frequencies_to_kplus_reaches(freq[1:])
            if r in occupied_regions
            else freq[1:]  # i.e. zero vector
            for r, freq in frequencies_by_region.items()
        }

        return regions

    def _sample_venn_diagram(
        self,
        primitive_regions: Dict[int, List],
        sample_size: int,
    ) -> Dict[int, int]:
        """Return primitive regions with sampled reaches.

        Args:
            primitive_regions:  A dictionary in which each key is the binary
              representation of a primitive region of the Venn diagram, and
              each value is a list of the k+ reaches in the corresponding
              region.
              Note that the binary representation of a key represents the
              formation of publisher IDs in that primitive region. For example,
              primitive_regions[key] with key = 5 = bin('101') is the region
              which belongs to pub_id-0 and id-2.
              The k+ reaches for a given region is given as a list r[] where
              r[k] is the number of people who were reached AT LEAST k+1 times.
            sample_size:  The total number of sampled reach from the primitive
              regions.
        Returns:
            A dictionary in which each key is the binary representation of a
              primitive region of the Venn diagram, and each value is the
              sampled reach in the corresponding gregion.
              Note that the binary representation of the key represents the
              formation of publisher IDs in that primitive region. For example,
              primitive_regions[key] with key = 5 = bin('101') is the region
              which belongs to pub_id-0 and id-2.
        """

        region_repr_and_reach_pairs = [
            (region_repr, kplus_reaches[0])
            for region_repr, kplus_reaches in primitive_regions.items()
        ]
        region_repr_seq, reach_population = list(zip(*region_repr_and_reach_pairs))

        if sample_size > sum(reach_population):
            raise ValueError(
                f"The given sample size is {sample_size} which is"
                f" larger than the total number of reach = {sum(reach_population)}"
            )

        sampled_reach = self._params.generator.multivariate_hypergeometric(
            reach_population, sample_size
        )

        return {
            region_repr: r for region_repr, r in zip(region_repr_seq, sampled_reach)
        }

    def _add_dp_noise_to_primitive_regions(
        self,
        primitive_regions: Dict[int, int],
        budget: PrivacyBudget,
        privacy_budget_split: float,
    ) -> Dict[int, int]:
        """Add differential privacy noise to every primitive region

        Args:
            primitive_regions:  A dictionary in which each key is the binary
              representation of a primitive region of the Venn diagram, and
              each value is the reach in the corresponding region.
              Note that the binary representation of the key represents the
              formation of publisher IDs in that primitive region. For example,
              primitive_regions[key] with key = 5 = bin('101') is the region
              which belongs to pub_id-0 and id-2.
            budget:  The amount of privacy budget that can be consumed while
              satisfying the request.
            privacy_budget_split:  Specifies the proportion of the privacy
              budget that should be allocated to the operation in this function.
        Returns:
            A dictionary in which each key is the binary representation of a
              primitive region of the Venn diagram, and each value is the
              noised reach in the corresponding region.
              Note that the binary representation of the key represents the
              formation of publisher IDs in that primitive region. For example,
              primitive_regions[key] with key = 5 = bin('101') is the region
              which belongs to pub_id-0 and id-2.
        """
        noiser = GeometricEstimateNoiser(
            budget.epsilon * privacy_budget_split,
            np.random.RandomState(
                seed=self._params.generator.integers(low=0, high=1e9)
            ),
        )

        noise_event = NoisingEvent(
            PrivacyBudget(
                budget.epsilon * privacy_budget_split,
                budget.delta * privacy_budget_split,
            ),
            DP_NOISE_MECHANISM_DISCRETE_LAPLACE,
            {"privacy_budget_split": privacy_budget_split},
        )

        noised_primitive_regions = {
            region_repr: max(0, noiser(reach))
            for region_repr, reach in primitive_regions.items()
        }

        self._privacy_tracker.append(noise_event)

        return noised_primitive_regions

    def _scale_up_reach_in_primitive_regions(
        self,
        primitive_regions: Dict[int, int],
        true_cardinality: int,
        std_cardinality_estimate: float,
        budget: PrivacyBudget,
        privacy_budget_split: float,
    ) -> Dict[int, int]:
        """Scale up the reaches in the primitive regions

        Given the primitive Venn diagram regions, we scale up all reaches in
        the regions by a scaling factor to match the cardinality estimate
        in the Halo system.

        Args:
            primitive_regions:  A dictionary in which each key is the binary
              representation of a primitive region of the Venn diagram, and
              each value is the non-negative reach in the corresponding region.
              Note that the binary representation of the key represents the
              formation of publisher IDs in that primitive region. For example,
              primitive_regions[key] with key = 5 = bin('101') is the region
              which belongs to pub_id-0 and id-2.
            true_cardinality:  The true value of the reach given the dataset.
            std_cardinality_estimate:  The standard deviation of the
              cardinality estimate in the Halo system.
            budget:  The amount of privacy budget that can be consumed while
              satisfying the request.
            privacy_budget_split:  Specifies the proportion of the privacy
              budget that should be allocated to the operation in this function.
        Returns:
            A dictionary in which each key is the binary representation of a
              primitive region of the Venn diagram, and each value is the
              scaled reach in the corresponding region.
              Note that the binary representation of the key represents the
              formation of publisher IDs in that primitive region. For example,
              primitive_regions[key] with key = 5 = bin('101') is the region
              which belongs to pub_id-0 and id-2.
        """
        noiser = GeometricEstimateNoiser(
            budget.epsilon * privacy_budget_split,
            np.random.RandomState(
                seed=self._params.generator.integers(low=0, high=1e9)
            ),
        )
        cardinality_estimate = noiser(
            self._params.generator.normal(true_cardinality, std_cardinality_estimate)
        )
        noise_event = NoisingEvent(
            PrivacyBudget(
                budget.epsilon * privacy_budget_split,
                budget.delta * privacy_budget_split,
            ),
            DP_NOISE_MECHANISM_DISCRETE_LAPLACE,
            {"privacy_budget_split": privacy_budget_split},
        )
        self._privacy_tracker.append(noise_event)

        sum_sampled_reach = sum(primitive_regions.values())

        # scaling factor = cardinality estimate / sum of reaches in the primitive regions
        scaling_factor = (
            cardinality_estimate / sum_sampled_reach if sum_sampled_reach else 0
        )
        scaled_primitive_regions = {
            region: reach * scaling_factor
            for region, reach in primitive_regions.items()
        }

        return scaled_primitive_regions

    def _aggregate_reach_in_primitive_venn_diagram_regions(
        self, pub_ids: List[int], primitive_regions: Dict[int, int]
    ) -> int:
        """Returns the aggregated reach from the primitive Venn diagram regions.

        To obtain the union reach of the given subset of publishers, we sum up
        the reaches from the primitive regions which belong to at least one of
        the given publisher. Note that the binary representation of the key of
        a primitive region represents the formation of publisher IDs in that
        primitive region.

        For example, given a subset of publisher ids, {0}, out of the whole set
        {0, 1, 2}, the reaches in the following primitive regions will be summed
        up:

            region with key = 1 = bin('001'): belongs to pub_id-0
            region with key = 3 = bin('011'): belongs to pub_id-0 and 1
            region with key = 5 = bin('101'): belongs to pub_id-0 and 2
            region with key = 7 = bin('111'): belongs to pub_id-0, 1, and 2

        Args:
            pub_ids:  The list of target publisher IDs for computing aggregated
              reach.
            primitive_regions:  A dictionary in which each key is the binary
              representation of a primitive region of the Venn diagram, and
              each value is the reach in the corresponding region.
              Note that the binary representation of the key represents the
              formation of publisher IDs in that primitive region. For example,
              primitive_regions[key] with key = 5 = bin('101') is the region
              which belongs to pub_id-0 and id-2.
        Returns:
            The sum of reaches from the given publishers.
        """
        targeted_pub_repr = sum(1 << pub_id for pub_id in pub_ids)
        aggregated_reach = sum(
            reach for r, reach in primitive_regions.items() if r & targeted_pub_repr
        )

        return aggregated_reach

    def _generate_reach_points_from_venn_diagram(
        self, spends: List[float], primitive_regions: Dict[int, int]
    ) -> List[ReachPoint]:
        """Return the reach points of the powerset of active publishers.

        For each subset of active publishers, compute reach estimate for those
        users who are reached by at least one of the active publishers in the
        subset. Note that the reach points generated by the implementation
        contain 1+ reaches.

        Args:
            spends:  The hypothetical spend vector, equal in length to the
              number of publishers.  spends[i] is the amount that is spent with
              publisher i.
            primitive_regions:  A dictionary in which each key is the binary
              representation of a primitive region of the Venn diagram, and
              each value is the reach in the corresponding region.
              Note that the binary representation of the key represents the
              formation of publisher IDs in that primitive region. For example,
              primitive_regions[key] with key = 5 = bin('101') is the region
              which belongs to pub_id-0 and id-2.
        Returns:
            A list of ReachPoint. Each reach point represents the mapping from
            the spends of a subset of active publishers to the number of people
            reached in this subset.
        """
        active_pub_set = [i for i in range(len(spends)) if spends[i]]
        active_pub_powerset = chain.from_iterable(
            combinations(active_pub_set, set_size)
            for set_size in range(1, len(active_pub_set) + 1)
        )
        impressions = self._data_set.impressions_by_spend(spends)

        reach_points = []
        for sub_pub_ids in active_pub_powerset:
            sub_reach = self._aggregate_reach_in_primitive_venn_diagram_regions(
                sub_pub_ids, primitive_regions
            )

            pub_subset = set(sub_pub_ids)
            pub_vector = np.array([int(i in pub_subset) for i in range(len(spends))])
            sub_imps = np.array(impressions) * pub_vector
            sub_spends = np.array(spends) * pub_vector

            reach_points.append(
                ReachPoint(sub_imps.tolist(), [sub_reach], sub_spends.tolist())
            )

        return reach_points

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
