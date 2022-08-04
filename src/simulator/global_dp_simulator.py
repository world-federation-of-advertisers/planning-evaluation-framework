# Copyright 2022 The Private Cardinality Estimation Framework Authors
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
"""Simulates a hypothetical Halo system that uses local-DP LiquidLegions.

The actual Halo system was simulated in halo_simulator.py.  There, publishers share
raw sketches with MPC, and MPC outputs the reach and frequency estimates with global
DP noises.  Running MPC multiple times, we obtain the the reach and frequency of
different subsets of publishers, and can build a planning model from them.

A planning model can be used for estimating the reach and frequency of unobserved
subset, and for other purposes such as ShareShift.  Now, suppose that we are only
interested in the reach of subsets, then the local-DP approach is an alternative
to the planning model.  As implemented in this module, in a local DP approach,
each publisher locally noises their LiquidLegions by flipping each register.
After flipping, the sketches are already differentially private (DP), and they can 
be published.  The reach of any subset of publishers can be estimated by these
public sketches.

By implementing the local DP approach here, we want to show by evaluations that
our planning model estimates subset reach more accurately than the local-DP
approach.  And of course, the planning model is applicable to a wider range of
use cases.
"""

from itertools import combinations
from typing import List, Set, Union
import numpy as np
from absl import logging


from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
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
from wfa_cardinality_estimation_evaluation_framework.estimators.same_key_aggregator import (
    StandardizedHistogramEstimator,
)
from wfa_cardinality_estimation_evaluation_framework.estimators.estimator_noisers import (
    GeometricEstimateNoiser,
)


class GlobalDpSimulator:
    """Simulator for the Local DP reach estimation system.

    Each publisher locally noises their LiquidLegions by flipping each register.
    After flipping, the sketches are differentially private (DP), and they can
    be published.  The reach of any subset of publishers can be estimated by these
    public sketches.
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
        self.computed = False

    def obtain_global_dp_estimates(self, budget: PrivacyBudget):
        """Obtain Gloabl DP estimates given a privacy budget."""
        if self.computed:
            return
        # The following is not a real privacy budget tracker.
        self._privacy_tracker.append(
            NoisingEvent(
                budget=PrivacyBudget(budget.epsilon, budget.delta),
                mechanism=DP_NOISE_MECHANISM_DISCRETE_LAPLACE,
                params={},
            )
        )
        sketches = [
            pub.liquid_legions_sketch(spend)
            for pub, spend in zip(self._publishers, self._campaign_spends)
        ]
        p = self._data_set.publisher_count
        # In the following, use the EDP * user DP definition.
        per_query_reach_epsilon = 0.0072 * p / 2 ** (p - 1)
        if p <= 3:
            per_query_freq_epsilon = 0.2015 * p / 2 ** (p - 1)
        else:
            per_query_freq_epsilon = 0.01  # Set an arbitrary tiny number

        all_subset_sketches = {}
        for i in range(p):
            all_subset_sketches[(i,)] = sketches[i]
        for r in range(2, p + 1):
            for comb in combinations(range(p), r):
                all_subset_sketches[
                    comb
                ] = StandardizedHistogramEstimator.merge_two_sketches(
                    all_subset_sketches[comb[:-1]], all_subset_sketches[comb[-1:]]
                )

        self.all_subset_reaches = {}
        for key in all_subset_sketches:
            self.all_subset_reaches[key] = StandardizedHistogramEstimator(
                max_freq=1,
                reach_noiser_class=GeometricEstimateNoiser,
                frequency_noiser_class=GeometricEstimateNoiser,
                reach_epsilon=per_query_reach_epsilon,
                frequency_epsilon=per_query_freq_epsilon,
                reach_delta=0,
                frequency_delta=0,
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
            ).estimate_cardinality(all_subset_sketches[key])
            self.all_subset_reaches[key] = [
                round(x) for x in self.all_subset_reaches[key]
            ]
        self.computed = True

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

    def find_subset(self, spends: List[float]) -> Union[Set, None]:
        """Find if a spend vector corresponds to a subset.

        Args:
            spends:  The hypothetical amount spend vector, equal in length to
                the number of publishers.  spends[i] is the amount that is
                spent with publisher i.

        Returns:
            A given spends corresponds to a subset if for each 1 <= i <= p,
            spends[i] = 0 or self._campaign_spends[0].  If spends does
            correspond to a subset, returns the subset.  Otherwise, returns
            None.
        """
        subset = set()
        for i in range(len(spends)):
            if spends[i] == 0:
                continue
            if (
                spends[i] >= (1 - 1e-3) * self._campaign_spends[i]
                and spends[i] <= (1 + 1e-3) * self._campaign_spends[i]
            ):
                subset.add(i)
                continue
            return None
        return subset

    def simulated_reach_by_spend(
        self,
        spends: List[float],
        max_frequency: int = 1,
    ) -> ReachPoint:
        """Returns a simulated differentially private reach estimate.

        Args:
            spends:  The hypothetical amount spend vector, equal in length to
                the number of publishers.  spends[i] is the amount that is
                spent with publisher i.
            budget:  The amount of privacy budget that can be consumed while
                satisfying the request.
            max_frequency:  The maximum frequency in the output ReachPoint. This
                is purely for the interoperability with modeling_strategy and
                experimental_trial.  So far, local DP LiquidLegions can only be
                used to estimated 1+  reach.  So, if the given max_frequency > 1,
                we will set the k+ reach to be 0 for all k > 1 in the output
                ReachPoint.

        Returns:
            A ReachPoint representing the differentially private estimate of
            the reach that would have been obtained for this spend allocation.

            To obtain this, we compare the given `spends` with
            `self._campaign_spends`.  If `spends` align with
            `self._campaign_spends` on all non-zero elements, then it corresponds
            to a subset of publishers in the campaign, and we output the subset
            reach estimated by the local DP sketches.  Otherwise, output reach
            = np.nan.
        """
        if not self.computed:
            raise ValueError("Estimate subset reaches first.")
        subset = self.find_subset(spends)
        if subset is None:
            kplus_reaches = [np.nan] * max_frequency
        else:
            kplus_reaches = self.all_subset_reaches[tuple(subset)]
            kplus_reaches = np.minimum.accumulate(np.maximum(kplus_reaches, 0))

        return ReachPoint(
            impressions=self._data_set.impressions_by_spend(spends),
            kplus_reaches=kplus_reaches,
            spends=spends,
            universe_size=self._data_set.universe_size,
        )
