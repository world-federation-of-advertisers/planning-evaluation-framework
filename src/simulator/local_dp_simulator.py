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

from typing import List, Set, Union
import numpy as np
from absl import logging

from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import (
    BlipNoiser,
    SurrealDenoiser,
    FirstMomentEstimator,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    DP_NOISE_MECHANISM_BLIP,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import NoisingEvent
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyBudget
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyTracker
from wfa_planning_evaluation_framework.simulator.publisher import Publisher
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    SystemParameters,
)


class LocalDpSimulator:
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
        self.local_dp_sketches_obtained = False

    def obtain_local_dp_sketches(self, budget: PrivacyBudget):
        """Obtain Local DP sketches given a privacy budget."""
        if self.local_dp_sketches_obtained:
            logging.vlog(
                2,
                "Local DP sketches had been created. The new sketching request is ignored.",
            )
            return
        # Under the vid * EDP prviacy definition, in a local DP approach we
        # simply charge the given budget on each EDP.
        self._privacy_tracker.append(
            NoisingEvent(
                budget=PrivacyBudget(budget.epsilon, budget.delta),
                mechanism=DP_NOISE_MECHANISM_BLIP,
                params={},
            )
        )
        raw_sketches = [
            pub.liquid_legions_sketch(spend).exponential_bloom_filter
            for pub, spend in zip(self._publishers, self._campaign_spends)
        ]
        total_epsilon = self._data_set.publisher_count * (0.0072 + 0.2015)
        noiser = BlipNoiser(
            epsilon=total_epsilon,
            random_state=np.random.RandomState(
                seed=self._params.generator.integers(1e9)
            ),
        )
        self.local_dp_sketches = [noiser(sketch) for sketch in raw_sketches]
        self.denoiser = SurrealDenoiser(epsilon=total_epsilon)
        self.local_dp_sketches_obtained = True

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
        if not self.local_dp_sketches_obtained:
            raise RuntimeError("Create local DP sketches first.")
        subset = self.find_subset(spends)
        if subset is None:
            reach = np.nan
        else:
            estimator = FirstMomentEstimator(method="exp", denoiser=self.denoiser)
            reach = estimator([self.local_dp_sketches[i] for i in subset])[0]
        return ReachPoint(
            impressions=self._data_set.impressions_by_spend(spends),
            kplus_reaches=[reach] + [np.nan] * (max_frequency - 1),
            spends=spends,
            universe_size=self._data_set.universe_size,
        )
