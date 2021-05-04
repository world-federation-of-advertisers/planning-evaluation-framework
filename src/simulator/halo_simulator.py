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

from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    DP_NOISE_MECHANISM_GAUSSIAN,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import NoisingEvent
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyBudget
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyTracker
from wfa_planning_evaluation_framework.simulator.publisher import Publisher
from wfa_planning_evaluation_framework.simulator.simulation_parameters import (
    SimulationParameters,
)


class HaloSimulator:
    """Simulator for the Halo System.

    A Halo Simulator simulates the Halo System from the Measurement Coordinator
    downwards.  In particular, it simulates the SMPC and the behavior of the
    publishers.  The Halo Simulator represents the functionality and interface
    that is assumed to exist by the modeling strategies that are examined as
    part of this evaluation effort.
    """

    def __init__(
        self,
        data_set: DataSet,
        params: SimulationParameters,
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
        for i in range(data_set.publisher_count):
            self._publishers.append(
                Publisher(data_set._data[i], i, params, privacy_tracker)
            )

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
        epsilon_split: float = 0.5,
        max_frequency: int = 1,
    ) -> ReachPoint:
        """Returns a simulated differentially private reach estimate.

        Args:
            spends:  The hypothetical amount spend vector, equal in length to
              the number of publishers.  spends[i] is the amount that is
              spent with publisher i.
            budget:  The amount of privacy budget that can be consumed while
              satisfying the request.
            epsilon_split:  Specifies the proportion of the privacy budget that
              should be allocated to reach estimation.  The remainder is allocated
              to frequency estimation.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            A ReachPoint representing the differentially private estiamte of
            the reach that would have been obtained for this spend allocation.
            This estimate is obtained by simulating the construction of
            Liquid Legions sketches, one per publisher, combining them, and
            adding differentially private noise to the result.
        """
        combined_sketch = self._publishers[0].liquid_legions_sketch(spends[0])
        estimator = StandardizedHistogramEstimator(
            max_freq=max_frequency, epsilon=budget.epsilon, epsilon_split=epsilon_split
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
        self._privacy_tracker.append(
            NoisingEvent(
                budget,
                DP_NOISE_MECHANISM_GAUSSIAN,
                {"epsilon": budget.epsilon, "epsilon_split": epsilon_split},
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
    ) -> List[Tuple[Set, ReachPoint]]:
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
            A list of pairs (S, R), where S specifies a subset of publishers
            and R is a ReachPoint representing the differentially private
            estimate of the number of people reached in this subset.
            The set S is given as a subset of the integers 0..p-1, where p
            is the number of publishers.
        """
        raise NotImplementedError()

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
