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
"""The Planner Simulator

The purpose of the Simulator is to simulate the execution
of a specific modeling strategy with respect a specific data set.
"""

from typing import List

from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.simulator.halo_simulator import HaloSimulator
from wfa_planning_evaluation_framework.simulator.modeling_strategy import (
    ModelingStrategy,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyBudget
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyTracker
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    SystemParameters,
)


class PlannerSimulator:
    def __init__(
        self,
        halo: HaloSimulator,
        modeling_strategy: ModelingStrategy,
        params: SystemParameters,
        privacy_tracker: PrivacyTracker,
    ):
        """Simulator for the planner.

        Args:
          data_set:  The data that will be used as ground truth for this
            halo instance.
          params:  Configuration parameters for this halo instance.
          privacy_tracker: A PrivacyTracker object that will be used to track
            privacy budget consumption for this simulation.
        """
        self._halo = halo
        self._params = params
        self._privacy_tracker = privacy_tracker
        self._modeling_strategy = modeling_strategy

    def fit_model(self, budget: PrivacyBudget) -> None:
        """Fits a model using data_set, parameters and modeling_strategy.

        budget:  A PrivacyBudget object specifying how much privacy budget
          is to be consumed when fitting this model.
        """
        self._model = self._modeling_strategy.fit(self._halo, self._params, budget)

    def true_reach_by_spend(
        self, spends: List[float], max_frequency: int = 1
    ) -> ReachPoint:
        """Returns the true reach obtained for a given spend vector.

        Args:
            spends:  The hypothetical amount spend vector, equal in length to
              the number of publishers.  spends[i] is the amount that is
              spent with publisher i.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            A ReachPoint representing the true reach that would have been
            obtained for this spend allocation.
        """
        return self._halo.true_reach_by_spend(spends, max_frequency)

    def modeled_reach_by_spend(
        self, spends: List[float], max_frequency: int = 1
    ) -> ReachPoint:
        """Returns the reach estimated via the reach surface model.

        Args:
            spends:  The hypothetical amount spend vector, equal in length to
              the number of publishers.  spends[i] is the amount that is
              spent with publisher i.
            max_frequency:  The maximum frequency for which to report reach.
        Returns:
            A ReachPoint representing the true reach that would have been
            obtained for this spend allocation.
        """
        return self._model.by_spend(spends, max_frequency)
