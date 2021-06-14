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
"""A trivial modeling strategy for use with single publisher models."""

from typing import Dict
from typing import Type

from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.simulator.halo_simulator import HaloSimulator
from wfa_planning_evaluation_framework.simulator.modeling_strategy import (
    ModelingStrategy,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyBudget
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    SystemParameters,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    PrivacyBudget,
)


# The following is the maximum frequency used for obtaining measurements
# from the Halo simulator.
MAX_MEASUREMENT_FREQUENCY = 20

class SinglePublisherStrategy(ModelingStrategy):
    """A trival modeling strategy for use with single publisher models."""

    def fit(
        self, halo: HaloSimulator, params: SystemParameters, budget: PrivacyBudget
    ) -> ReachSurface:
        """Returns a reach curve for a single publisher modeling strategy.

        Args:
            halo: A Halo object for simulating the behavior of the Halo system.
            params:  Simulation parameters.
            budget:  A PrivacyBudget object specifying how much privacy budget
              is to be consumed for this operation.
        Returns:
            A differentially private ReachSurface model which can be queried
            for reach and frequency estimates for arbitrary spend allocations.
        """

        if halo.publisher_count != 1:
            raise ValueError("SinglePublisherStrategy cannot be used with multiple publishers")

        total_reach = halo.simulated_reach_by_spend(
            halo.campaign_spends, budget, max_frequency=MAX_MEASUREMENT_FREQUENCY
        )

        curve = self._single_pub_model(
            [total_reach], **self._single_pub_model_kwargs
        )
        curve._fit()
        return curve
