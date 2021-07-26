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
"""Proposed modeling strategy for the M3 milestone."""

from typing import Dict
from typing import Type

from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
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


class M3Strategy(ModelingStrategy):
    """Modeling strategy proposed for implementation in M3 milestone."""

    def fit(
        self, halo: HaloSimulator, params: SystemParameters, budget: PrivacyBudget
    ) -> ReachSurface:
        """Returns the reach surface computed using the M3 proposal

        Args:
            halo: A Halo object for simulating the behavior of the Halo system.
            params:  Simulation parameters.
            budget:  A PrivacyBudget object specifying how much privacy budget
              is to be consumed for this operation.
        Returns:
            A differentially private ReachSurface model which can be queried
            for reach and frequency estimates for arbitrary spend allocations.
        """

        p = len(params.campaign_spend)

        # TODO: Compute total budget usage with advanced composition or PLD's
        per_request_budget = PrivacyBudget(
            budget.epsilon / (2 * p + 1), budget.delta / (2 * p + 1)
        )

        total_reach = halo.simulated_reach_by_spend(
            params.campaign_spend, per_request_budget
        )

        # Compute reach for each publisher
        single_pub_reach_list = []
        for i in range(p):
            spend_vec = [0.0] * p
            spend_vec[i] = params.campaign_spend[i]
            reach_point = halo.simulated_reach_by_spend(
                spend_vec, per_request_budget, max_frequency=10
            )
            kplus_reaches = [
                reach_point.reach(k) for k in range(1, reach_point.max_frequency + 1)
            ]
            single_pub_reach = ReachPoint(
                [reach_point.impressions[i]], kplus_reaches, [reach_point.spends[i]]
            )
            single_pub_reach_list.append(single_pub_reach)

        # Compute reach for all publishers but one
        all_but_one_reach = []
        if p > 2:
            for i in range(p):
                spend_vec = params.campaign_spend.copy()
                spend_vec[i] = 0.0
                reach = halo.simulated_reach_by_spend(spend_vec, per_request_budget)
                all_but_one_reach.append(reach)

        # Compute reach curve for each publisher
        single_pub_curves = []
        for i in range(p):
            curve = self._single_pub_model(
                [single_pub_reach_list[i]], **self._single_pub_model_kwargs
            )
            curve._fit()
            single_pub_curves.append(curve)

        if p == 1:
            return single_pub_curves[0]

        training_points = all_but_one_reach + [total_reach]
        reach_surface = self._multi_pub_model(
            single_pub_curves, training_points, **self._multi_pub_model_kwargs
        )

        return reach_surface
