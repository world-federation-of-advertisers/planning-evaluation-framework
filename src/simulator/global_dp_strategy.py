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
"""Proposed modeling strategy for the M3 milestone."""

from typing import Dict
from typing import Type


from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.models.global_dp_reach_surface import (
    GlobalDpReachSurface,
)
from wfa_planning_evaluation_framework.simulator.global_dp_simulator import (
    GlobalDpSimulator,
)
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


class GlobalDpLiquidlegionsStrategy(ModelingStrategy):
    """Modeling strategy that predicts by unioning local sketches."""

    def fit(
        self, halo: GlobalDpSimulator, params: SystemParameters, budget: PrivacyBudget
    ) -> ReachSurface:
        """Returns the reach surface computed using the Local DP approach.

        Args:
            halo:  A simulator of the Halo system. The simulator of this class should
                be set as the LocalDpSimulator.  Note that it is a hypothetical
                simulator, i.e., a simulator of a possible option of, instead of the
                actual Halo system.
                (I'm calling this arg as `halo` to be consistent with halo_simulator,
                for interoperability with experimental_trial.)
            params:  Simulation parameters.
            budget:  A PrivacyBudget object specifying how much privacy budget
              is to be consumed for this operation.

        Returns:
            A differentially private ReachSurface model which can be queried
            for reach and frequency estimates for arbitrary spend allocations.
        """
        halo.obtain_global_dp_estimates(budget)
        return GlobalDpReachSurface(halo)
