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
"""A modeling strategy estimates reach surfaces using Halo instances."""

from typing import Dict
from typing import Type

from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.simulator.halo_simulator import HaloSimulator
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyBudget
from wfa_planning_evaluation_framework.simulator.simulation_parameters import (
    SimulationParameters,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    PrivacyBudget,
)


class ModelingStrategy:
    # TODO: Add args for pricing model when these are introduced.
    def __init__(
        self,
        single_pub_model: Type[ReachCurve],
        single_pub_model_kwargs: Dict,
        multi_pub_model: Type[ReachSurface],
        multi_pub_model_kwargs: Dict,
    ):
        """Initializes a modeling strategy object."""
        self._single_pub_model = single_pub_model
        self._single_pub_model_kwargs = single_pub_model_kwargs
        self._multi_pub_model = multi_pub_model
        self._multi_pub_model_kwargs = multi_pub_model_kwargs

    def fit(
        self, halo: HaloSimulator, params: SimulationParameters, budget: PrivacyBudget
    ) -> ReachSurface:
        """Returns the reach surface using this Halo instance.

        Args:
            halo: A Halo object for simulating the behavior of the Halo system.
            params:  Simulation parameters.
            budget:  A PrivacyBudget object specifying how much privacy budget
              is to be consumed for this operation.
        Returns:
            A differentially private ReachSurface model which can be queried
            for reach and frequency estimates for arbitrary spend allocations.
        """
        raise NotImplementedError()
