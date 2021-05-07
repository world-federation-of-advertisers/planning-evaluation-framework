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

from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.simulator.halo_simulator import HaloSimulator
from wfa_planning_evaluation_framework.simulator.simulation_parameters import (
    SimulationParameters,
)


class ModelingStrategy:
    def fit(
        self,
        halo: HaloSimulator,
        params: SimulationParameters,
        epsilon: float,
        delta: float,
    ) -> ReachSurface:
        """Returns the reach surface using this Halo instance.

        Args:
            halo: A Halo object for simulating the behavior of the Halo system.
            params:  Simulation parameters.
            epsilon: The epsilon component of the privacy budget allocated to
              this request.
            delta:  The delta component of the privacy budget allocated to this
              requiest.
        Returns:
            A differentially private ReachSurface model which can be queried
            for reach and frequency estimates for arbitrary spend allocations.
        """
        raise NotImplementedError()
