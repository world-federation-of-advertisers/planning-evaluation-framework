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
"""Class of a reach surface that predicts reach using local-DP sketches."""

from typing import List
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.simulator.local_dp_simulator import (
    LocalDpSimulator,
)


class LocalDpReachSurface(ReachSurface):
    """Predicts subset reach by deduplicating local DP LiquidLegions."""

    def __init__(self, local_dp_simulator: LocalDpSimulator):
        self.local_dp_simulator = local_dp_simulator
        # experimental_trial requires saving the training data as ReachPoints
        # in any ReachSurface, to "record the max frequency in the data produced
        # by halo".  See the following line:
        # https://github.com/world-federation-of-advertisers/planning-evaluation-framework/blob/18e63b167b5fa444dc44ddb29c8cc4c9c0097c0e/src/driver/experimental_trial.py#L339
        # This is not applicable to the LocalDpReachSurface because the local DP
        # approach only produces 1+ reach.  A principled modification is to
        # refactor experimental_trial to exclude the requirement that every
        # ReachSurface should have the self._data: List[ReachPoint] attribute.
        #
        # But, it seems tedious to refactor experimental_trial. So here, for
        # simplicity, we add a fake self._data attribute in LocalDpReachSurface.
        # This fake attribute is only for interoperability.
        self._data = [
            ReachPoint(
                impressions=[0],
                kplus_reaches=[0],
            )
        ]

    def by_spend(self, spend: List[float], max_frequency: int = 1) -> ReachPoint:
        """Returns the estimated ReachPoint at a spend allocation."""
        return self.local_dp_simulator.simulated_reach_by_spend(spend, max_frequency)
