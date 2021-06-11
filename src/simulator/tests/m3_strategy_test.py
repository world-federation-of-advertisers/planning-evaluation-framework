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
"""Tests for M3 stragtegy."""

from absl.testing import absltest
import numpy as np
from typing import List

from wfa_planning_evaluation_framework.models.gamma_poisson_model import (
    GammaPoissonModel,
)
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.models.restricted_pairwise_union_reach_surface import (
    RestrictedPairwiseUnionReachSurface,
)
from wfa_planning_evaluation_framework.simulator.m3_strategy import M3Strategy
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    PrivacyBudget,
)
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    LiquidLegionsParameters,
    SystemParameters,
)


class FakeHalo:
    def __init__(self):
        hist = [8124, 5464, 3191, 1679, 815, 371, 159, 64, 23, 6, 0]
        self.curve = GammaPoissonModel([ReachPoint([10000.0], hist, [100.0])])
        self.curve._fit()

    @property
    def campaign_spends(self):
        return [100.0, 100.0]

    def simulated_reach_by_spend(
        self,
        spends: List[float],
        budget: PrivacyBudget,
        privacy_budget_split: float = 0.5,
        max_frequency: int = 1,
    ) -> ReachPoint:
        rp1 = self.curve.by_spend([spends[0]], max_frequency=max_frequency)
        rp2 = self.curve.by_spend([spends[1]], max_frequency=max_frequency)

        imps = rp1.impressions[0] + rp2.impressions[0]
        spend = rp1.spends[0] + rp2.spends[0]
        freqs = [rp1.reach(i) + rp2.reach(i) for i in range(1, max_frequency + 1)]
        return ReachPoint([imps], freqs, [spend])


class M3StrategyTest(absltest.TestCase):
    # TODO: Re-active the following test after bug is fixed in Pairwise Union Model.
    def donttest_m3_strategy(self):
        halo = FakeHalo()
        params = SystemParameters(
            [100.0, 100.0], LiquidLegionsParameters(), np.random.default_rng(seed=1)
        )
        budget = PrivacyBudget(1.0, 1e-5)
        m3strategy = M3Strategy(
            GammaPoissonModel, {}, RestrictedPairwiseUnionReachSurface, {}
        )
        surface = m3strategy.fit(halo, params, budget)

        expected0 = surface.by_spend([100.0, 0.0]).reach(1)
        actual0 = halo.simulated_reach_by_spend([100.0, 0.0], budget).reach(1)
        self.assertAlmostEqual(expected0, actual0, delta=10)

        expected1 = surface.by_spend([0.0, 100.0]).reach(1)
        actual1 = halo.simulated_reach_by_spend([0.0, 100.0], budget).reach(1)
        self.assertAlmostEqual(expected1, actual1, delta=10)

        expected2 = surface.by_spend([100.0, 100.0]).reach(1)
        actual2 = halo.simulated_reach_by_spend([100.0, 100.0], budget).reach(1)
        self.assertAlmostEqual(expected2, actual2, delta=10)


if __name__ == "__main__":
    absltest.main()
