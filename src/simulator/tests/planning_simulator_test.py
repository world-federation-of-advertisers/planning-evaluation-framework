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
"""Tests for planning_simulator.py."""

from absl.testing import absltest
from collections import defaultdict
import numpy as np
from unittest.mock import patch

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.simulator.halo_simulator import HaloSimulator
from wfa_planning_evaluation_framework.simulator.modeling_strategy import (
    ModelingStrategy,
)
from wfa_planning_evaluation_framework.simulator.planner_simulator import (
    PlannerSimulator,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyBudget
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyTracker
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    LiquidLegionsParameters,
    SystemParameters,
)


class FakeHaloSimulator:
    def true_reach_by_spend(self, spends, max_frequency):
        return ReachPoint([10], [3], [1])


class FakeReachSurface:
    def by_spend(self, spend, max_frequency: int = 1):
        return ReachPoint(spend, [4], spend)


class FakeModelingStrategy(ModelingStrategy):
    def __init__(self):
        pass

    def fit(self, halo, params, budget):
        return FakeReachSurface()


class PlanningSimulatorTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulator = PlannerSimulator(
            FakeHaloSimulator(),
            FakeModelingStrategy(),
            SystemParameters([], LiquidLegionsParameters(), np.random.default_rng(1)),
            PrivacyTracker(),
        )

    def test_fit_model(self):
        self.simulator.fit_model(PrivacyBudget(1.0, 2.0))
        self.assertIsNotNone(self.simulator._model)

    def test_true_reach_by_spend(self):
        self.assertEqual(self.simulator.true_reach_by_spend([10]).reach(1), 3)

    def test_modeled_reach_by_spend(self):
        self.assertEqual(self.simulator.modeled_reach_by_spend([10]).reach(1), 4)


if __name__ == "__main__":
    absltest.main()
