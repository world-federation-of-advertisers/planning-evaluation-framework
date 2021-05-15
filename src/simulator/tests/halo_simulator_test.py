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
"""Tests for halo_simulator.py."""

from absl.testing import absltest
import numpy as np
from unittest.mock import patch

from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.simulator.halo_simulator import HaloSimulator
from wfa_planning_evaluation_framework.simulator.publisher import Publisher
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyBudget
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyTracker
from wfa_planning_evaluation_framework.simulator.simulation_parameters import (
    LiquidLegionsParameters,
    SimulationParameters,
)


class FakeLaplaceMechanism:
    def __call__(self, x):
        return [2 * y for y in x]


class HaloSimulatorTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "test")

        cls.params = SimulationParameters(
            [0.04, 0.05], LiquidLegionsParameters(), np.random.default_rng(1)
        )
        cls.privacy_tracker = PrivacyTracker()
        cls.halo = HaloSimulator(data_set, cls.params, cls.privacy_tracker)

    def test_true_reach_by_spend(self):
        reach_point = self.halo.true_reach_by_spend([0.04, 0.04], 3)
        self.assertEqual(reach_point.reach(1), 2)
        self.assertEqual(reach_point.reach(2), 2)
        self.assertEqual(reach_point.reach(3), 0)

    def test_simulated_reach_by_spend_no_privacy(self):
        reach_point = self.halo.simulated_reach_by_spend(
            [0.04, 0.04], PrivacyBudget(100.0, 0.0), 0.5, 3
        )
        self.assertEqual(reach_point.reach(1), 4)
        self.assertEqual(reach_point.reach(2), 2)
        self.assertEqual(reach_point.reach(3), 0)

    @patch(
        "wfa_planning_evaluation_framework.simulator.halo_simulator.LaplaceMechanism"
    )
    def test_simulated_reach_by_spend_with_privacy(self, mock_laplace_mechanism):
        mock_laplace_mechanism.return_value = FakeLaplaceMechanism()
        reach_point = self.halo.simulated_reach_by_spend(
            [0.04, 0.04], PrivacyBudget(1.0, 0.0), 0.5, 3
        )
        self.assertTrue(reach_point.reach(1) >= 0)

    def test_privacy_tracker(self):
        self.assertEqual(self.halo.privacy_tracker.mechanisms, [])
        reach_point = self.halo.simulated_reach_by_spend(
            [0.04, 0.04], PrivacyBudget(1.0, 0.0), 0.5, 3
        )
        self.assertEqual(
            self.halo.privacy_tracker.mechanisms,
            ["Discrete Gaussian", "Discrete Gaussian"],
        )


if __name__ == "__main__":
    absltest.main()
