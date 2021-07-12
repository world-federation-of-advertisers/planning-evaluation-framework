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
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    LiquidLegionsParameters,
    SystemParameters,
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

        cls.params = SystemParameters(
            [0.4, 0.5], LiquidLegionsParameters(), np.random.default_rng(1)
        )
        cls.privacy_tracker = PrivacyTracker()
        cls.halo = HaloSimulator(data_set, cls.params, cls.privacy_tracker)

    def test_max_spends(self):
        self.assertEqual(self.halo.max_spends, (0.05, 0.06))

    def test_campaign_spends(self):
        expected = (0.02, 0.03)
        actual = self.halo.campaign_spends
        self.assertLen(actual, len(expected))
        for i in range(len(actual)):
            self.assertAlmostEqual(
                actual[i],
                expected[i],
                msg=f"At position {i} got {actual[i]} expected {expected[i]}",
                delta=0.0001,
            )

    def test_true_reach_by_spend(self):
        reach_point = self.halo.true_reach_by_spend([0.04, 0.04], 3)
        self.assertEqual(reach_point.reach(1), 2)
        self.assertEqual(reach_point.reach(2), 2)
        self.assertEqual(reach_point.reach(3), 0)

    def test_simulated_reach_by_spend_no_privacy(self):
        reach_point = self.halo.simulated_reach_by_spend(
            [0.04, 0.04], PrivacyBudget(100.0, 0.0), 0.5, 3
        )
        self.assertEqual(reach_point.reach(1), 3)
        self.assertEqual(reach_point.reach(2), 1)
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

    def test_form_venn_diagram_regions_with_2_inactive_publishers_and_1plus_reach(self):
        pdf1 = PublisherData([(1, 0.04)], "pdf1")
        pdf2 = PublisherData([(1, 0.04)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "test")
        params = SystemParameters(
            [0.4, 0.5], LiquidLegionsParameters(), np.random.default_rng(1)
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)

        spends = [0.01, 0.01]
        max_freq = 1
        expected_pub_set_by_region = {}
        expected_regions = {}
        pub_set_by_region, regions = halo._form_venn_diagram_regions(spends, max_freq)
        self.assertEqual(expected_pub_set_by_region, pub_set_by_region)
        self.assertEqual(expected_regions, regions)

    def test_form_venn_diagram_regions_with_simple_2_publishers_and_1plus_reach(self):
        pdf1 = PublisherData([(1, 0.01)], "pdf1")
        pdf2 = PublisherData([(1, 0.01)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "test")
        params = SystemParameters(
            [0.4, 0.5], LiquidLegionsParameters(), np.random.default_rng(1)
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)

        spends = [0.01, 0.01]
        max_freq = 1
        expected_pub_set_by_region = {1: set([0]), 3: set([0, 1])}
        expected_regions = {3: [1]}
        pub_set_by_region, regions = halo._form_venn_diagram_regions(spends, max_freq)
        self.assertEqual(expected_pub_set_by_region, pub_set_by_region)
        self.assertEqual(expected_regions, regions)

    def test_form_venn_diagram_regions_with_2_publishers_and_1plus_reach(self):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "test")
        params = SystemParameters(
            [0.4, 0.5], LiquidLegionsParameters(), np.random.default_rng(1)
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)

        spends = [0.04, 0.04]
        max_freq = 1
        expected_pub_set_by_region = {1: set([0]), 3: set([0, 1])}
        expected_regions = {1: [1], 3: [1]}
        pub_set_by_region, regions = halo._form_venn_diagram_regions(spends, max_freq)
        self.assertEqual(expected_pub_set_by_region, pub_set_by_region)
        self.assertEqual(expected_regions, regions)

    def test_form_venn_diagram_regions_with_2_publishers_and_2plus_reach(self):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "test")
        params = SystemParameters(
            [0.4, 0.5], LiquidLegionsParameters(), np.random.default_rng(1)
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)

        spends = [0.05, 0.08]
        max_freq = 2
        expected_pub_set_by_region = {
            1: set([0]),
            2: set([1]),
            3: set([0, 1]),
        }
        expected_regions = {1: [2, 1], 2: [1, 0], 3: [1, 1]}
        pub_set_by_region, regions = halo._form_venn_diagram_regions(spends, max_freq)
        self.assertEqual(expected_pub_set_by_region, pub_set_by_region)
        self.assertEqual(expected_regions, regions)

    def test_form_venn_diagram_regions_with_3_publishers_and_1plus_reach(self):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
        pdf3 = PublisherData([(2, 0.01), (3, 0.03), (4, 0.05)], "pdf3")
        data_set = DataSet([pdf1, pdf2, pdf3], "test")
        params = SystemParameters(
            [0.4, 0.5, 0.4], LiquidLegionsParameters(), np.random.default_rng(1)
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)

        spends = [0.04, 0.04, 0.0]
        max_freq = 1
        expected_pub_set_by_region = {1: set([0]), 3: set([0, 1])}
        expected_regions = {1: [1], 3: [1]}
        pub_set_by_region, regions = halo._form_venn_diagram_regions(spends, max_freq)
        self.assertEqual(expected_pub_set_by_region, pub_set_by_region)
        self.assertEqual(expected_regions, regions)

    def test_form_venn_diagram_regions_with_3_publishers_and_2plus_reach(self):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
        pdf3 = PublisherData([(2, 0.01), (3, 0.03), (4, 0.05)], "pdf3")
        data_set = DataSet([pdf1, pdf2, pdf3], "test")
        params = SystemParameters(
            [0.4, 0.5, 0.4], LiquidLegionsParameters(), np.random.default_rng(1)
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)

        spends = [0.05, 0.08, 0.0]
        max_freq = 2
        expected_pub_set_by_region = {
            1: set([0]),
            2: set([1]),
            3: set([0, 1]),
        }
        expected_regions = {1: [2, 1], 2: [1, 0], 3: [1, 1]}
        pub_set_by_region, regions = halo._form_venn_diagram_regions(spends, max_freq)
        self.assertEqual(expected_pub_set_by_region, pub_set_by_region)
        self.assertEqual(expected_regions, regions)

    def test_privacy_tracker(self):
        self.assertEqual(self.halo.privacy_tracker.mechanisms, [])
        reach_point = self.halo.simulated_reach_by_spend(
            [0.04, 0.04], PrivacyBudget(1.0, 0.0), 0.5, 3
        )
        self.assertEqual(
            self.halo.privacy_tracker.mechanisms,
            ["Discrete Gaussian", "Discrete Gaussian"],
        )

    def test_class_setup_with_campaign_spend_fractions_generator(self):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "test")
        params = SystemParameters(
            liquid_legions=LiquidLegionsParameters(),
            generator=np.random.default_rng(1),
            campaign_spend_fractions_generator=lambda npublishers: [0.2] * npublishers,
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)
        self.assertAlmostEqual(halo._campaign_spends[0], 0.01, 7)
        # using assertAlmostEqual here because of a rounding error
        self.assertAlmostEqual(halo._campaign_spends[1], 0.012, 7)


if __name__ == "__main__":
    absltest.main()
