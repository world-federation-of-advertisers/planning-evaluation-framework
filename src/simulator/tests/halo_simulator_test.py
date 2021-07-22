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
from absl.testing import parameterized
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


class HaloSimulatorTest(parameterized.TestCase):
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

    @parameterized.parameters(
        [1, 0],
        [1e5, 731820],
    )
    def test_cardinality_estimate_variance(self, cardinality, variance):
        self.assertAlmostEqual(
            self.halo._cardinality_estimate_variance(cardinality),
            variance,
            0,
            msg=f"The variance for estimating n={cardinality} is not correct.",
        )

    @parameterized.parameters(
        [1, np.random.default_rng(0), 1],
        [1e5, np.random.default_rng(0), 8259],
    )
    def test_num_active_registers(self, cardinality, random_state, num_active):
        self.assertEqual(
            self.halo._num_active_registers(cardinality, random_state),
            num_active,
            msg=f"The number of active registers for n={cardinality} is not correct.",
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "with_1_region",
            "regions": {3: [1]},
            "sample_size": 1,
            "random_generator": np.random.default_rng(0),
            "expected": {3: [1]},
        },
        {
            "testcase_name": "with_10_regions",
            "regions": {i: [i ** 2 + 1] for i in range(10)},
            "sample_size": 20,
            "random_generator": np.random.default_rng(0),
            "expected": {i: [n] for i, n in enumerate([0, 0, 0, 1, 1, 1, 2, 3, 4, 8])},
        },
    )
    def test_sample_venn_diagram(
        self, regions, sample_size, random_generator, expected
    ):
        self.assertEqual(
            self.halo._sample_venn_diagram(regions, sample_size, random_generator),
            expected,
        )

    def test_sample_venn_diagram_with_invalid_input(self):
        with self.assertRaises(ValueError):
            self.halo._sample_venn_diagram({3: [1]}, 20)

    @parameterized.named_parameters(
        # testcase_name, num_publishers, spends, regions, expected
        {
            "testcase_name": "with_empty_regions",
            "num_publishers": 2,
            "spends": [0.005, 0.01],
            "regions": {},
            "expected": [
                ReachPoint([0, 0], [0], [0.005, 0]),
                ReachPoint([0, 0], [0], [0, 0.005]),
                ReachPoint([0, 0], [0], [0.005, 0.005]),
            ],
        },
        {
            "testcase_name": "without_active_publishers",
            "num_publishers": 2,
            "spends": [0, 0],
            "regions": {},
            "expected": [],
        },
        {
            "testcase_name": "with_1_active_pub_from_2_pubs",
            "num_publishers": 2,
            "spends": [0.04, 0.02],
            "regions": {1: [2]},
            "expected": [
                ReachPoint([3, 0], [2], [0.04, 0]),
                ReachPoint([0, 0], [0], [0, 0.04]),
                ReachPoint([3, 0], [2], [0.04, 0.04]),
            ],
        },
        {
            "testcase_name": "with_2_active_pubs",
            "num_publishers": 2,
            "spends": [0.04, 0.04],
            "regions": {1: [1], 3: [1]},
            "expected": [
                ReachPoint([3, 0], [2], [0.04, 0]),
                ReachPoint([0, 1], [1], [0, 0.04]),
                ReachPoint([3, 1], [2], [0.04, 0.04]),
            ],
        },
        {
            "testcase_name": "with_2_active_pubs_from_3_pubs",
            "num_publishers": 3,
            "spends": [0.05, 0.08, 0.0],
            "regions": {1: [2, 1], 2: [1, 0], 3: [1, 1]},
            "expected": [
                ReachPoint([4, 0, 0], [3], [0.05, 0, 0]),
                ReachPoint([0, 2, 0], [2], [0, 0.08, 0]),
                ReachPoint([4, 2, 0], [4], [0.05, 0.08, 0]),
            ],
        },
    )
    def test_generate_reach_points_from_venn_diagram(
        self, num_publishers, spends, regions, expected
    ):
        pdfs = [
            PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1"),
            PublisherData([(2, 0.03), (4, 0.06)], "pdf2"),
            PublisherData([(2, 0.01), (3, 0.03), (4, 0.05)], "pdf3"),
        ]
        data_set = DataSet(pdfs[:num_publishers], "test")
        params = SystemParameters(
            [0.4] * num_publishers,
            LiquidLegionsParameters(),
            np.random.default_rng(1),
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)

        reach_points = halo._generate_reach_points_from_venn_diagram(spends, regions)

        self.assertEqual(len(reach_points), len(expected))

        for i, (r_pt, expected_r_pt) in enumerate(zip(reach_points, expected)):
            self.assertEqual(
                r_pt.impressions,
                expected_r_pt.impressions,
                msg=f"The impressions of No.{i + 1} reach point is not correct",
            )
            self.assertEqual(
                r_pt.reach(1),
                expected_r_pt.reach(1),
                msg=f"The reach of No.{i + 1} reach point is not correct",
            )

    @parameterized.named_parameters(
        # testcase_name, pub_ids, regions, expected
        ("without_publisher", [], {1: [1], 3: [1]}, 0),
        ("without_region", [0, 1], {}, 0),
        ("with_1R1P_1plus_reach", [0], {3: [1]}, 1),
        ("with_1R2P_1plus_reach", [0, 1], {3: [1]}, 1),
        ("with_3R1P_1plus_reach", [0], {1: [2], 2: [1], 3: [1]}, 3),
        ("with_3R2P_1plus_reach", [0, 1], {1: [2], 2: [1], 3: [1]}, 4),
        ("with_1R1P_2plus_reach", [0], {3: [1, 1]}, 1),
        ("with_1R2P_2plus_reach", [0, 1], {3: [1, 1]}, 1),
        ("with_3R1P_2plus_reach", [0], {1: [2, 1], 2: [1, 1], 3: [1, 1]}, 3),
        ("with_3R2P_2plus_reach", [0, 1], {1: [2, 1], 2: [1, 1], 3: [1, 1]}, 4),
    )
    def test_aggregate_reach_in_primitive_venn_diagram_regions(
        self, pub_ids, regions, expected
    ):
        agg_reach = self.halo._aggregate_reach_in_primitive_venn_diagram_regions(
            pub_ids, regions
        )
        self.assertEqual(agg_reach, expected)

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
