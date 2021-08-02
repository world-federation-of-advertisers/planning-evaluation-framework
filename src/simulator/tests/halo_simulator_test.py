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
from dataclasses import dataclass

from wfa_cardinality_estimation_evaluation_framework.estimators.base import (
    EstimateNoiserBase,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.simulator.halo_simulator import (
    HaloSimulator,
    MAX_ACTIVE_PUBLISHERS,
)
from wfa_planning_evaluation_framework.simulator.publisher import Publisher
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    PrivacyBudget,
    PrivacyTracker,
    NoisingEvent,
    DP_NOISE_MECHANISM_DISCRETE_LAPLACE,
)
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    LiquidLegionsParameters,
    SystemParameters,
)


class FakeLaplaceMechanism:
    def __call__(self, x):
        return [2 * y for y in x]


@dataclass
class FakeNoiser(EstimateNoiserBase):
    fixed_noise: float

    def __call__(self, estimate):
        return estimate + self.fixed_noise


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

    def test_form_venn_diagram_regions_with_publishers_more_than_limit(self):
        num_publishers = MAX_ACTIVE_PUBLISHERS + 1
        data_set = DataSet(
            [PublisherData([(1, 0.01)], f"pdf{i + 1}") for i in range(num_publishers)],
            "test",
        )
        params = SystemParameters(
            [0.4] * num_publishers, LiquidLegionsParameters(), np.random.default_rng(1)
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)

        spends = [0.01] * num_publishers
        with self.assertRaises(ValueError):
            halo._form_venn_diagram_regions(spends)

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
        expected_regions = {}
        regions = halo._form_venn_diagram_regions(spends, max_freq)
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
        expected_regions = {3: [1]}
        regions = halo._form_venn_diagram_regions(spends, max_freq)
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
        expected_regions = {1: [1], 3: [1]}
        regions = halo._form_venn_diagram_regions(spends, max_freq)
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
        expected_regions = {1: [2, 1], 2: [1, 0], 3: [1, 1]}
        regions = halo._form_venn_diagram_regions(spends, max_freq)
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
        expected_regions = {1: [1], 3: [1]}
        regions = halo._form_venn_diagram_regions(spends, max_freq)
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
        expected_regions = {1: [2, 1], 2: [1, 0], 3: [1, 1]}
        regions = halo._form_venn_diagram_regions(spends, max_freq)
        self.assertEqual(expected_regions, regions)

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

    @parameterized.named_parameters(
        {
            "testcase_name": "with_1_regions",
            "regions": {1: 1},
            "budget": PrivacyBudget(0.2, 0.4),
            "privacy_budget_split": 0.7,
            "fixed_noise": 1,
            "expected_regions": {1: 2},
        },
        {
            "testcase_name": "with_3_regions",
            "regions": {1: 1, 2: 0, 3: 1},
            "budget": PrivacyBudget(0.1, 0.1),
            "privacy_budget_split": 0.3,
            "fixed_noise": 2,
            "expected_regions": {1: 3, 2: 2, 3: 3},
        },
    )
    @patch(
        "wfa_planning_evaluation_framework.simulator.halo_simulator.GeometricEstimateNoiser"
    )
    def test_add_dp_noise_to_primitive_regions(
        self,
        mock_geometric_estimate_noiser,
        regions,
        budget,
        privacy_budget_split,
        fixed_noise,
        expected_regions,
    ):
        mock_geometric_estimate_noiser.return_value = FakeNoiser(fixed_noise)

        halo = HaloSimulator(DataSet([], "test"), SystemParameters(), PrivacyTracker())

        noised_regions = halo._add_dp_noise_to_primitive_regions(
            regions, budget, privacy_budget_split
        )

        self.assertEqual(noised_regions, expected_regions)
        self.assertEqual(
            halo.privacy_tracker._epsilon_sum, budget.epsilon * privacy_budget_split
        )
        self.assertEqual(
            halo.privacy_tracker._delta_sum, budget.delta * privacy_budget_split
        )
        self.assertEqual(len(halo.privacy_tracker._noising_events), 1)
        self.assertEqual(
            halo.privacy_tracker._noising_events[0].budget.epsilon,
            budget.epsilon * privacy_budget_split,
        )
        self.assertEqual(
            halo.privacy_tracker._noising_events[0].budget.delta,
            budget.delta * privacy_budget_split,
        )
        self.assertEqual(
            halo.privacy_tracker._noising_events[0].mechanism,
            DP_NOISE_MECHANISM_DISCRETE_LAPLACE,
        )
        self.assertEqual(
            halo.privacy_tracker._noising_events[0].params,
            {"privacy_budget_split": privacy_budget_split},
        )

    @parameterized.named_parameters(
        # testcase_name, num_publishers, spends, regions, expected
        {
            "testcase_name": "with_empty_regions",
            "num_publishers": 2,
            "spends": [0.005, 0.01],
            "regions": {},
            "expected": [
                ReachPoint([0, 0], [0], [0.005, 0]),
                ReachPoint([0, 0], [0], [0, 0.01]),
                ReachPoint([0, 0], [0], [0.005, 0.01]),
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
                ReachPoint([0, 0], [0], [0, 0.02]),
                ReachPoint([3, 0], [2], [0.04, 0.02]),
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

        # Note that the reach points generated from the Venn diagram only
        # contain 1+ reaches.
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
            self.assertEqual(
                r_pt.spends,
                expected_r_pt.spends,
                msg=f"The spends of No.{i + 1} reach point is not correct",
            )

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
