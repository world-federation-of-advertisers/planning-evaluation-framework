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
    fixed_noise: float = 1.0

    def __call__(self, estimate):
        return estimate + self.fixed_noise


class FakeRandomGenerator:
    def multinomial(self, n, pval, size=None):
        result = np.zeros_like(pval)
        result[0] = 1
        return result

    def multivariate_hypergeometric(self, colors, nsample):
        samples = [0] * len(colors)
        index = 0

        while nsample:
            if samples[index] < colors[index]:
                samples[index] += 1
                nsample -= 1
            index = (index + 1) % len(samples)

        return samples

    def integers(self, low, high=None, size=None):
        return np.random.default_rng().integers(low=low, high=high, size=size)

    def normal(self, loc=0.0, scale=1.0):
        return loc + scale


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
        self.assertEqual(reach_point.reach(1), 2)
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

    @patch(
        "wfa_planning_evaluation_framework.simulator.halo_simulator.StandardizedHistogramEstimator.estimate_cardinality"
    )
    def test_simulated_reach_by_spend_with_negative_noise(
        self, mock_estimate_cardinality
    ):
        mock_estimate_cardinality.return_value = [10, 5, -1, 0, -1]
        reach_point = self.halo.simulated_reach_by_spend(
            [1.0, 1.0], PrivacyBudget(1.0, 0.0), 0.5, 5
        )
        self.assertTrue(all([reach_point.reach(i + 1) >= 0 for i in range(5)]))

    @parameterized.parameters(
        [1, 0],
        [1e5, 731820],
    )
    def test_liquid_legions_cardinality_estimate_variance(self, cardinality, variance):
        self.assertAlmostEqual(
            self.halo._liquid_legions_cardinality_estimate_variance(cardinality),
            variance,
            0,
            msg=f"The variance for estimating n={cardinality} is not correct.",
        )

    @parameterized.parameters(
        [1, 1],
        [1e5, 8251],
    )
    def test_liquid_legions_num_active_regions(self, cardinality, num_active):
        self.assertEqual(
            self.halo._liquid_legions_num_active_regions(cardinality),
            num_active,
            msg=f"The number of active registers for n={cardinality} is not correct.",
        )

    def test_simulated_venn_diagram_reach_by_spend_without_active_pub(self):
        pdfs = [
            PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1"),
            PublisherData([(2, 0.03), (4, 0.06)], "pdf2"),
            PublisherData([(2, 0.01), (3, 0.03), (4, 0.05)], "pdf3"),
        ]
        data_set = DataSet(pdfs, "test")
        params = SystemParameters(
            [0.4, 0.5, 0.4],
            LiquidLegionsParameters(),
            FakeRandomGenerator(),
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)

        spends = [0, 0, 0]
        budget = PrivacyBudget(0.2, 0.4)
        privacy_budget_split = 0.5
        max_freq = 1

        reach_points = halo.simulated_venn_diagram_reach_by_spend(
            spends, budget, privacy_budget_split, max_freq
        )

        expected_reach_points = []

        self.assertEqual(expected_reach_points, reach_points)
        self.assertEqual(halo.privacy_tracker.privacy_consumption.epsilon, 0)
        self.assertEqual(halo.privacy_tracker.privacy_consumption.delta, 0)
        self.assertEqual(len(halo.privacy_tracker._noising_events), 0)

    @parameterized.named_parameters(
        {
            # true cardinality = 3, sample_size = 1, std = 1, fixed_noise = -10'
            # original_primitive_regions = {2^2: 3}
            # sampled_regions = {2^2: 1}
            # noised_regions = {2^2: 1 - 10}
            # cardinality_estimate = (3 + 1) - 10 = 0
            # scaled_regions = {2^2: 0}
            "testcase_name": "with_large_negative_noise",
            "spends": [0, 0, 0.05],
            "budget": PrivacyBudget(0.2, 0.4),
            "privacy_budget_split": 0.5,
            "fixed_noise": -10.0,
            "expected_reach_points": [ReachPoint([0, 0, 3], [0], [0, 0, 0.05])],
        },
        {
            # true cardinality = 3, sample_size = 1, std = 1, fixed_noise = 1
            # original_primitive_regions = {2^2: 3}
            # sampled_regions = {2^2: 1}
            # noised_regions = {2^2: 1 + 1}
            # cardinality_estimate = (3 + 1) + 1 = 5
            # scaled_regions = {2^2: 5}
            "testcase_name": "with_1_active_pub",
            "spends": [0, 0, 0.05],
            "budget": PrivacyBudget(0.2, 0.4),
            "privacy_budget_split": 0.5,
            "fixed_noise": 1.0,
            "expected_reach_points": [ReachPoint([0, 0, 3], [5], [0, 0, 0.05])],
        },
        {
            # true cardinality = 3, sample_size = 1, std = 1, fixed_noise = 2
            # original_primitive_regions = {2^1: 0, 2^2: 2, 2^1 + 2^2: 1}
            # sampled_regions = {2^1: 0, 2^2: 1, 2^1 + 2^2: 0}
            # noised_regions = {2^1: 2, 2^2: 3, 2^1 + 2^2: 2}
            # cardinality_estimate = (3 + 1) + 2 = 6
            # scaled_regions = {2^1: 12 / 7, 2^2: 18 / 7, 2^1 + 2^2: 12 / 7}
            "testcase_name": "with_2_active_pubs",
            "spends": [0, 0.04, 0.05],
            "budget": PrivacyBudget(0.2, 0.4),
            "privacy_budget_split": 0.3,
            "fixed_noise": 2.0,
            "expected_reach_points": [
                ReachPoint([0, 1, 0], [24 / 7], [0, 0.04, 0]),
                ReachPoint([0, 0, 3], [30 / 7], [0, 0, 0.05]),
                ReachPoint([0, 1, 3], [42 / 7], [0, 0.04, 0.05]),
            ],
        },
        {
            # true cardinality = 3, sample_size = 1, std = 1, fixed_noise = 2
            # original_primitive_regions = {2^1: 0, 2^2: 3, 2^1 + 2^2: 0}
            # sampled_regions = {2^1: 0, 2^2: 1, 2^1 + 2^2: 0}
            # noised_regions = {2^1: 2, 2^2: 3, 2^1 + 2^2: 2}
            # cardinality_estimate = (3 + 1) + 2 = 6
            # scaled_regions = {2^1: 12 / 7, 2^2: 18 / 7, 2^1 + 2^2: 12 / 7}
            "testcase_name": "with_1_active_pub_and_1_inactive_pub",
            "spends": [0.0, 0.02, 0.05],
            "budget": PrivacyBudget(0.2, 0.4),
            "privacy_budget_split": 0.3,
            "fixed_noise": 2.0,
            "expected_reach_points": [
                ReachPoint([0, 0, 0], [24 / 7], [0, 0.02, 0]),
                ReachPoint([0, 0, 3], [30 / 7], [0, 0, 0.05]),
                ReachPoint([0, 0, 3], [42 / 7], [0, 0.02, 0.05]),
            ],
        },
    )
    @patch.object(
        HaloSimulator, "_liquid_legions_cardinality_estimate_variance", return_value=1.0
    )
    @patch(
        "wfa_planning_evaluation_framework.simulator.halo_simulator.GeometricEstimateNoiser"
    )
    def test_simulated_venn_diagram_reach_by_spend(
        self,
        mock_geometric_estimate_noiser,
        mock_cardinality_estimate_variance,
        spends,
        budget,
        privacy_budget_split,
        fixed_noise,
        expected_reach_points,
    ):
        mock_geometric_estimate_noiser.return_value = FakeNoiser(fixed_noise)

        pdfs = [
            PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1"),
            PublisherData([(2, 0.03), (4, 0.06)], "pdf2"),
            PublisherData([(2, 0.01), (3, 0.03), (4, 0.05)], "pdf3"),
        ]
        data_set = DataSet(pdfs, "test")
        params = SystemParameters(
            [0.4, 0.5, 0.4],
            LiquidLegionsParameters(),
            FakeRandomGenerator(),
        )
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)

        reach_points = halo.simulated_venn_diagram_reach_by_spend(
            spends, budget, privacy_budget_split
        )

        # Examine reach points
        for i, (r_pt, expected_r_pt) in enumerate(
            zip(reach_points, expected_reach_points)
        ):
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

        # Examine privacy tracker
        expected_noise_event_primitive_regions = NoisingEvent(
            PrivacyBudget(
                budget.epsilon * privacy_budget_split,
                budget.delta * privacy_budget_split,
            ),
            DP_NOISE_MECHANISM_DISCRETE_LAPLACE,
            {"privacy_budget_split": privacy_budget_split},
        )

        expected_noise_event_cardinality = NoisingEvent(
            PrivacyBudget(
                budget.epsilon * (1 - privacy_budget_split),
                budget.delta * (1 - privacy_budget_split),
            ),
            DP_NOISE_MECHANISM_DISCRETE_LAPLACE,
            {"privacy_budget_split": (1 - privacy_budget_split)},
        )

        expected_noise_events = [
            expected_noise_event_primitive_regions,
            expected_noise_event_cardinality,
        ]

        self.assertEqual(
            halo.privacy_tracker.privacy_consumption.epsilon,
            expected_noise_event_primitive_regions.budget.epsilon
            + expected_noise_event_cardinality.budget.epsilon,
        )
        self.assertEqual(
            halo.privacy_tracker.privacy_consumption.delta,
            expected_noise_event_primitive_regions.budget.delta
            + expected_noise_event_cardinality.budget.delta,
        )
        self.assertEqual(len(halo.privacy_tracker._noising_events), 2)

        for noise_event, expected_noise_event in zip(
            halo.privacy_tracker._noising_events, expected_noise_events
        ):
            self.assertEqual(
                noise_event.budget.epsilon,
                expected_noise_event.budget.epsilon,
            )
            self.assertEqual(
                noise_event.budget.delta,
                expected_noise_event.budget.delta,
            )
            self.assertEqual(
                noise_event.mechanism,
                expected_noise_event.mechanism,
            )
            self.assertEqual(
                noise_event.params,
                expected_noise_event.params,
            )

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

    @parameterized.named_parameters(
        # testcase_name, num_publishers, spends, regions, expected
        {
            "testcase_name": "with_2_inactive_pubs_and_1plus_reaches",
            "num_publishers": 2,
            "spends": [0.005, 0.01],
            "max_freq": 1,
            "expected": {
                1: [0],
                2: [0],
                3: [0],
            },
        },
        {
            "testcase_name": "with_2_active_pubs_and_1plus_reaches",
            "num_publishers": 2,
            "spends": [0.04, 0.04],
            "max_freq": 1,
            "expected": {
                1: [1],
                2: [0],
                3: [1],
            },
        },
        {
            "testcase_name": "with_2_active_pubs_and_2plus_reaches",
            "num_publishers": 2,
            "spends": [0.05, 0.08],
            "max_freq": 2,
            "expected": {
                1: [2, 1],
                2: [1, 0],
                3: [1, 1],
            },
        },
        {
            "testcase_name": "with_2_active_pubs_out_of_3_and_1plus_reaches",
            "num_publishers": 3,
            "spends": [0.04, 0.04, 0.0],
            "max_freq": 1,
            "expected": {
                1: [1],
                2: [0],
                3: [1],
            },
        },
        {
            "testcase_name": "with_2_active_pubs_out_of_3_and_2plus_reaches",
            "num_publishers": 3,
            "spends": [0.05, 0.08, 0.0],
            "max_freq": 2,
            "expected": {1: [2, 1], 2: [1, 0], 3: [1, 1]},
        },
        {
            "testcase_name": "with_3_active_pubs_and_1plus_reaches",
            "num_publishers": 3,
            "spends": [0.05, 0.08, 0.03],
            "max_freq": 1,
            "expected": {
                1: [1],
                2: [1],
                3: [0],
                4: [0],
                5: [1],
                6: [0],
                7: [1],
            },
        },
        {
            "testcase_name": "with_3_active_pubs_and_2plus_reaches",
            "num_publishers": 3,
            "spends": [0.05, 0.08, 0.03],
            "max_freq": 2,
            "expected": {
                1: [1, 1],
                2: [1, 0],
                3: [0, 0],
                4: [0, 0],
                5: [1, 1],
                6: [0, 0],
                7: [1, 1],
            },
        },
    )
    def test_form_venn_diagram_regions(
        self, num_publishers, spends, max_freq, expected
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

        regions = halo._form_venn_diagram_regions(spends, max_freq)
        self.assertEqual(expected, regions)

    @parameterized.named_parameters(
        {
            "testcase_name": "with_1_region_and_1plus_reaches",
            "regions": {3: [1]},
            "sample_size": 1,
            "expected": {3: 1},
        },
        # regions = [1, 2, 5, 10, 17]
        {
            "testcase_name": "with_5_regions_1plus_reaches_and_fake_rng",
            "regions": {i: [i ** 2 + 1] for i in range(5)},
            "sample_size": 20,
            "expected": {i: n for i, n in enumerate([1, 2, 5, 6, 6])},
        },
        # regions = [[0, 0], [2, 1], [0, 0], [10, 3], [0, 0], [26, 5]]
        {
            "testcase_name": "with_6_regions_2plus_reaches_and_fake_rng",
            "regions": {i: [i ** 2 + 1, i] if i % 2 else [0, 0] for i in range(6)},
            "sample_size": 20,
            "expected": {i: n for i, n in enumerate([0, 2, 0, 9, 0, 9])},
        },
    )
    def test_sample_venn_diagram(self, regions, sample_size, expected):
        params = SystemParameters([0], LiquidLegionsParameters(), FakeRandomGenerator())
        halo = HaloSimulator(DataSet([], "test"), params, PrivacyTracker())
        self.assertEqual(
            halo._sample_venn_diagram(regions, sample_size),
            expected,
        )

    def test_sample_venn_diagram_with_invalid_input(self):
        with self.assertRaises(ValueError):
            self.halo._sample_venn_diagram({3: [1]}, 20)

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
            halo.privacy_tracker.privacy_consumption.epsilon, budget.epsilon * privacy_budget_split
        )
        self.assertEqual(
            halo.privacy_tracker.privacy_consumption.delta, budget.delta * privacy_budget_split
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
        {
            "testcase_name": "with_1_regions",
            "regions": {1: 1},
            "true_cardinality": 20,
            "std": 1,
            "budget": PrivacyBudget(0.2, 0.4),
            "privacy_budget_split": 0.7,
            "fixed_noise": 1,
            "expected": {1: 22},  # cardinality estimate = 20 + 1 + 1 = 22
        },
        {
            "testcase_name": "with_3_regions",
            "regions": {1: 2, 2: 2, 3: 1},
            "true_cardinality": 16,
            "std": 2,
            "budget": PrivacyBudget(0.1, 0.1),
            "privacy_budget_split": 0.5,
            "fixed_noise": 2,
            "expected": {1: 8, 2: 8, 3: 4},  # cardinality estimate = 16 + 2 + 2 = 20
        },
        {
            "testcase_name": "with_empty_regions",
            "regions": {1: 0, 2: 0, 3: 0},
            "true_cardinality": 20,
            "std": 1,
            "budget": PrivacyBudget(0.2, 0.4),
            "privacy_budget_split": 0.7,
            "fixed_noise": 1,
            "expected": {1: 0, 2: 0, 3: 0},
        },
    )
    @patch(
        "wfa_planning_evaluation_framework.simulator.halo_simulator.GeometricEstimateNoiser"
    )
    def test_scale_up_reach_in_primitive_regions(
        self,
        mock_geometric_estimate_noiser,
        regions,
        true_cardinality,
        std,
        budget,
        privacy_budget_split,
        fixed_noise,
        expected,
    ):
        mock_geometric_estimate_noiser.return_value = FakeNoiser(fixed_noise)

        params = SystemParameters([0], LiquidLegionsParameters(), FakeRandomGenerator())
        halo = HaloSimulator(DataSet([], "test"), params, PrivacyTracker())

        scaled_regions = halo._scale_up_reach_in_primitive_regions(
            regions, true_cardinality, std, budget, privacy_budget_split
        )

        self.assertEqual(scaled_regions, expected)

        self.assertEqual(
            halo.privacy_tracker.privacy_consumption.epsilon, budget.epsilon * privacy_budget_split
        )
        self.assertEqual(
            halo.privacy_tracker.privacy_consumption.delta, budget.delta * privacy_budget_split
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
        # testcase_name, pub_ids, regions, expected
        ("without_publisher", [], {1: 1, 2: 0, 3: 1}, 0),
        ("without_region", [0, 1], {}, 0),
        ("with_1_occupied_region_1_pub", [0], {1: 0, 2: 0, 3: 1}, 1),
        ("with_1_occupied_region_2_pubs", [0, 1], {1: 0, 2: 0, 3: 1}, 1),
        ("with_3_occupied_regions_1_pub", [0], {1: 2, 2: 1, 3: 1}, 3),
        ("with_3_occupied_regions_2_pubs", [0, 1], {1: 2, 2: 1, 3: 1}, 4),
    )
    def test_aggregate_reach_in_primitive_venn_diagram_regions(
        self, pub_ids, regions, expected
    ):
        agg_reach = self.halo._aggregate_reach_in_primitive_venn_diagram_regions(
            pub_ids, regions
        )
        self.assertEqual(agg_reach, expected)

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
            "testcase_name": "with_1_region_2_active_pubs",
            "num_publishers": 2,
            "spends": [0.04, 0.02],
            "regions": {1: 2},
            "expected": [
                ReachPoint([3, 0], [2], [0.04, 0]),
                ReachPoint([0, 0], [0], [0, 0.02]),
                ReachPoint([3, 0], [2], [0.04, 0.02]),
            ],
        },
        {
            "testcase_name": "with_3_regions_2_active_pubs",
            "num_publishers": 2,
            "spends": [0.04, 0.04],
            "regions": {1: 1, 2: 0, 3: 1},
            "expected": [
                ReachPoint([3, 0], [2], [0.04, 0]),
                ReachPoint([0, 1], [1], [0, 0.04]),
                ReachPoint([3, 1], [2], [0.04, 0.04]),
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
            campaign_spend_fractions_generator=lambda dataset: [0.2]
            * dataset.publisher_count,
        )
        params = params.update_from_dataset(data_set)
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(data_set, params, privacy_tracker)
        self.assertAlmostEqual(halo._campaign_spends[0], 0.01, 7)
        # using assertAlmostEqual here because of a rounding error
        self.assertAlmostEqual(halo._campaign_spends[1], 0.012, 7)


if __name__ == "__main__":
    absltest.main()
