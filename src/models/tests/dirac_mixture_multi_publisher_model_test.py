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
"""Tests for dirac_mixture_multi_publisher_model.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from typing import List, Tuple
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.dirac_mixture_single_publisher_model import (
    DiracMixtureSinglePublisherModel,
)
from wfa_planning_evaluation_framework.models.dirac_mixture_multi_publisher_model import (
    MultivariateMixedPoissonOptimizer,
    DiracMixtureMultiPublisherModel,
)


class FakeRandomGenerator:
    def choice(self, a: List, p: np.ndarray, size: int) -> np.ndarray:
        """Fake method to replace np.random.choice."""
        if len(a) == 2:
            n = int(size * p[0])
            return np.array([a[0]] * n + [a[1]] * (size - n))
        return np.array([len(a) - 1] * size)

    def random(self, size: Tuple) -> np.ndarray:
        """Fake method to replace np.random.random."""
        return np.zeros(shape=size)


class MultivariateMixedPoissonOptimizerTest(absltest.TestCase):
    def test_normalize_rows(self):
        matrix = np.array([[1, 2, 1], [2, 3, 5]])
        res = MultivariateMixedPoissonOptimizer.normalize_rows(matrix)
        expected = np.array([[0.25, 0.5, 0.25], [0.2, 0.3, 0.5]])
        np.testing.assert_almost_equal(res, expected, decimal=2)
        # For 2d arrays, absltest.assertSequenceAlmostEqual no longer works.

    def test_find_single_pub_pmfs(self):
        inst = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array(
                [[1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]
            ),
            frequency_histograms_on_observable_directions=np.array(
                [[2, 2], [1, 3], [1, 4], [2, 3], [2, 4]]
            ),
        )
        res = inst.find_single_pub_pmfs()
        self.assertLen(res, 3)
        np.testing.assert_almost_equal(res[0], [0.5, 0.5], decimal=3)
        np.testing.assert_almost_equal(res[1], [0.2, 0.8], decimal=3)
        np.testing.assert_almost_equal(res[2], [0.333, 0.667], decimal=3)

    def test_in_bound_weighted_sampling(self):
        inst = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array(
                [[1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]
            ),
            frequency_histograms_on_observable_directions=np.array(
                [[2, 2], [1, 3], [1, 4], [2, 3], [2, 4]]
            ),
            rng=FakeRandomGenerator(),
            ncomponents=1,
            dilusion=0,
        )
        res = inst.in_bound_weighted_sampling()
        expected = np.array([[0, 0, 0], [1, 1, 1]])
        np.testing.assert_almost_equal(res, expected, decimal=3)

    def test_in_bound_weighted_sampling_with_more_components(self):
        inst = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array(
                [[1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]
            ),
            frequency_histograms_on_observable_directions=np.array(
                [[0, 3], [1, 3], [0, 3], [2, 3], [0, 3]]
            ),
            rng=FakeRandomGenerator(),
            ncomponents=5,
            dilusion=0,
        )
        res = inst.in_bound_weighted_sampling()
        expected = np.array(
            [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        )
        np.testing.assert_almost_equal(res, expected, decimal=3)
        inst.dilusion = 1
        res = inst.in_bound_weighted_sampling()
        expected = np.array(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        )
        np.testing.assert_almost_equal(res, expected, decimal=3)

    def test_obtain_pmf_matrix_with_zero_component(self):
        res = MultivariateMixedPoissonOptimizer.obtain_pmf_matrix(
            component_vector=np.array([0, 0]),
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            max_freq=1,
        )
        expected = np.array([[1, 0], [1, 0], [1, 0]])
        np.testing.assert_almost_equal(res, expected, decimal=2)

    def test_obtain_pmf_matrix_with_non_zero_component(self):
        res = MultivariateMixedPoissonOptimizer.obtain_pmf_matrix(
            component_vector=np.array([1, 2]),
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            max_freq=1,
        )
        expected = np.array([[0.368, 0.632], [0.135, 0.865], [0.050, 0.950]])
        np.testing.assert_almost_equal(res, expected, decimal=2)

    def test_fit_with_zero_reach(self):
        """When given reach = zero, the fit should have weight = 1 on the zero component."""
        optimizer = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            frequency_histograms_on_observable_directions=np.array(
                [[1, 0], [1, 0], [1, 0]]
            ),
            ncomponents=1,
            dilusion=1,
            rng=FakeRandomGenerator(),
        )
        optimizer.fit()
        # weight should be concentrated at the zero component.
        expected_ws = np.array([1, 0])
        np.testing.assert_almost_equal(optimizer.ws, expected_ws, decimal=2)

    def test_fit_with_zero_reach(self):
        """A test case where we can manually compute and confirm the fit."""
        optimizer = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            frequency_histograms_on_observable_directions=np.array(
                [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7]]
            ),
            ncomponents=1,
            rng=FakeRandomGenerator(),
        )
        optimizer.fit()
        expected_components = np.array([[0, 0], [1, 1]])
        expected_ws = np.array([0.322, 0.678])
        np.testing.assert_almost_equal(
            optimizer.components,
            expected_components,
            decimal=2,
        )
        np.testing.assert_almost_equal(
            optimizer.ws,
            expected_ws,
            decimal=2,
        )

    def test_predict_with_zero_impression(self):
        optimizer = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            frequency_histograms_on_observable_directions=np.array(
                [[0.7, 0.3, 0], [0.6, 0.4, 0], [0.3, 0.7, 0]]
            ),
            ncomponents=1,
            rng=FakeRandomGenerator(),
        )

        # Set arbitrary parameters
        optimizer.fitted = True
        optimizer.components = np.array([[0, 0], [1, 0], [1, 1]])
        optimizer.ws = np.array([0.3, 0.2, 0.5])

        # Zero impressions
        res = optimizer.predict(hypothetical_direction=np.array([0, 0]))
        expected = np.array([1, 0, 0])
        np.testing.assert_almost_equal(
            res,
            expected,
            decimal=2,
            err_msg="Unexpected for zero impression, no customized_max_freq",
        )
        res = optimizer.predict(
            hypothetical_direction=np.array([0, 0]), customized_max_freq=1
        )
        expected = np.array([1, 0])
        np.testing.assert_almost_equal(
            res,
            expected,
            decimal=2,
            err_msg="Unexpected for zero impression, smaller customized_max_freq",
        )

        res = optimizer.predict(
            hypothetical_direction=np.array([0, 0]), customized_max_freq=3
        )
        expected = np.array([1, 0, 0, 0])
        np.testing.assert_almost_equal(
            res,
            expected,
            decimal=2,
            err_msg="Unexpected for zero impression, larger customized_max_freq",
        )

    def test_predict_with_infinite_impression(self):
        optimizer = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            frequency_histograms_on_observable_directions=np.array(
                [[0.7, 0.3, 0], [0.6, 0.4, 0], [0.3, 0.7, 0]]
            ),
            ncomponents=1,
            rng=FakeRandomGenerator(),
        )
        optimizer.fitted = True
        optimizer.components = np.array([[0, 0], [1, 0], [1, 1]])
        optimizer.ws = np.array([0.3, 0.2, 0.5])
        # Very large impressions
        res = optimizer.predict(hypothetical_direction=np.array([10000, 10000]))
        # The predicted freq hist under very large impressions should only have mass
        # on the zero frequency and max frequency.
        expected = np.array([0.3, 0, 0.7])
        np.testing.assert_almost_equal(
            res, expected, decimal=2, err_msg="Unexpected for infinite impressions"
        )

    def test_predict_with_practical_impression(self):
        optimizer = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            frequency_histograms_on_observable_directions=np.array(
                [[0.7, 0.3, 0], [0.6, 0.4, 0], [0.3, 0.7, 0]]
            ),
            ncomponents=1,
            rng=FakeRandomGenerator(),
        )
        optimizer.fitted = True
        optimizer.components = np.array([[0, 0], [1, 0], [1, 1]])
        optimizer.ws = np.array([0.3, 0.2, 0.5])
        # Practical impressions
        res = optimizer.predict(hypothetical_direction=np.array([0.5, 0.5]))
        expected = np.array([0.605, 0.245, 0.150])
        np.testing.assert_almost_equal(
            res, expected, decimal=2, err_msg="Unexpected for moderate impressions"
        )


class DiracMixtureMultiPublisherModelTest(parameterized.TestCase):
    def test_set_common_universe_size(self):
        rp1 = ReachPoint(impressions=[3, 0, 0], kplus_reaches=[1])
        rp2 = ReachPoint(impressions=[5, 0, 0], kplus_reaches=[2], universe_size=10)
        rp3 = ReachPoint(impressions=[2, 4, 0], kplus_reaches=[3], universe_size=20)
        model = DiracMixtureMultiPublisherModel(reach_points=[rp1, rp2, rp3])
        self.assertEqual(model.common_universe_size, 20)
        for rp in model._data:
            self.assertEqual(rp.universe_size, 20)

    def test_ensure_compatible_num_publishers(self):
        imps = [3, 4, 5]
        reaches = [1, 2, 3]
        reach_curves = [
            DiracMixtureSinglePublisherModel(
                data=[
                    ReachPoint(
                        impressions=[imp], kplus_reaches=[reach], universe_size=15
                    )
                ]
            )
            for imp, reach in zip(imps, reaches)
        ]
        reach_points = [ReachPoint(impressions=imps, kplus_reaches=[4])]
        try:
            DiracMixtureMultiPublisherModel.ensure_compatible_num_publishers(
                reach_curves, reach_points
            )
        except:
            self.fail("Rejected compatible inputs")
        reach_points += [ReachPoint(impressions=[1, 2, 3, 4], kplus_reaches=[4])]
        with self.assertRaises(ValueError):
            DiracMixtureMultiPublisherModel.ensure_compatible_num_publishers(
                reach_curves, reach_points
            )

    def test_obtain_observable_directions(self):
        rp1 = ReachPoint(impressions=[3, 0, 0], kplus_reaches=[1])
        rp2 = ReachPoint(impressions=[5, 0, 0], kplus_reaches=[2])
        rp3 = ReachPoint(impressions=[2, 4, 0], kplus_reaches=[3])
        rp4 = ReachPoint(impressions=[0, 0, 7], kplus_reaches=[4])
        rp5 = ReachPoint(impressions=[0, 8, 0], kplus_reaches=[5])
        rp6 = ReachPoint(impressions=[4, 6, 10], kplus_reaches=[10])
        (
            baseline_imps,
            directions,
        ) = DiracMixtureMultiPublisherModel.obtain_observable_directions(
            reach_points=[rp1, rp2, rp3, rp4, rp5, rp6],
        )
        np.testing.assert_almost_equal(baseline_imps, [5, 8, 10], decimal=2)
        expected = np.array(
            [
                [3 / 5, 0, 0],
                [1, 0, 0],
                [2 / 5, 4 / 8, 0],
                [0, 0, 7 / 10],
                [0, 1, 0],
                [4 / 5, 6 / 8, 1],
            ]
        )
        np.testing.assert_almost_equal(directions, expected, decimal=2)

    @parameterized.named_parameters(
        {
            "testcase_name": "zero_reach",
            "reaches": [0, 0, 0],
            "expected_weights": [1, 0],
        },
        {
            "testcase_name": "full_reach",
            "reaches": [15, 15, 15],
            "expected_weights": [0, 1],
        },
        {
            "testcase_name": "practical_reach",
            "reaches": [7, 8, 9],
            "expected_weights": [0.156, 0.844],
        },
    )
    def test_fit(self, reaches, expected_weights):
        rc1 = DiracMixtureSinglePublisherModel(
            data=[ReachPoint(impressions=[4], kplus_reaches=[2], universe_size=15)]
        )
        rc2 = DiracMixtureSinglePublisherModel(
            data=[ReachPoint(impressions=[6], kplus_reaches=[4], universe_size=15)]
        )
        rp1 = ReachPoint(
            impressions=[20, 0], kplus_reaches=[reaches[0]], universe_size=15
        )
        rp2 = ReachPoint(
            impressions=[0, 20], kplus_reaches=[reaches[1]], universe_size=15
        )
        rp3 = ReachPoint(
            impressions=[10, 10], kplus_reaches=[reaches[2]], universe_size=15
        )
        model = DiracMixtureMultiPublisherModel(
            reach_curves=[rc1, rc2],
            reach_points=[rp1, rp2, rp3],
            ncomponents=1,
            dilusion=1,
            rng=FakeRandomGenerator(),
        )
        model._fit()
        expected_components = np.array([[0, 0], [1, 1]])
        np.testing.assert_equal(model.optimizer.components, expected_components)
        np.testing.assert_almost_equal(model.optimizer.ws, expected_weights, decimal=2)

    @parameterized.named_parameters(
        {
            "testcase_name": "all_weight_on_zero",
            "mixture_weights": [1, 0],
            "impressions": [3, 5],
            "expected_kplus_reaches": [0, 0],
        },
        {
            "testcase_name": "zero_impression",
            "mixture_weights": [0.4, 0.6],
            "impressions": [0, 0],
            "expected_kplus_reaches": [0, 0],
        },
        {
            "testcase_name": "infinite_impression",
            "mixture_weights": [0.4, 0.6],
            "impressions": [1000, 1000],
            # 0.6 times a universe of 10 equals 6.  These 6 people should all be
            # reached under infinite impressions.
            "expected_kplus_reaches": [6, 6],
        },
        {
            "testcase_name": "practical_impression",
            "mixture_weights": [0.4, 0.6],
            "impressions": [3, 5],
            "expected_kplus_reaches": [3, 1],
        },
    )
    def test_by_impressions_no_single_pub_reach_agreement(
        self, mixture_weights, impressions, expected_kplus_reaches
    ):
        rp1 = ReachPoint(impressions=[4, 0], kplus_reaches=[0], universe_size=10)
        rp2 = ReachPoint(impressions=[0, 6], kplus_reaches=[0], universe_size=10)
        rp3 = ReachPoint(impressions=[4, 6], kplus_reaches=[0], universe_size=10)
        # The above parameters don't matter since we'll manually set its optimizer
        # parameters below.
        model = DiracMixtureMultiPublisherModel(
            reach_points=[rp1, rp2, rp3],
            ncomponents=1,
            dilusion=1,
            rng=FakeRandomGenerator(),
        )
        model._fit()
        model.optimizer.components = np.array([[0, 0], [0.5, 0.3]])

        model.optimizer.ws = np.array(mixture_weights)
        res = model.by_impressions_no_single_pub_reach_agreement(
            impressions=impressions, max_frequency=2
        )
        self.assertEqual(res.reach(1), expected_kplus_reaches[0])
        self.assertEqual(res.reach(2), expected_kplus_reaches[1])

    def test_backsolve_impression(self):
        curve = lambda x: x ** 2
        target_reach = 26
        expected = 5
        starting_impression = 9
        res = DiracMixtureMultiPublisherModel.backsolve_impression(
            curve, target_reach, starting_impression
        )
        self.assertEqual(res, expected)
        starting_impression = 2
        res = DiracMixtureMultiPublisherModel.backsolve_impression(
            curve, target_reach, starting_impression
        )
        self.assertEqual(res, expected)

    def test_by_impressions_with_single_pub_reach_agreement(self):
        rc1 = DiracMixtureSinglePublisherModel(
            data=[ReachPoint(impressions=[20], kplus_reaches=[11], universe_size=30)]
        )
        rc2 = DiracMixtureSinglePublisherModel(
            data=[ReachPoint(impressions=[20], kplus_reaches=[7], universe_size=30)]
        )
        rp1 = ReachPoint(impressions=[20, 0], kplus_reaches=[6], universe_size=30)
        rp2 = ReachPoint(impressions=[0, 20], kplus_reaches=[12], universe_size=30)
        rp3 = ReachPoint(impressions=[20, 20], kplus_reaches=[13], universe_size=30)
        # The above parameters don't matter since we'll manually set its optimizer
        # parameters below.
        model = DiracMixtureMultiPublisherModel(
            reach_curves=[rc1, rc2],
            reach_points=[rp1, rp2, rp3],
            ncomponents=5,
            dilusion=1,
            rng=FakeRandomGenerator(),
        )
        model._fit()
        res = model.by_impressions_with_single_pub_reach_agreement(
            impressions=[20, 0], max_frequency=2
        )
        self.assertEqual(res.reach(1), 11)
        # As a comparison, look at the result when not forcing single
        # pub agreement
        res = model.by_impressions_no_single_pub_reach_agreement(
            impressions=[20, 0], max_frequency=2
        )
        self.assertEqual(res.reach(1), 9)
        res = model.by_impressions_with_single_pub_reach_agreement(
            impressions=[0, 20], max_frequency=2
        )
        self.assertEqual(res.reach(1), 7)


if __name__ == "__main__":
    absltest.main()
