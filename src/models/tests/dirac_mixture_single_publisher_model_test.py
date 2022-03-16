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
"""Tests for dirac_mixture_single_publisher_model.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.dirac_mixture_single_publisher_model import (
    UnivariateMixedPoissonOptimizer,
    DiracMixtureSinglePublisherModel,
)


class UnivariateMixedPoissonOptimizerTest(absltest.TestCase):
    cls = UnivariateMixedPoissonOptimizer

    def test_validate_frequency_histogram(self):
        try:
            self.cls.validate_frequency_histogram(np.array([0, 1, 1]))
        except:
            self.fail("Reject a valid histogram")
        with self.assertRaises(ValueError):
            self.cls.validate_frequency_histogram(np.array([0, 1, -1]))
        with self.assertRaises(ValueError):
            self.cls.validate_frequency_histogram(np.array([0, 0, 0]))

    def test_in_bound_purely_weighted_grid(self):
        pmf = np.array([0.3, 0, 0.4, 0.2, 0.1])
        ncomponents = 10
        res = self.cls.in_bound_purely_weighted_grid(ncomponents, pmf)
        expected = np.array([0, 0.333, 0.667, 2, 2.25, 2.5, 2.75, 3, 3.5, 4])
        self.assertSequenceAlmostEqual(res, expected, places=2)

    def test_in_bound_uniform_grid(self):
        pmf = np.array([0.3, 0, 0.4, 0.2, 0.1])
        ncomponents = 10
        res = self.cls.in_bound_uniform_grid(ncomponents, pmf)
        expected = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        self.assertSequenceAlmostEqual(res, expected, places=2)

    def test_in_bound_grid(self):
        pmf = np.array([0.3, 0, 0.4, 0.2, 0.1])
        ncomponents = 10
        res = self.cls.in_bound_grid(ncomponents, pmf, 0.5)
        expected = np.array([0, 0.5, 1, 2, 2.333, 2.667, 3, 3.5, 4, 4.5])
        self.assertSequenceAlmostEqual(res, expected, places=2)

    def test_truncated_poisson_pmf_vec(self):
        res = self.cls.truncated_poisson_pmf_vec(poisson_mean=1, max_freq=3)
        expected = np.array([0.368, 0.368, 0.184, 0.0800])
        self.assertSequenceAlmostEqual(res, expected, places=2)

    def test_cross_entropy(self):
        observed = np.array([[2, 3], [1, 4]])
        fitted = np.array([[1, 3], [2, 4]])
        res = self.cls.cross_entropy(observed, fitted).value
        expected = -(2 * np.log(1) + 3 * np.log(3) + 1 * np.log(2) + 4 * np.log(4))
        self.assertAlmostEqual(res, expected, places=2, msg="Unexpected for 2d array")
        # Further test the case with zero
        observed = np.array([[2, 3, 1, 0]])
        fitted = np.array([[1, 3, 2, 0]])
        res = self.cls.cross_entropy(observed, fitted).value
        expected = -(2 * np.log(1) + 3 * np.log(3) + 1 * np.log(2) + 0)
        self.assertAlmostEqual(res, expected, places=2, msg="Unexpected for 1d array")

    def test_relative_entropy(self):
        observed = np.array([[2, 3], [1, 4]])
        fitted = np.array([[1, 3], [2, 4]])
        res = self.cls.relative_entropy(observed, fitted).value
        expected = (
            2 * np.log(2 / 1)
            + 3 * np.log(3 / 3)
            + 1 * np.log(1 / 2)
            + 4 * np.log(4 / 4)
        )
        self.assertAlmostEqual(res, expected, places=2, msg="Unexpected for 2d array")
        # Further test the case with zero
        observed = np.array([[2, 3, 1, 0]])
        fitted = np.array([[1, 3, 2, 0]])
        res = self.cls.relative_entropy(observed, fitted).value
        expected = 2 * np.log(2 / 1) + 3 * np.log(3 / 3) + 1 * np.log(1 / 2)
        self.assertAlmostEqual(res, expected, places=2, msg="Unexpected for 1d array")

    def test_solve_optimal_weights(self):
        observed = np.array([[1.5, 1.5], [1.5, 1.5]])
        component_a = np.array([[1, 2], [2, 1]])
        component_b = np.array([[2, 1], [1, 2]])
        res = self.cls.solve_optimal_weights(
            observed, [component_a, component_b], self.cls.cross_entropy
        )
        expected = np.array([0.5, 0.5])
        self.assertSequenceAlmostEqual(res, expected, places=2)

    def test_fit_zero_reach(self):
        optimizer = self.cls(frequency_histogram=np.array([5, 0, 0, 0]), ncomponents=3)
        optimizer.fit()
        expected_ws = np.array([1, 0, 0])
        expected_components = np.array([0, 0.333, 0.667])
        self.assertSequenceAlmostEqual(
            optimizer.ws,
            expected_ws,
            places=2,
            msg="Unexpected weights for zero reach",
        )
        self.assertSequenceAlmostEqual(
            optimizer.components,
            expected_components,
            places=2,
            msg="Unexpected_components for zero reach",
        )

    def test_fit_non_zero_reach(self):
        optimizer = self.cls(frequency_histogram=np.array([3, 2, 2, 1]), ncomponents=4)
        optimizer.fit()
        expected_ws = np.array([0.133, 0, 0.535, 0.332])
        expected_components = np.array([0, 0.5, 1, 2])
        self.assertSequenceAlmostEqual(
            optimizer.ws,
            expected_ws,
            places=2,
            msg="Unexpected weights for non-zero reach",
        )
        self.assertSequenceAlmostEqual(
            optimizer.components,
            expected_components,
            places=2,
            msg="Unexpected_components for non-zero reach",
        )

    def test_predict_zero_scaling_factor(self):
        optimizer = self.cls(frequency_histogram=np.array([3, 2, 1]), ncomponents=2)
        # Set arbitray parameters for testing
        optimizer.fitted = True
        optimizer.components = np.array([0, 1])
        optimizer.ws = np.array([0.3, 0.7])
        res = optimizer.predict(scaling_factor=0)
        expected = np.array([1, 0, 0])
        self.assertSequenceAlmostEqual(
            res,
            expected,
            places=2,
            msg="Unexpected when customized_max_freq is not specified",
        )
        ## Test customized_max_freq
        res = optimizer.predict(scaling_factor=0, customized_max_freq=1)
        expected = np.array([1, 0])
        self.assertSequenceAlmostEqual(
            res,
            expected,
            places=2,
            msg="Unexpected for smaller customized_max_freq",
        )
        res = optimizer.predict(scaling_factor=0, customized_max_freq=6)
        expected = np.array([1, 0, 0, 0, 0, 0, 0])
        self.assertSequenceAlmostEqual(
            res,
            expected,
            places=2,
            msg="Unexpected for larger customized_max_freq",
        )

    def test_predict_infinite_scaling_factor(self):
        optimizer = self.cls(frequency_histogram=np.array([3, 2, 1]), ncomponents=2)
        optimizer.fitted = True
        optimizer.components = np.array([0, 1])
        optimizer.ws = np.array([0.3, 0.7])
        res = optimizer.predict(scaling_factor=10000)
        expected = np.array([0.3, 0, 0.7])
        self.assertSequenceAlmostEqual(res, expected, places=2)

    def test_predict_practical_scaling_factor(self):
        optimizer = self.cls(frequency_histogram=np.array([3, 2, 1]), ncomponents=2)
        optimizer.fitted = True
        optimizer.components = np.array([0, 1])
        optimizer.ws = np.array([0.3, 0.7])
        res = optimizer.predict(scaling_factor=0.5)
        expected = np.array([0.725, 0.212, 0.063])
        self.assertSequenceAlmostEqual(res, expected, places=2)


class DiracMixtureSinglePublisherModelTest(parameterized.TestCase):
    cls = DiracMixtureSinglePublisherModel

    def test_debiased_clip(self):
        res = self.cls.debiased_clip(noised_histogram=np.array([20, 6, -3, -2]))
        expected = np.array([20, 1, 0, 0])
        self.assertSequenceAlmostEqual(res, expected, places=2)

    def test_obtain_zero_included_histogram(self):
        rp = ReachPoint(impressions=[10], kplus_reaches=[5, 3, 2])
        res = self.cls.obtain_zero_included_histogram(universe_size=8, rp=rp)
        expected = np.array([3, 2, 1, 2])
        self.assertSequenceAlmostEqual(res, expected)

    def test_fit_with_zero_reach(self):
        rp = ReachPoint(impressions=[10], kplus_reaches=[0])
        model = self.cls(data=[rp], ncomponents=2)
        model._fit()
        expected = np.array([1, 0])
        self.assertSequenceAlmostEqual(model.optimizer.ws, expected, places=2)

    def test_fit_with_non_zero_reach(self):
        rp = ReachPoint(impressions=[10], kplus_reaches=[3], spends=[1.0])
        model = self.cls(data=[rp], ncomponents=2)
        model._fit()
        expected = np.array([0.614, 0.386])
        self.assertSequenceAlmostEqual(model.optimizer.ws, expected, places=2)

    @parameterized.named_parameters(
        {"testcase_name": "zero_impression", "impressions": 0, "expected": 0},
        {"testcase_name": "sample_impressions", "impressions": 10, "expected": 3},
        {"testcase_name": "double_impressions", "impressions": 20, "expected": 4},
    )
    def test_by_impressions(self, impressions: int, expected: int):
        rp = ReachPoint(impressions=[10], kplus_reaches=[3])
        model = self.cls(data=[rp], ncomponents=5)
        res = model.by_impressions(impressions=[impressions])
        self.assertEqual(
            res._kplus_reaches[0],
            expected,
        )

    def test_by_impressions_with_infinite_impressions(self):
        rp = ReachPoint(impressions=[10], kplus_reaches=[3])
        model = self.cls(data=[rp], ncomponents=5)
        res = model.by_impressions(impressions=[10000])
        expected = round(model.N * (1 - model.optimizer.ws[0]), 0)
        self.assertEqual(
            res._kplus_reaches[0],
            expected,
        )

    @parameterized.named_parameters(
        {"testcase_name": "zero_spend", "spend": 0.0, "expected": 0},
        {"testcase_name": "sample_spend", "spend": 1.0, "expected": 3},
        {"testcase_name": "double_spend", "spend": 2.0, "expected": 4},
    )
    def test_by_spend(self, spend: float, expected: int):
        rp = ReachPoint(impressions=[10], kplus_reaches=[3], spends=[1.0])
        model = self.cls(data=[rp], ncomponents=5)
        res = model.by_spend(spends=[spend])
        self.assertEqual(
            res._kplus_reaches[0],
            expected,
        )

    def test_by_spend_with_infinite_spend(self):
        rp = ReachPoint(impressions=[10], kplus_reaches=[3], spends=[1.0])
        model = self.cls(data=[rp], ncomponents=5)
        res = model.by_spend(spends=[10000.0])
        expected = round(model.N * (1 - model.optimizer.ws[0]), 0)
        self.assertEqual(
            res._kplus_reaches[0],
            expected,
        )


if __name__ == "__main__":
    absltest.main()
