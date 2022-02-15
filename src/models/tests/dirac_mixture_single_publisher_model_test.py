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
"""Tests for dirac_mixture_single_publisher_model.py."""

from absl.testing import absltest
import numpy as np
from wfa_planning_evaluation_framework.models.dirac_mixture_single_publisher_model import (
    UnivariateMixedPoissonOptimizer,
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

    def test_weighted_grid_sampling(self):
        pmf = np.array([0.2, 0.3, 0.5])
        ncomponents = 8
        res = self.cls.weighted_grid_sampling(ncomponents, pmf)
        expected = np.array([0, 1, 1.5, 2, 2.25, 2.5, 2.75, 3])
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

    def test_solve_optimal_weights(self):
        observed = np.array([[1.5, 1.5], [1.5, 1.5]])
        component_a = np.array([[1, 2], [2, 1]])
        component_b = np.array([[2, 1], [1, 2]])
        res = self.cls.solve_optimal_weights(
            observed, [component_a, component_b], self.cls.cross_entropy
        )
        expected = np.array([0.5, 0.5])
        self.assertSequenceAlmostEqual(res, expected, places=2)

    def test_fit(self):
        # zero reach
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

        # non-zero reach
        optimizer = self.cls(frequency_histogram=np.array([3, 2, 2, 1]), ncomponents=4)
        optimizer.fit()
        expected_ws = np.array([0.133, 0.535, 0.332, 0])
        expected_components = np.array([0, 1, 2, 4])
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

    def test_predict(self):
        optimizer = self.cls(frequency_histogram=np.array([3, 2, 1]), ncomponents=2)
        # Set arbitray parameters for testing
        optimizer.fitted = True
        optimizer.components = np.array([0, 1])
        optimizer.ws = np.array([0.3, 0.7])

        # Zero scaling_factor
        res = optimizer.predict(scaling_factor=0)
        expected = np.array([1, 0, 0])
        self.assertSequenceAlmostEqual(
            res,
            expected,
            places=2,
            msg="Unexpected for zero scaling, no customized_max_freq",
        )
        ## Test customized_max_freq
        res = optimizer.predict(scaling_factor=0, customized_max_freq=1)
        expected = np.array([1, 0])
        self.assertSequenceAlmostEqual(
            res,
            expected,
            places=2,
            msg="Unexpected for zero scaling, smaller customized_max_freq",
        )
        res = optimizer.predict(scaling_factor=0, customized_max_freq=6)
        expected = np.array([1, 0, 0, 0, 0, 0, 0])
        self.assertSequenceAlmostEqual(
            res,
            expected,
            places=2,
            msg="Unexpected for zero scaling, larger customized_max_freq",
        )

        # Moderate scaling factor
        res = optimizer.predict(scaling_factor=0.5)
        expected = np.array([0.725, 0.212, 0.063])
        self.assertSequenceAlmostEqual(
            res, expected, places=2, msg="Unexpected for moderate scaling"
        )

        # Infinite scaling_factor
        res = optimizer.predict(scaling_factor=10000)
        expected = np.array([0.3, 0, 0.7])
        self.assertSequenceAlmostEqual(
            res, expected, places=2, msg="Unexpected for infinite scaling"
        )


if __name__ == "__main__":
    absltest.main()
