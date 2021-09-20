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
"""Tests for restricted_pairwise_union_reach_surface.py."""

import warnings
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import DEFAULT
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.models.restricted_pairwise_union_reach_surface import (
    RestrictedPairwiseUnionReachSurface,
)
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve


class LinearCappedReachCurve(ReachCurve):
    """A curve of simple form that facilitates testing."""

    def _fit(self) -> None:
        self._max_reach = self._data[0].reach()

    def by_impressions(self, impressions: [int], max_frequency: int = 1) -> ReachPoint:
        """Returns the estimated reach for a given impression vector."""
        kplus_frequencies = [
            min(sum(impressions), self.max_reach) // i
            for i in range(1, max_frequency + 1)
        ]
        return ReachPoint(impressions, kplus_frequencies)


class RestrictedPairwiseUnionReachSurfaceTest(parameterized.TestCase):
    def assertPointsAlmostEqualToPrediction(
        self, surface, reach_points, tolerance=0.05, msg=""
    ):
        for reach_point in reach_points:
            prediction = (
                surface.by_spend(reach_point.spends)
                if (reach_point.spends)
                else surface.by_impressions(reach_point.impressions)
            )
            self.assertAlmostEqual(
                prediction.reach(),
                reach_point.reach(),
                delta=reach_point.reach() * tolerance,
                msg=msg,
            )

    def generate_true_reach(self, a, reach_curves, impressions, spends=None):
        p = len(reach_curves)
        reach_vector = [
            reach_curve.by_impressions([impression]).reach()
            for reach_curve, impression in zip(reach_curves, impressions)
        ]
        reach = sum(reach_vector)
        for i in range(len(reach_curves)):
            for j in range(len(reach_curves)):
                reach -= (a[i * p + j] * reach_vector[i] * reach_vector[j]) / (
                    max(reach_curves[i].max_reach, reach_curves[j].max_reach) * 2
                )
        return ReachPoint(impressions, [reach], spends)

    def generate_sample_reach_curves(self, num_publishers, decay_rate, universe_size):
        max_reaches = [
            universe_size * (decay_rate ** pub_num) for pub_num in range(num_publishers)
        ]
        reach_curves = []
        for max_reach in max_reaches:
            curve = LinearCappedReachCurve([ReachPoint([max_reach], (max_reach,))])
            curve._fit()
            reach_curves.append(curve)
        return reach_curves

    def generate_sample_matrix_a(self, num_publishers, lbd_const=0):
        lbd = [lbd_const] * num_publishers
        true_a = np.ones(num_publishers * num_publishers)
        for i in range(num_publishers):
            for j in range(num_publishers):
                true_a[i * num_publishers + j] = lbd[i] * lbd[j] if i != j else 0
        return true_a

    def generate_sample_reach_points(
        self, true_a, reach_curves, size, universe_size, random_seed
    ):
        reach_points = []
        random_generator = np.random.default_rng(random_seed)
        for _ in range(size):
            impressions = [
                random_generator.uniform(0, universe_size / 2)
                for _ in range(len(reach_curves))
            ]
            reach_points.append(
                self.generate_true_reach(true_a, reach_curves, impressions)
            )
        return reach_points

    def test_construct_D_from_r_and_m(self):
        r = [1, 2, 3]
        m = [3, 2, 1]
        D = RestrictedPairwiseUnionReachSurface._construct_D_from_r_and_m(r, m)
        expected_D = np.array(
            [[0.16667, 0.33333, 0.5], [0.33333, 1, 1.5], [0.5, 1.5, 4.5]]
        )
        np.testing.assert_array_almost_equal(D, expected_D, decimal=3)

    def test_check_lbd_feasiblity(self):
        lbd1 = [0.5, 0.5, 0.5]
        lbd2 = [1, 1, 1]
        feasiblity = RestrictedPairwiseUnionReachSurface._check_lbd_feasiblity
        self.assertTrue(feasiblity(lbd1))
        self.assertFalse(feasiblity(lbd2))

    def test_compute_u(self):
        lbd = [0.2, 0.3, 0.4]
        i = 0
        D = np.ones((3, 3))
        u = RestrictedPairwiseUnionReachSurface._compute_u(lbd, i, D)
        self.assertAlmostEqual(u, 0.12, delta=1e-4)

    def test_compute_v(self):
        lbd = [0.2, 0.3, 0.4]
        i = 0
        D = np.ones((3, 3))
        v = RestrictedPairwiseUnionReachSurface._compute_v(lbd, i, D)
        self.assertAlmostEqual(v, 0.7, delta=1e-4)

    def test_get_feasible_bound(self):
        lbd = [0.2, 0.5, 0.5]
        i = 0
        B = RestrictedPairwiseUnionReachSurface._get_feasible_bound(lbd, i)
        self.assertAlmostEqual(B, 1, delta=1e-4)
        feasiblity = RestrictedPairwiseUnionReachSurface._check_lbd_feasiblity
        lbd[0] = B
        self.assertTrue(feasiblity(lbd))
        lbd[0] = B + 0.1
        self.assertTrue(not feasiblity(lbd))

    @parameterized.parameters(
        [RestrictedPairwiseUnionReachSurface._truncated_uniform_initial_lbd],
        [RestrictedPairwiseUnionReachSurface._scaled_from_simplex_initial_lbd],
    )
    def test_init_lbd_sampler(self, init_lbd_sampler):
        sample = [init_lbd_sampler(10) for _ in range(30)]
        feasiblity = RestrictedPairwiseUnionReachSurface._check_lbd_feasiblity
        self.assertTrue(all([feasiblity(s) for s in sample]))

    def test_by_impressions(self):
        num_publishers = 5
        training_size = 10
        universe_size = 20000
        decay_rate = 1

        reach_curves = self.generate_sample_reach_curves(
            num_publishers, decay_rate, universe_size
        )
        true_a = self.generate_sample_matrix_a(num_publishers, 2 / num_publishers)
        training_reach_points = self.generate_sample_reach_points(
            true_a, reach_curves, training_size, universe_size, 1
        )

        surface = RestrictedPairwiseUnionReachSurface(
            reach_curves, training_reach_points
        )
        surface._fit()
        test_reach_points = self.generate_sample_reach_points(
            true_a, reach_curves, training_size, universe_size, 2
        )
        self.assertPointsAlmostEqualToPrediction(
            surface, training_reach_points, msg="High discrepancy on training points"
        )
        self.assertPointsAlmostEqualToPrediction(
            surface, test_reach_points, msg="High discrepancy on testing points"
        )

    def test_by_impressions_zero_lambda(self):
        num_publishers = 5
        training_size = 10
        universe_size = 20000
        decay_rate = 1

        reach_curves = self.generate_sample_reach_curves(
            num_publishers, decay_rate, universe_size
        )
        true_a = self.generate_sample_matrix_a(num_publishers)
        training_reach_points = self.generate_sample_reach_points(
            true_a, reach_curves, training_size, universe_size, 1
        )

        surface = RestrictedPairwiseUnionReachSurface(
            reach_curves, training_reach_points
        )
        surface._fit()
        test_reach_points = self.generate_sample_reach_points(
            true_a, reach_curves, training_size, universe_size, 2
        )
        self.assertPointsAlmostEqualToPrediction(
            surface, training_reach_points, tolerance=0.00001
        )
        self.assertPointsAlmostEqualToPrediction(
            surface, test_reach_points, tolerance=0.00001
        )

        
if __name__ == "__main__":
    absltest.main()
