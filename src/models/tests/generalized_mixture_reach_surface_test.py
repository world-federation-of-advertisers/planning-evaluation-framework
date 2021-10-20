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
"""Tests for generalized_mixture_reach_surface.py."""

import warnings
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import DEFAULT
from absl.testing import absltest
import numpy as np
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.models.generalized_mixture_reach_surface import (
    GeneralizedMixtureReachSurface,
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


class GeneralizedMixtureReachSurfaceTest(absltest.TestCase):
    def assertPointsAlmostEqualToPrediction(
        self, surface, reach_points, tolerance=0.03
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
            )

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

    # def generate_sample_reach_curves(self, num_publishers, decay_rate, universe_size):
    #     max_reaches = [
    #         universe_size * (decay_rate ** pub_num) for pub_num in range(num_publishers)
    #     ]
    #     return [
    #         LinearCappedReachCurve([ReachPoint([max_reach], (max_reach,))])
    #         for max_reach in max_reaches
    #     ]

    def generate_sample_reach_points(
        self,
        num_clusters,
        num_publishers,
        true_a,
        reach_curves,
        size,
        universe_size,
        N,
        random_seed,
    ):
        reach_points = []
        random_generator = np.random.default_rng(random_seed)
        for _ in range(size):
            impressions = [
                random_generator.uniform(0, N / 2) for _ in range(len(reach_curves))
            ]
            reach_points.append(
                self.generate_true_reach(
                    num_clusters, num_publishers, true_a, reach_curves, impressions, N
                )
            )
        return reach_points

    def generate_true_reach(
        self, num_clusters, num_publishers, a, reach_curves, impressions, N, spends=None
    ):
        reach_vector = [
            reach_curve.by_impressions([impression]).reach()
            for reach_curve, impression in zip(reach_curves, impressions)
        ]
        reach = 0
        for j in range(num_clusters):
            w = 1
            for i in range(num_publishers):
                w *= 1 - ((a[j + i * num_clusters] * reach_vector[i]) / N)
            reach += 1 - w
        reach *= N
        return ReachPoint(impressions, [reach], spends)

    def test_by_impressions(self):
        universe_size = 200
        num_publishers = 20
        num_clusters = 10
        training_size = 200
        decay_rate = 0.8
        random_seed = 1

        random_generator = np.random.default_rng(random_seed)

        reach_curves = self.generate_sample_reach_curves(
            num_publishers, decay_rate, universe_size
        )

        N = max([reach_curve.max_reach for reach_curve in reach_curves]) * 2

        true_a = random_generator.dirichlet(
            np.ones(num_clusters), size=num_publishers
        ).flatten()

        training_reach_points = self.generate_sample_reach_points(
            num_clusters,
            num_publishers,
            true_a,
            reach_curves,
            training_size,
            universe_size,
            N,
            1,
        )

        surface = GeneralizedMixtureReachSurface(
            reach_curves, training_reach_points, num_clusters
        )
        surface._fit()
        test_reach_points = self.generate_sample_reach_points(
            num_clusters,
            num_publishers,
            true_a,
            reach_curves,
            training_size,
            universe_size,
            N,
            2,
        )

        self.assertPointsAlmostEqualToPrediction(surface, training_reach_points)
        self.assertPointsAlmostEqualToPrediction(surface, test_reach_points)


if __name__ == "__main__":
    absltest.main()
