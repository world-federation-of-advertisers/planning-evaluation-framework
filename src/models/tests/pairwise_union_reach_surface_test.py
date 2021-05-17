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
"""Tests for pairwise_union_reach_surface.py."""

from absl.testing import absltest
import numpy as np
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.models.pairwise_union_reach_surface import PairwiseUnionReachSurface
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve


class LinearCappedReachCurve(ReachCurve):
  """Linear ReachCurve that is capped at max_reach."""

  def _fit(self) -> None:
    self._max_reach = self._data[0].reach()

  def by_impressions(self,
                     impressions: [int],
                     max_frequency: int = 1) -> ReachPoint:
    """Returns the estimated reach for a given impression vector."""
    kplus_frequencies = [
        min(sum(impressions), self.max_reach) // i
        for i in range(1, max_frequency + 1)
    ]
    return ReachPoint(impressions, kplus_frequencies)


class PairwiseUnionReachSurfaceTest(absltest.TestCase):

  def assertPointsAlmostEqualToPrediction(self, surface, reach_points):
    for reach_point in reach_points:
      self.assertAlmostEqual(
          surface.by_impressions(reach_point.impressions).reach(),
          reach_point.reach(),
          delta=reach_point.reach() * 0.00001)

  def generate_true_reach(self, a, reach_curves, impressions):
    p = len(reach_curves)
    reach_vector = [
        reach_curve.by_impressions(impression).reach()
        for reach_curve, impression in zip(reach_curves, impressions)
    ]
    reach = sum(reach_vector)
    for i in range(len(reach_curves)):
      for j in range(len(reach_curves)):
        reach -= (a[i * p + j] * reach_vector[i] * reach_vector[j]) / (
            max(reach_curves[i].max_reach, reach_curves[j].max_reach) * 2)
    return ReachPoint(impressions, [reach])

  def generate_sample_reach_curves(self, num_publishers, decay_rate,
                                   universe_size):
    max_reaches = [
        universe_size * (decay_rate**pub_num)
        for pub_num in range(num_publishers)
    ]
    return [
        LinearCappedReachCurve([ReachPoint([max_reach], (max_reach,))])
        for max_reach in max_reaches
    ]

  def genereate_sample_matrix_a(self, num_publishers):
    true_a = np.array([
        np.random.dirichlet(np.ones(num_publishers)) * np.random.uniform()
        for _ in range(num_publishers)
    ]).flatten()
    for i in range(num_publishers):
      true_a[i * num_publishers + i] = 0
    return true_a

  def generate_sample_reach_points(self, true_a, reach_curves, size,
                                   universe_size):
    reach_points = []
    for _ in range(size):
      impressions = [[np.random.uniform(0, universe_size / 2)]
                     for _ in range(len(reach_curves))]
      reach_points.append(
          self.generate_true_reach(true_a, reach_curves, impressions))

    return reach_points

  def test_by_impressions(self):
    num_publishers = 3
    training_size = 50
    universe_size = 200000
    decay_rate = 0.8

    reach_curves = self.generate_sample_reach_curves(num_publishers, decay_rate,
                                                     universe_size)
    true_a = self.genereate_sample_matrix_a(num_publishers)
    training_reach_points = self.generate_sample_reach_points(
        true_a, reach_curves, training_size, universe_size)

    surface = PairwiseUnionReachSurface(reach_curves, training_reach_points)
    test_reach_points = self.generate_sample_reach_points(
        true_a, reach_curves, training_size, universe_size)
    self.assertPointsAlmostEqualToPrediction(surface, training_reach_points)
    self.assertPointsAlmostEqualToPrediction(surface, test_reach_points)


if __name__ == "__main__":
  absltest.main()
