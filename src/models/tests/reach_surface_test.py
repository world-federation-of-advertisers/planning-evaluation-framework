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
"""Tests for reach_surface.py."""

from absl.testing import absltest

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface


class FakeReachSurface(ReachSurface):

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

  def by_spend(self, spend: [float], max_frequency: int = 1) -> ReachPoint:
    """Returns the estimated reach for a given spend vector."""
    impressions = [100 * s for s in spend]
    kplus_frequencies = [
        min(sum(impressions), self.max_reach) // i
        for i in range(1, max_frequency + 1)
    ]
    return ReachPoint(impressions, kplus_frequencies, spend)


class ReachSurfaceTest(absltest.TestCase):

  def test_by_impressions(self):
    surface = FakeReachSurface([ReachPoint([100, 200], [100])])
    self.assertEqual(surface.by_impressions([100, 200]).reach(), 100)
    self.assertEqual(surface.by_impressions([100, 200], 3).reach(2), 50)
    self.assertEqual(surface.by_impressions([100, 200], 3).reach(3), 33)

  def test_by_spend(self):
    surface = FakeReachSurface([ReachPoint([1000, 2000], [1000])])
    self.assertEqual(surface.by_spend([1, 2]).reach(), 300)
    self.assertEqual(surface.by_spend([1, 2], 3).reach(2), 150)
    self.assertEqual(surface.by_spend([1, 2], 3).reach(3), 100)
    self.assertEqual(surface.by_spend([0.5, 1], 3).reach(3), 50)

  def test_max_reach(self):
    surface = FakeReachSurface([ReachPoint([1000, 2000], [1000])])
    self.assertEqual(surface.max_reach, 1000)


if __name__ == '__main__':
  absltest.main()
