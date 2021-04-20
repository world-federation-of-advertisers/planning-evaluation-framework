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
"""Tests for single_publisher_impression_model.py."""

from absl.testing import absltest

from wfa_planning_evaluation_framework.models.reach_by_impression_model import ReachPoint
from wfa_planning_evaluation_framework.models.reach_by_impression_model import ReachCurve


class FakeReachCurve(ReachCurve):

  def _fit(self) -> None:
    self._max_reach = self.data[0].reach_at_frequency[0]

  def frequencies(self, impressions: int, max_frequency: int) -> int:
    return [
        min(impressions, self.max_reach) / i
        for i in range(1, max_frequency + 1)
    ]


class ReachCurveTest(absltest.TestCase):

  def test_object_creation(self):
    model = FakeReachCurve([ReachPoint(200, (100,))])
    self.assertLen(model.data, 1)
    self.assertEqual(model.data[0].impressions, 200)
    self.assertEqual(model.data[0].reach_at_frequency[0], 100)

  def test_reach(self):
    model = FakeReachCurve([ReachPoint(100, (200,))])
    self.assertEqual(model.reach(100), 100)

  def test_frequencies(self):
    model = FakeReachCurve([ReachPoint(100, (200,))])
    self.assertEqual(model.frequencies(100, 2), [100, 50])

  def test_max_reach(self):
    model = FakeReachCurve([ReachPoint(100, (200,))])
    self.assertEqual(model.max_reach, 200)

  def test_impressions_at_reach_quantile(self):
    model = FakeReachCurve([ReachPoint(100, (200,))])
    self.assertEqual(model.impressions_at_reach_quantile(0.001), 0)
    self.assertEqual(model.impressions_at_reach_quantile(0.9), 180)
    self.assertEqual(model.impressions_at_reach_quantile(0.999), 199)


if __name__ == '__main__':
  absltest.main()
