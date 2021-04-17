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
"""Tests for goerg_model.py."""

from absl.testing import absltest

from wfa_planning_evaluation_framework.models.single_publisher_impression_model import ReachPoint
from wfa_planning_evaluation_framework.models.goerg_model import GoergModel


class GoergModelTest(absltest.TestCase):

  def test_reach(self):
    model = GoergModel([ReachPoint(200, (100,))])
    self.assertEqual(model.reach(200), 100)
    self.assertAlmostEqual(model.reach(100), 200./3.)

  def test_frequencies(self):
    r = ReachPoint(200, (100,))
    model = GoergModel([ReachPoint(200, (100,))])
    self.assertEqual(model.frequencies(200, 3), [100, 50, 25])

  def test_max_reach(self):
    model = GoergModel([ReachPoint(200, (100,))])
    self.assertEqual(model.max_reach, 200)
    model = GoergModel([ReachPoint(300, (100,))])
    self.assertEqual(model.max_reach, 150)

  def test_impressions_at_reach_quantile(self):
    m = GoergModel([ReachPoint(200, (100,))])
    self.assertEqual(m.impressions_at_reach_quantile(0.001), 0)
    self.assertEqual(m.impressions_at_reach_quantile(1./3.), 100)
    self.assertEqual(m.impressions_at_reach_quantile(0.5), 200)


if __name__ == '__main__':
  absltest.main()
