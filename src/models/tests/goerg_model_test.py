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

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.goerg_model import GoergModel


class GoergModelTest(absltest.TestCase):
    def test_reach(self):
        model = GoergModel([ReachPoint([200], (100,))])
        self.assertEqual(model.by_impressions([200]).reach(), 100)
        self.assertAlmostEqual(model.by_impressions([100]).reach(), 200.0 / 3.0)

    def test_frequencies(self):
        model = GoergModel([ReachPoint([200], (100,))])
        self.assertEqual(model.by_impressions([200], 3).reach(1), 100)
        self.assertEqual(model.by_impressions([200], 3).reach(2), 50)
        self.assertEqual(model.by_impressions([200], 3).reach(3), 25)

    def test_max_reach(self):
        model = GoergModel([ReachPoint([200], (100,))])
        self.assertEqual(model.max_reach, 200)
        model = GoergModel([ReachPoint([300], (100,))])
        self.assertEqual(model.max_reach, 150)

    def test_impressions_at_reach_quantile(self):
        model = GoergModel([ReachPoint([200], (100,))])
        self.assertEqual(model.impressions_at_reach_quantile(0.001), 0)
        self.assertEqual(model.impressions_at_reach_quantile(1.0 / 3.0), 100)
        self.assertEqual(model.impressions_at_reach_quantile(0.5), 200)

    def test_spend_at_reach_quantile(self):
        model = GoergModel([ReachPoint([20000], (10000,), [200.0])])
        self.assertEqual(model.spend_at_reach_quantile(0.001), 0)
        self.assertEqual(model.spend_at_reach_quantile(1.0 / 3.0), 100)
        self.assertEqual(model.spend_at_reach_quantile(0.5), 200)


if __name__ == "__main__":
    absltest.main()
