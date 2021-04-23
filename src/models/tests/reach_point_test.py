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
"""Tests for reach_point.py."""

from absl.testing import absltest

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint


class ReachPointTest(absltest.TestCase):
    def test_impressions(self):
        point = ReachPoint([200, 300], [150, 100])
        self.assertEqual(point.impressions[0], 200)
        self.assertEqual(point.impressions[1], 300)

    def test_reach(self):
        point = ReachPoint([200, 300], [100, 50])
        self.assertEqual(point.reach(1), 100)
        self.assertEqual(point.reach(2), 50)
        self.assertRaises(ValueError, point.reach, 3)

    def test_frequency(self):
        point = ReachPoint([200, 300], [200, 125, 75])
        self.assertEqual(point.frequency(1), 75)
        self.assertEqual(point.frequency(2), 50)
        self.assertRaises(ValueError, point.frequency, 3)

    def test_spends(self):
        point = ReachPoint([200, 300], [100, 50])
        self.assertIsNone(point.spends)
        point2 = ReachPoint([200, 300], [100, 50], [10.0, 20.0])
        self.assertEqual(point2.spends[0], 10.0)
        self.assertEqual(point2.spends[1], 20.0)


if __name__ == "__main__":
    absltest.main()
