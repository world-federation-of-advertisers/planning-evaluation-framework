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

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve


class FakeReachCurve(ReachCurve):
    def _fit(self) -> None:
        self._max_reach = self._data[0].reach()

    def by_impressions(self, impressions: [int], max_frequency: int = 1) -> ReachPoint:
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


class ReachCurveTest(absltest.TestCase):
    def test_reach(self):
        model = FakeReachCurve([ReachPoint([200], (100,))])
        model._fit()
        self.assertEqual(model.by_impressions([200]).reach(), 100)

    def test_max_reach(self):
        model = FakeReachCurve([ReachPoint([200], (100,))])
        model._fit()
        self.assertEqual(model.max_reach, 100)

    def test_impressions_at_reach_quantile(self):
        model = FakeReachCurve([ReachPoint([200], (100,))])
        model._fit()
        self.assertEqual(model.impressions_at_reach_quantile(0.001), 0)
        self.assertEqual(model.impressions_at_reach_quantile(0.9), 90)
        self.assertEqual(model.impressions_at_reach_quantile(0.999), 99)

    def test_impressions_at_spend_quantile(self):
        model = FakeReachCurve([ReachPoint([20000], (10000,))])
        model._fit()
        self.assertEqual(model.spend_at_reach_quantile(0.001), 0)
        self.assertEqual(model.spend_at_reach_quantile(0.9), 90)
        self.assertEqual(model.spend_at_reach_quantile(0.999), 99)


if __name__ == "__main__":
    absltest.main()
