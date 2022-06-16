# Copyright 2022 The Private Cardinality Estimation Framework Authors
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
"""Tests for independent_model.py."""

from absl.testing import absltest
from typing import List
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.independent_model import IndependentModel


class FakeReachCurve(ReachCurve):
    def by_impressions(
        self, impressions: List[int], max_frequency: int = 1
    ) -> ReachPoint:
        if len(impressions) != 1:
            raise ValueError("Impressions vector must have a length of 1.")
        kplus_reaches = impressions
        for i in range(max_frequency - 1):
            kplus_reaches.append(int(kplus_reaches[-1] / 2))
        return ReachPoint(impressions, kplus_reaches)


class IndependentModelTest(absltest.TestCase):
    def test_by_imprepythossions(self):
        curves = [FakeReachCurve([ReachPoint([0], [0])])] * 2
        model = IndependentModel(reach_curves=curves, universe_size=2**10)
        self.assertSequenceEqual(
            model.by_impressions(
                impressions=[2**9, 2**9], max_frequency=3
            )._kplus_reaches,
            [768, 512, 320],
        )
        curves = [FakeReachCurve([ReachPoint([0], [0])])] * 3
        model = IndependentModel(reach_curves=curves, universe_size=2**10)
        self.assertSequenceEqual(
            model.by_impressions(
                impressions=[2**9, 2**9, 2**9], max_frequency=3
            )._kplus_reaches,
            [896, 704, 512],
        )


if __name__ == "__main__":
    absltest.main()
