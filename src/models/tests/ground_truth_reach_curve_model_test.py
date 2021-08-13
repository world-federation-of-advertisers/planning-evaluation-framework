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
"""Tests for GroundTruthReachCurveModel"""

from absl.testing import absltest
from numpy.random import RandomState
from tempfile import TemporaryDirectory

from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.models.ground_truth_reach_curve_model import (
    GroundTruthReachCurveModel,
)


class GroundTruthReachCurveModelTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "test")
        cls.data_set = data_set
        cls.curve1 = GroundTruthReachCurveModel(data_set, 0)
        cls.curve2 = GroundTruthReachCurveModel(data_set, 1)

    def test_by_spend(self):
        self.assertEqual(self.curve1.by_spend([0.0]).reach(1), 0)
        self.assertEqual(self.curve2.by_spend([0.0]).reach(1), 0)
        self.assertEqual(self.curve1.by_spend([0.02]).reach(1), 2)
        self.assertEqual(self.curve2.by_spend([0.03]).reach(1), 1)

    def test_by_impressions(self):
        self.assertEqual(self.curve1.by_impressions([0]).reach(1), 0)
        self.assertEqual(self.curve1.by_impressions([3]).reach(1), 2)
        self.assertEqual(self.curve2.by_impressions([0]).reach(1), 0)
        self.assertEqual(self.curve2.by_impressions([2]).reach(1), 2)

    def test_impressions_for_spend(self):
        self.assertEqual(self.curve1.impressions_for_spend(0.0), 0)
        self.assertEqual(self.curve1.impressions_for_spend(0.05), 4)
        self.assertEqual(self.curve2.impressions_for_spend(0.0), 0)
        self.assertEqual(self.curve2.impressions_for_spend(0.05), 1)


if __name__ == "__main__":
    absltest.main()
