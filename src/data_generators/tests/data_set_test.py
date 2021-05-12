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
"""Tests for publisher_data_file.py."""

from absl.testing import absltest
from numpy.random import RandomState
from tempfile import TemporaryDirectory

from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)
from wfa_planning_evaluation_framework.data_generators.homogeneous_impression_generator import (
    HomogeneousImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet


class DataSetTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "test")
        cls.data_set = data_set

    def test_properties(self):
        self.assertEqual(self.data_set.publisher_count, 2)
        self.assertEqual(self.data_set.name, "test")

    def test_spend_by_impressions(self):
        self.assertEqual(self.data_set.spend_by_impressions([0, 0]), [0, 0])
        self.assertEqual(self.data_set.spend_by_impressions([2, 0]), [0.02, 0])
        self.assertEqual(self.data_set.spend_by_impressions([0, 2]), [0, 0.06])
        self.assertEqual(self.data_set.spend_by_impressions([2, 2]), [0.02, 0.06])

    def test_impressions_by_spend(self):
        self.assertEqual(self.data_set.impressions_by_spend([0, 0]), [0, 0])
        self.assertEqual(self.data_set.impressions_by_spend([0.02, 0]), [2, 0])
        self.assertEqual(self.data_set.impressions_by_spend([0, 0.06]), [0, 2])
        self.assertEqual(self.data_set.impressions_by_spend([0.02, 0.06]), [2, 2])

    def test_reach_by_impressions(self):
        self.assertEqual(self.data_set.reach_by_impressions([0, 0]).reach(), 0)
        self.assertEqual(self.data_set.reach_by_impressions([4, 0]).reach(), 3)
        self.assertEqual(self.data_set.reach_by_impressions([4, 0]).reach(2), 1)
        self.assertEqual(self.data_set.reach_by_impressions([0, 2]).reach(), 2)
        self.assertEqual(self.data_set.reach_by_impressions([4, 2]).reach(), 4)
        self.assertEqual(self.data_set.reach_by_impressions([3, 1]).reach(), 2)

    def test_reach_by_spend(self):
        self.assertEqual(self.data_set.reach_by_spend([0, 0]).reach(), 0)
        self.assertEqual(self.data_set.reach_by_spend([0.05, 0]).reach(), 3)
        self.assertEqual(self.data_set.reach_by_spend([0.05, 0]).reach(2), 1)
        self.assertEqual(self.data_set.reach_by_spend([0, 0.06]).reach(), 2)
        self.assertEqual(self.data_set.reach_by_spend([0.05, 0.06]).reach(), 4)
        self.assertEqual(self.data_set.reach_by_spend([0.04, 0.03]).reach(), 2)

    def test_read_and_write_data_set(self):
        with TemporaryDirectory() as d:
            self.data_set.write_data_set(d)
            new_data_set = DataSet.read_data_set("{}/test".format(d))
            self.assertEqual(new_data_set.publisher_count, 2)
            self.assertEqual(new_data_set.name, "test")
            self.assertEqual(new_data_set.reach_by_impressions([4, 0]).reach(), 3)
            self.assertEqual(new_data_set.reach_by_impressions([0, 2]).reach(), 2)
            self.assertEqual(new_data_set.reach_by_impressions([4, 2]).reach(), 4)


if __name__ == "__main__":
    absltest.main()
