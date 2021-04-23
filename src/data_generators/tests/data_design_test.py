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
from tempfile import TemporaryDirectory

from wfa_planning_evaluation_framework.data_generators.publisher_data_file import (
    PublisherDataFile,
)
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet


class DataDesignTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        pdf11 = PublisherDataFile([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf11")
        pdf12 = PublisherDataFile([(2, 0.03), (4, 0.06)], "pdf12")
        data_set1 = DataSet([pdf11, pdf12], "test1")

        pdf21 = PublisherDataFile([(1, 0.01), (2, 0.02), (2, 0.04), (3, 0.05)], "pdf21")
        pdf22 = PublisherDataFile([(2, 0.03), (3, 0.06)], "pdf22")
        data_set2 = DataSet([pdf21, pdf22], "test2")

        cls.data_design = DataDesign([data_set1, data_set2])

    def test_properties(self):
        self.assertEqual(self.data_design.data_set_count, 2)

    def test_lookup(self):
        ds1 = self.data_design.data_set(0)
        self.assertEqual(ds1.reach_by_impressions([4, 2]).reach(), 4)
        ds2 = self.data_design.data_set(1)
        self.assertEqual(ds2.reach_by_impressions([4, 2]).reach(), 3)

    def test_read_and_write_data_design(self):
        with TemporaryDirectory() as d:
            self.data_design.write_data_design(d)
            new_data_design = DataDesign.read_data_design(d)
            self.assertEqual(new_data_design.data_set_count, 2)
            ds1 = new_data_design.data_set(0)
            self.assertEqual(ds1.reach_by_impressions([4, 2]).reach(), 4)
            ds2 = new_data_design.data_set(1)
            self.assertEqual(ds2.reach_by_impressions([4, 2]).reach(), 3)


if __name__ == "__main__":
    absltest.main()
