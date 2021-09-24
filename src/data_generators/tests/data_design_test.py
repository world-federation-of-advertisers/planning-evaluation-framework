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
from unittest.mock import patch
from tempfile import TemporaryDirectory

from cloudpathlib.local import LocalGSClient, LocalGSPath

from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
import wfa_planning_evaluation_framework.data_generators.data_design as data_design
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet


class DataDesignTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        pdf11 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf11")
        pdf12 = PublisherData([(2, 0.03), (4, 0.06)], "pdf12")
        cls.data_set1 = DataSet([pdf11, pdf12], "ds1")

        pdf21 = PublisherData([(1, 0.01), (2, 0.02), (2, 0.04), (3, 0.05)], "pdf21")
        pdf22 = PublisherData([(2, 0.03), (3, 0.06)], "pdf22")
        cls.data_set2 = DataSet([pdf21, pdf22], "ds2")

    def tearDown(self):
        LocalGSClient.reset_default_storage_dir()

    @patch.object(data_design, "GSPath", LocalGSPath)
    def test_constructor_with_cloud_path(self):
        file_gs_path = LocalGSPath(
            "gs://DataDesignTest/dir/dummy.txt"
        )
        dir_gs_path = file_gs_path.parent
        file_gs_path.write_text("For creating the target directory.")

        dd = DataDesign(str(dir_gs_path))
        self.assertEqual(dd.count, 0)
        self.assertEqual(dd.names, [])

    def test_properties(self):
        with TemporaryDirectory() as d:
            dd = DataDesign(d)
            self.assertEqual(dd.count, 0)
            self.assertEqual(dd.names, [])
            dd.add(self.data_set1)
            self.assertEqual(dd.count, 1)
            self.assertEqual(dd.names, ["ds1"])
            dd.add(self.data_set2)
            self.assertEqual(dd.count, 2)
            self.assertEqual(dd.names, ["ds1", "ds2"])

    def test_lookup(self):
        with TemporaryDirectory() as d:
            dd1 = DataDesign(d)
            dd1.add(self.data_set1)
            dd1.add(self.data_set2)
            dd2 = DataDesign(d)
            ds1 = dd2.by_name("ds1")
            self.assertEqual(ds1.reach_by_impressions([4, 2]).reach(), 4)
            ds2 = dd2.by_name("ds2")
            self.assertEqual(ds2.reach_by_impressions([4, 2]).reach(), 3)


if __name__ == "__main__":
    absltest.main()
