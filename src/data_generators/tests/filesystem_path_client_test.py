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
"""Tests for filesystem_path_client.py."""

from absl.testing import absltest
from absl.testing import parameterized
from unittest.mock import patch

from pathlib import Path
from cloudpathlib import GSPath
import wfa_planning_evaluation_framework.data_generators.filesystem_path_client as filesystem_path_client
from wfa_planning_evaluation_framework.data_generators.filesystem_path_client import (
    FilesystemPathClient,
)


class FakeGSClient:
    def __init__(self, storage_client=None):
        self.check = False

    def set_as_default_client(self):
        self.check = True


class FilesystemPathClientTest(parameterized.TestCase):
    def tearDown(self):
        FilesystemPathClient.reset_default_gs_client()

    @parameterized.named_parameters(
        {
            "testcase_name": "with_local_path",
            "path": "/local/path",
            "expected": Path("/local/path"),
        },
        {
            "testcase_name": "with_gs_path",
            "path": "gs://path",
            "expected": GSPath("gs://path"),
        },
    )
    def test_get_fs_path(self, path, expected):
        fs_path_client = FilesystemPathClient()
        path = fs_path_client.get_fs_path(path)
        self.assertEqual(expected, path)

    @patch.object(filesystem_path_client, "GSClient", FakeGSClient)
    def test_get_default_gs_client(self):
        gs_client = FilesystemPathClient.get_default_gs_client()
        self.assertTrue(gs_client.check)


if __name__ == "__main__":
    absltest.main()
