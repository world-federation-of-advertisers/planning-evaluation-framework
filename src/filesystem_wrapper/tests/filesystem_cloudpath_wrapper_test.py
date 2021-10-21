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
"""Tests for filesystem_cloudpath_wrapper.py."""
import tempfile
import pathlib
import os
import glob
from absl.testing import absltest
from unittest.mock import patch

import cloudpathlib.local

from wfa_planning_evaluation_framework.filesystem_wrapper import (
    filesystem_cloudpath_wrapper,
)


@patch.object(
    filesystem_cloudpath_wrapper,
    "CloudPath",
    cloudpathlib.local.LocalGSPath,
)
class FilesystemPathlibWrapperTest(absltest.TestCase):
    @patch.object(
        filesystem_cloudpath_wrapper,
        "GSClient",
        cloudpathlib.local.LocalGSClient,
    )
    def setUp(self):
        # Client setup
        filesystem_cloudpath_wrapper.FilesystemCloudpathWrapper.set_default_client_to_GSClient()
        self.client = cloudpathlib.local.LocalGSClient.get_default_client()

        # Set up paths of temporary directories and files
        self.num_tempfiles = 10
        self.bucket_gs_path = "gs://FilesystemPathlibWrapperTest"
        self.parent_dir = "parent"
        self.child_dir = "child"
        self.dir_path = os.path.join(
            self.bucket_gs_path, self.parent_dir, self.child_dir
        )
        self.tempfile_names = [f"file{i}" for i in range(self.num_tempfiles)]
        self.tempfile_paths = [
            os.path.join(self.dir_path, file_name) for file_name in self.tempfile_names
        ]

        # Create temporary directories and files
        for path in self.tempfile_paths:
            self.client.CloudPath(path).touch()

        # Get a filesystem object of cloudpath wrapper
        self.filesystem = filesystem_cloudpath_wrapper.FilesystemCloudpathWrapper()

    def tearDown(self):
        cloudpathlib.local.LocalGSClient.reset_default_storage_dir()
        filesystem_cloudpath_wrapper.FilesystemCloudpathWrapper.reset_default_client()

    def test_glob(self):
        pattern = "*"
        glob_files = list(self.filesystem.glob(self.dir_path, pattern))
        expected_glob_files = [
            str(p) for p in self.client.CloudPath(self.dir_path).glob(pattern)
        ]
        self.assertEqual(glob_files, expected_glob_files)
        self.assertEqual(len(glob_files), self.num_tempfiles)

    def test_open(self):
        with self.filesystem.open(self.tempfile_paths[0]) as f:
            self.assertTrue(f.readable())

    def test_write_text(self):
        content = "Test write text"
        tempfile_path = self.tempfile_paths[0]
        self.filesystem.write_text(tempfile_path, content)
        with self.client.CloudPath(tempfile_path).open() as f:
            self.assertEqual(f.read(), content)

    def test_mkdir(self):
        # It's not possible to make empty directory on cloud storage
        pass

    def test_unlink(self):
        tempfile_path = self.tempfile_paths[0]
        self.filesystem.unlink(tempfile_path)
        self.assertFalse(self.client.CloudPath(tempfile_path).exists())

    def test_exists(self):
        for p in self.tempfile_paths:
            self.assertTrue(self.filesystem.exists(p))

    def test_is_dir(self):
        self.assertTrue(self.filesystem.is_dir(self.dir_path))
        self.assertFalse(self.filesystem.is_dir(self.tempfile_paths[0]))

    def test_is_file(self):
        self.assertFalse(self.filesystem.is_file(self.dir_path))
        self.assertTrue(self.filesystem.is_file(self.tempfile_paths[0]))

    def test_joinpath(self):
        grandchild_dir = "grandchild_dir"
        tempfile_name = self.tempfile_names[0]
        inputs = (
            self.bucket_gs_path,
            self.parent_dir,
            self.child_dir,
            grandchild_dir,
            tempfile_name,
        )

        path = self.filesystem.joinpath(*inputs)
        expected = os.path.join(*inputs)

        self.assertEqual(path, expected)

    def test_parent(self):
        dir_name = "dir"
        path = os.path.join(self.bucket_gs_path, dir_name)
        self.assertEqual(self.filesystem.parent(path), self.bucket_gs_path)


if __name__ == "__main__":
    absltest.main()
