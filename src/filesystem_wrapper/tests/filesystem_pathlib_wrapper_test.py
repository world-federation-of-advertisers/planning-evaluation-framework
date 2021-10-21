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
"""Tests for filesystem_pathlib_wrapper.py."""
import tempfile
import pathlib
import os
import glob
from absl.testing import absltest

from wfa_planning_evaluation_framework.filesystem_wrapper import (
    filesystem_pathlib_wrapper,
)


class FilesystemPathlibWrapperTest(absltest.TestCase):
    def setUp(self):
        self.filesystem = filesystem_pathlib_wrapper.FilesystemPathlibWrapper()

    def _generate_tempfiles(self, tempdir, tempfile_names=["file"]):
        for name in tempfile_names:
            pathlib.Path(tempdir).joinpath(name).touch()

    def test_name(self):
        parent_dir = "parent_dir"
        child_dir = "child_dir"
        path = os.path.join(parent_dir, child_dir)

        self.assertEqual(self.filesystem.name(path), child_dir)

    def test_parent(self):
        parent_dir = "parent_dir"
        child_dir = "child_dir"
        path = os.path.join(parent_dir, child_dir)

        self.assertEqual(self.filesystem.parent(path), parent_dir)

    def test_glob(self):
        pattern = "*"
        num_tempfiles = 10
        tempfile_names = [f"file{i}" for i in range(num_tempfiles)]

        with tempfile.TemporaryDirectory() as d:
            self._generate_tempfiles(d, tempfile_names)
            glob_files = list(self.filesystem.glob(d, pattern))
            expected_glob_files = glob.glob(os.path.join(d, pattern))
            self.assertEqual(glob_files, expected_glob_files)
            self.assertEqual(len(glob_files), num_tempfiles)

    def test_open(self):
        tempfile_name = "file"

        with tempfile.TemporaryDirectory() as d:
            self._generate_tempfiles(d, [tempfile_name])
            tempfile_path = os.path.join(d, tempfile_name)
            with self.filesystem.open(tempfile_path) as f:
                self.assertTrue(f.readable())

    def test_write_text(self):
        tempfile_name = "file"
        content = "Test write text"

        with tempfile.TemporaryDirectory() as d:
            self._generate_tempfiles(d, [tempfile_name])
            tempfile_path = os.path.join(d, tempfile_name)
            self.filesystem.write_text(tempfile_path, content)

            with open(tempfile_path) as f:
                self.assertEqual(f.read(), content)

    def test_mkdir(self):
        tempdir_name = "dir"

        with tempfile.TemporaryDirectory() as d:
            tempdir_path = os.path.join(d, tempdir_name)
            self.filesystem.mkdir(tempdir_path)
            self.assertTrue(os.path.exists(tempdir_path))

    def test_unlink(self):
        tempfile_name = "file"

        with tempfile.TemporaryDirectory() as d:
            self._generate_tempfiles(d, [tempfile_name])
            tempfile_path = os.path.join(d, tempfile_name)
            self.filesystem.unlink(tempfile_path)
            self.assertFalse(os.path.exists(tempfile_path))

    def test_exists(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self.filesystem.exists(d))

    def test_is_dir(self):
        tempfile_name = "file"

        with tempfile.TemporaryDirectory() as d:
            self._generate_tempfiles(d, [tempfile_name])
            tempfile_path = os.path.join(d, tempfile_name)
            self.assertTrue(self.filesystem.is_dir(d))
            self.assertFalse(self.filesystem.is_dir(tempfile_path))

    def test_is_file(self):
        tempfile_name = "file"

        with tempfile.TemporaryDirectory() as d:
            self._generate_tempfiles(d, [tempfile_name])
            tempfile_path = os.path.join(d, tempfile_name)
            self.assertFalse(self.filesystem.is_file(d))
            self.assertTrue(self.filesystem.is_file(tempfile_path))

    def test_joinpath(self):
        parent_dir = "parent_dir"
        child_dir = "child_dir"
        grandchild_dir = "grandchild_dir"

        path = self.filesystem.joinpath(parent_dir, child_dir, grandchild_dir)
        expected = os.path.join(parent_dir, child_dir, grandchild_dir)

        self.assertEqual(path, expected)


if __name__ == "__main__":
    absltest.main()
