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
"""A concrete class of the wrapper of filesystem cloudpath module."""

import cloudpathlib
from typing import Optional, IO, Iterable

from wfa_planning_evaluation_framework.filesystem_wrapper import filesystem_wrapper_base


class FilesystemCloudpathWrapper(filesystem_wrapper_base.FilesystemWrapperBase):
    """Filesystem wrappers for cloudpathlib module.

    The implementation follows the style of the Python standard library's
    [`pathlib` module](https://docs.python.org/3/library/pathlib.html)
    """

    def __init__(self):
        super().__init__()

    def glob(self, path: str, pattern: str) -> Iterable[str]:
        """Iterate over the subtree of the given path and yield all existing
        files (of any kind, including directories) matching the given relative
        pattern.
        """
        for obj in cloudpathlib.CloudPath(path).glob(pattern):
            yield str(obj)

    def open(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[bool] = None,
    ) -> IO:
        """
        Open the file pointed by the given path and return a file object, as
        the built-in open() function does.
        """
        return cloudpathlib.CloudPath(path).open(
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    def write_text(
        self,
        path: str,
        data: str,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[bool] = None,
    ) -> None:
        """
        Open the file at the given path in text mode, write to it, and close
        the file.
        """
        del newline  # Not used in CloudPath
        cloudpathlib.CloudPath(path).write_text(
            data=data, encoding=encoding, errors=errors
        )

    def mkdir(
        self,
        path: str,
        mode: Optional[int] = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        """
        Create a new directory at the given path.
        """
        # not possible to make empty directory on cloud storage
        pass

    def unlink(self, path: str, missing_ok: bool = False) -> None:
        """
        Remove the file or link at the given path.
        If the path is a directory, use rmdir() instead.
        """
        del missing_ok  # Not used in CloudPath
        cloudpathlib.CloudPath(path).unlink()

    def exists(self, path: str) -> bool:
        """
        Whether the given path exists.
        """
        return cloudpathlib.CloudPath(path).exists()

    def is_dir(self, path: str) -> bool:
        """
        Whether the given path is a directory.
        """
        return cloudpathlib.CloudPath(path).is_dir()

    def is_file(self, path: str) -> bool:
        """
        Whether the given path is a regular file (also True for symlinks
        pointing to regular files).
        """
        return cloudpathlib.CloudPath(path).is_file()

    def joinpath(self, *args) -> str:
        """Combine path(s) in one or several arguments, and return a new path"""
        if not args:
            return ""
        CLOUD_PATH_START_SYMBOL = "://"
        root = args[0].split(CLOUD_PATH_START_SYMBOL)
        cloud_domain = root[0]
        root_folder = root[1]
        path = cloudpathlib.CloudPath(f"{cloud_domain}://").joinpath(
            root_folder, *args[1:]
        )
        return str(path)

    def parent(self, path: str) -> str:
        """The logical parent of the given path."""
        return str(cloudpathlib.CloudPath(path).parent)
