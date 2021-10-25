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
"""A concrete class of the wrapper of filesystem pathlib module."""

import pathlib
from typing import Optional, IO, Iterable

from wfa_planning_evaluation_framework.filesystem_wrappers import filesystem_wrapper_base


class FilesystemPathlibWrapper(filesystem_wrapper_base.FilesystemWrapperBase):
    """Filesystem wrappers for pathlib module.

    The implementation follows the style of the Python standard library's
    [`pathlib` module](https://docs.python.org/3/library/pathlib.html)
    """

    def __init__(self):
        super().__init__()

    def name(self, path: str) -> str:
        """The final path component, if any."""
        return str(pathlib.Path(path).name)

    def parent(self, path: str) -> str:
        """The logical parent of the given path."""
        return str(pathlib.Path(path).parent)

    def glob(self, path: str, pattern: str) -> Iterable[str]:
        """Iterate over the subtree of the given path and yield all existing
        files (of any kind, including directories) matching the given relative
        pattern.
        """
        for obj in pathlib.Path(path).glob(pattern):
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
        return pathlib.Path(path).open(
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
        del newline  # newline argument is added to pathlib.Path after Python v3.11.0
        pathlib.Path(path).write_text(data=data, encoding=encoding, errors=errors)

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
        pathlib.Path(path).mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

    def unlink(self, path: str, missing_ok: bool = False) -> None:
        """
        Remove the file or link at the given path.
        If the path is a directory, use rmdir() instead.
        """
        pathlib.Path(path).unlink(missing_ok=missing_ok)

    def exists(self, path: str) -> bool:
        """
        Whether the given path exists.
        """
        return pathlib.Path(path).exists()

    def is_dir(self, path: str) -> bool:
        """
        Whether the given path is a directory.
        """
        return pathlib.Path(path).is_dir()

    def is_file(self, path: str) -> bool:
        """
        Whether the given path is a regular file (also True for symlinks
        pointing to regular files).
        """
        return pathlib.Path(path).is_file()

    def joinpath(self, *args) -> str:
        """Combine path(s) in one or several arguments, and return a new path"""
        return str(pathlib.Path(*args))
