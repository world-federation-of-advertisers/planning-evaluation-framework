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

from typing import Optional, IO, Iterable

import cloudpathlib
from google.cloud import storage

from wfa_planning_evaluation_framework.filesystem_wrapper import filesystem_wrapper_base

GSClient = cloudpathlib.GSClient
CloudPath = cloudpathlib.CloudPath


class FilesystemCloudpathWrapper(filesystem_wrapper_base.FilesystemWrapperBase):
    """Filesystem wrappers for cloudpathlib module.

    The implementation follows the style of the Python standard library's
    [`pathlib` module](https://docs.python.org/3/library/pathlib.html)
    """

    _default_client = None

    def __init__(self):
        super().__init__()

    @classmethod
    def set_default_client_to_gs_client(cls) -> None:
        """Get the default Google Cloud Storage client object used by cloudpathlib.

        When there is no credential provided, GSClient of cloudpathlib will
        create an anonymous client which can't access non-public buckets. On
        the other hand, Client in google.cloud.storage will fall back to the
        default inferred from the environment. As a result, if there is no
        default GSClient object, we first create a Client from
        google.cloud.storage and then use it to initiate a GSClient object and
        set it as the default for other operations.
        """
        if cls._default_client is None:
            cls._default_client = GSClient(storage_client=storage.Client())
            cls._default_client.set_as_default_client()

    @classmethod
    def reset_default_client(cls) -> None:
        """Reset the default client"""
        cls._default_client = None

    @classmethod
    def is_valid_to_set_gs_client(
        cls, path: str, filesystem: filesystem_wrapper_base.FilesystemWrapperBase
    ) -> bool:
        return (
            path.startswith("gs://")
            and isinstance(filesystem, cls)
            and not isinstance(cls._default_client, GSClient)
        )

    def name(self, path: str) -> str:
        """The final path component, if any."""
        return str(CloudPath(path).name)

    def parent(self, path: str) -> str:
        """The logical parent of the given path."""
        return str(CloudPath(path).parent)

    def glob(self, path: str, pattern: str) -> Iterable[str]:
        """Iterate over the subtree of the given path and yield all existing
        files (of any kind, including directories) matching the given relative
        pattern.
        """
        for obj in CloudPath(path).glob(pattern):
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
        return CloudPath(path).open(
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
        CloudPath(path).write_text(data=data, encoding=encoding, errors=errors)

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
        # It's not possible to make empty directory on cloud storage
        pass

    def unlink(self, path: str, missing_ok: bool = False) -> None:
        """
        Remove the file or link at the given path.
        If the path is a directory, use rmdir() instead.
        """
        del missing_ok  # Not used in CloudPath
        CloudPath(path).unlink()

    def exists(self, path: str) -> bool:
        """
        Whether the given path exists.
        """
        return CloudPath(path).exists()

    def is_dir(self, path: str) -> bool:
        """
        Whether the given path is a directory.
        """
        return CloudPath(path).is_dir()

    def is_file(self, path: str) -> bool:
        """
        Whether the given path is a regular file (also True for symlinks
        pointing to regular files).
        """
        return CloudPath(path).is_file()

    def joinpath(self, *args) -> str:
        """Combine path(s) in one or several arguments, and return a new path"""
        if not args:
            return ""
        CLOUD_PATH_START_SYMBOL = "://"
        root = args[0].split(CLOUD_PATH_START_SYMBOL)
        cloud_domain = root[0]
        root_folder = root[1]
        path = CloudPath(f"{cloud_domain}://").joinpath(root_folder, *args[1:])
        return str(path)
