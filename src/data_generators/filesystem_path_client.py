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
"""A client class for managing filesystem-path like modules."""

from pathlib import Path
from cloudpathlib import GSClient
from google.cloud import storage


GSPATH_PREFIX = "gs://"


class FilesystemPathClient:
    """A client that manages different usages of filesystem-path like modules.

    A FilesystemPathClient object takes as input a string of path and return
    the corresponding filesystem-path object based on the prefix of the path.
    """

    _default_gs_client = None

    @classmethod
    def get_default_gs_client(cls):
        """Get the default Google Cloud Storage client object used by cloudpathlib

        When there is no credential provided, GSClient of cloudpathlib will
        create an anonymous client which can't access non-public buckets. On
        the other hand, Client in google.cloud.storage will fall back to the
        default inferred from the environment. As a result, if there is no
        default GSClient object, we first create a Client from
        google.cloud.storage and then use it to initiate a GSClient object and
        set it as the default for other operations. Then, the GSClient object
        is returned.
        """
        if cls._default_gs_client is None:
            cls._default_gs_client = GSClient(storage_client=storage.Client())
            cls._default_gs_client.set_as_default_client()

        return cls._default_gs_client

    @classmethod
    def reset_default_gs_client(cls):
        """Reset the default GSClient"""
        cls._default_gs_client = None

    def _get_path_module(self, path: str):
        """Get the filesystem path module according to the prefix of input path"""
        if path.startswith(GSPATH_PREFIX):
            return self.get_default_gs_client().GSPath

        return Path

    def get_fs_path(self, path: str):
        """Return a filesystem path object"""
        fs_path_module = self._get_path_module(path)
        return fs_path_module(path)
