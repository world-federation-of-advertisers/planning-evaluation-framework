from pathlib import Path
from cloudpathlib import GSClient
from google.cloud import storage


class FilesystemPathClient:
    _default_gs_client = None

    @classmethod
    def get_default_gs_client(cls):
        # When there is no credential provided, GSClients of cloudpathlib
        # will create an anonymous client which can't access non-public
        # buckets. On the other hand, Clients in google.cloud.storage will
        # fall back to the default inferred from the environment. As a
        # result, we first create a Client from google.cloud.storage and
        # then use it to initiate a GSClient object and set it as the
        # default for other operations.
        if cls._default_gs_client is None:
            cls._default_gs_client = GSClient(storage_client=storage.Client())
            cls._default_gs_client.set_as_default_client()

        return cls._default_gs_client

    @classmethod
    def reset_default_gs_client(cls):
        cls._default_gs_client = None

    def _get_path_module(self, path):
        if path.startswith("gs://"):
            return self.get_default_gs_client().GSPath

        return Path

    def get_fs_path(self, path):
        fs_path_module = self._get_path_module(path)
        return fs_path_module(path)
