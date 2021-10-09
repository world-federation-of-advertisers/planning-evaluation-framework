from pathlib import Path
from cloudpathlib import GSClient


class FilesystemPathClient:
    def _get_path_module(self, path):
        if path.startswith("gs://"):
            # When there is no credential provided, GSClients of cloudpathlib
            # will create an anonymous client which can't access non-public
            # buckets. On the other hand, Clients in google.cloud.storage will
            # fall back to the default inferred from the environment. As a
            # result, we first create a Client from google.cloud.storage and
            # then use it to initiate a GSClient object and set it as the
            # default for other operations.
            client = GSClient.get_default_client()
            return client.GSPath

        return Path

    def get_fs_path(self, path):
        fs_path_module = self._get_path_module(path)
        return fs_path_module(path)
