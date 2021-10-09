from pathlib import Path
from cloudpathlib import GSClient


class FilesystemPathClient:
    def _get_path_module(self, path):
        if path.startswith("gs://"):
            client = GSClient.get_default_client()
            return client.GSPath

        return Path

    def get_fs_path(self, path):
        fs_path_module = self._get_path_module(path)
        return fs_path_module(path)
