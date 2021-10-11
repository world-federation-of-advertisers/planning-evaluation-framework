from absl.testing import absltest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pathlib import Path
from cloudpathlib import GSClient, GSPath
from google.cloud import storage
from cloudpathlib.local import LocalGSClient, LocalGSPath
import wfa_planning_evaluation_framework.data_generators.filesystem_path_client as filesystem_path_client
from wfa_planning_evaluation_framework.data_generators.filesystem_path_client import (
    FilesystemPathClient,
)


class FakeGSClient:
    def __init__(self, storage_client=None):
        self.check = False

    def set_as_default_client(self):
        self.check = True


class FilesystemPathClientTest(absltest.TestCase):
    def tearDown(self):
        FilesystemPathClient.reset_default_gs_client()

    def test_get_fs_path_with_local_path(self):
        local_path = "/local/path"
        expected = Path(local_path)

        fs_path_client = FilesystemPathClient()
        path = fs_path_client.get_fs_path(local_path)

        self.assertEqual(expected, path)

    def test_get_fs_path_with_gs_path(self):
        gs_path = "gs://path"
        expected = GSPath(gs_path)

        fs_path_client = FilesystemPathClient()
        path = fs_path_client.get_fs_path(gs_path)

        self.assertEqual(expected, path)

    @patch.object(filesystem_path_client, "GSClient", FakeGSClient)
    def test_get_default_gs_client(self):
        gs_client = FilesystemPathClient.get_default_gs_client()
        self.assertTrue(gs_client.check)


if __name__ == "__main__":
    absltest.main()
