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
"""Tests for experiment_driver.py."""

from absl.testing import absltest
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pathlib
import logging

import cloudpathlib.local
import apache_beam as beam
import apache_beam.options.pipeline_options as pipeline_options

import wfa_planning_evaluation_framework.driver.experimental_design as experimental_design
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.synthetic_data_design_generator import (
    SyntheticDataDesignGenerator,
)
from wfa_planning_evaluation_framework.data_generators import simple_data_design_example
from wfa_planning_evaluation_framework.driver.experiment_driver import ExperimentDriver
from wfa_planning_evaluation_framework.driver.experimental_trial import (
    ExperimentalTrial,
)
from wfa_planning_evaluation_framework.driver import sample_experimental_design
from wfa_planning_evaluation_framework.driver import m3_first_round_experimental_design
from wfa_planning_evaluation_framework.driver import (
    analysis_example_experimental_design,
)
from wfa_planning_evaluation_framework.driver import single_publisher_design
from wfa_planning_evaluation_framework.filesystem_wrappers import (
    filesystem_pathlib_wrapper,
)
from wfa_planning_evaluation_framework.filesystem_wrappers import (
    filesystem_cloudpath_wrapper,
)


class FakeExperimentalTrial(ExperimentalTrial):
    def evaluate(self, seed: int, filesystem=None):
        return pd.DataFrame(
            {
                "data_set_name": [self._data_set_name],
                "trial_descriptor": [str(self._trial_descriptor)],
            }
        )


class FakeEvaluateTrialDoFn(beam.DoFn):
    def process(self, trial, seed, filesystem):
        import pandas as pd

        yield pd.DataFrame({"col": [1]})


def fake_open(
    self,
    path,
    mode="r",
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
):
    if path.startswith("gs://"):
        path = cloudpathlib.local.LocalGSPath(path)
    else:
        path = pathlib.Path(path)

    return path.open(
        mode=mode,
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
    )


class ExperimentDriverTest(absltest.TestCase):
    def setUp(self):
        self.client = cloudpathlib.local.LocalGSClient()
        self.client.set_as_default_client()

    def tearDown(self):
        cloudpathlib.local.localclient.clean_temp_dirs()
        cloudpathlib.local.LocalGSClient.reset_default_storage_dir()

    def test_sample_experimental_design(self):
        sample_design = sample_experimental_design.generate_experimental_design_config(
            seed=1
        )
        self.assertLen(list(sample_design), 100)

    def test_m3_first_round_experimental_design(self):
        sample_design = (
            m3_first_round_experimental_design.generate_experimental_design_config(
                seed=1
            )
        )
        self.assertLen(list(sample_design), 288)

    def test_single_publisher_design(self):
        sp_design = list(
            single_publisher_design.generate_experimental_design_config(seed=1)
        )
        self.assertLen(sp_design, 100)

    @patch(
        "wfa_planning_evaluation_framework.driver.experiment.ExperimentalTrial",
        new=FakeExperimentalTrial,
    )
    def test_experiment_driver_with_sample_experimental_design(self):
        with TemporaryDirectory() as d:
            data_design_dir = d + "/data"
            output_file = d + "/output"
            intermediate_dir = d + "/intermediates"
            data_design_generator = SyntheticDataDesignGenerator(
                data_design_dir, simple_data_design_example.__file__, 1, False
            )
            data_design_generator()
            experimental_design = sample_experimental_design.__file__
            experiment_driver = ExperimentDriver(
                data_design_dir, experimental_design, output_file, intermediate_dir, 1
            )
            result = experiment_driver.execute()
            self.assertEqual(result.shape[0], 2700)

    @patch(
        "wfa_planning_evaluation_framework.driver.experiment.ExperimentalTrial",
        new=FakeExperimentalTrial,
    )
    def test_experiment_driver_with_m3_first_round_experimental_design(self):
        with TemporaryDirectory() as d:
            data_design_dir = d + "/data"
            output_file = d + "/output"
            intermediate_dir = d + "/intermediates"
            data_design_generator = SyntheticDataDesignGenerator(
                data_design_dir, simple_data_design_example.__file__, 1, False
            )
            data_design_generator()
            experimental_design = m3_first_round_experimental_design.__file__
            experiment_driver = ExperimentDriver(
                data_design_dir, experimental_design, output_file, intermediate_dir, 1
            )
            result = experiment_driver.execute()
            self.assertEqual(result.shape[0], 7776)

    @patch(
        "wfa_planning_evaluation_framework.driver.experiment.ExperimentalTrial",
        new=FakeExperimentalTrial,
    )
    def test_experiment_driver_with_analysis_example_experimental_design(self):
        with TemporaryDirectory() as d:
            data_design_dir = d + "/data"
            output_file = d + "/output"
            intermediate_dir = d + "/intermediates"
            data_design_generator = SyntheticDataDesignGenerator(
                data_design_dir, simple_data_design_example.__file__, 1, False
            )
            data_design_generator()
            experimental_design = analysis_example_experimental_design.__file__
            experiment_driver = ExperimentDriver(
                data_design_dir, experimental_design, output_file, intermediate_dir, 1
            )

            result = experiment_driver.execute()
            self.assertEqual(result.shape[0], 2592)

    @patch.object(
        experimental_design,
        "EvaluateTrialDoFn",
        FakeEvaluateTrialDoFn,
    )
    def test_experiment_driver_using_apache_beam_locally(self):
        with TemporaryDirectory() as d:
            data_design_dir = d + "/data"
            output_file = d + "/output.csv"
            intermediate_dir = d + "/intermediates"
            pathlib.Path(data_design_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(intermediate_dir).mkdir(parents=True, exist_ok=True)

            filesystem = filesystem_pathlib_wrapper.FilesystemPathlibWrapper()
            result = self._execute_experiment_driver_with_sample_experimental_design_using_apache_beam(
                data_design_dir, output_file, intermediate_dir, filesystem
            )
            self.assertEqual(result.shape[0], 2700)

    @patch.object(
        experimental_design,
        "EvaluateTrialDoFn",
        FakeEvaluateTrialDoFn,
    )
    @patch.object(
        filesystem_cloudpath_wrapper,
        "CloudPath",
        cloudpathlib.local.LocalGSPath,
    )
    @patch.object(
        filesystem_cloudpath_wrapper.FilesystemCloudpathWrapper,
        "set_default_client_to_gs_client",
        cloudpathlib.local.LocalGSClient.get_default_client,
    )
    @patch.object(
        filesystem_cloudpath_wrapper.FilesystemCloudpathWrapper,
        "open",
        fake_open,
    )
    def test_experiment_driver_using_apache_beam_and_cloud_path(self):
        parent_dir_path = self.client.CloudPath("gs://ExperimentDriverTest/parent")
        data_design_dir_path = parent_dir_path.joinpath("data_design")
        output_file_path = parent_dir_path.joinpath("output.csv")
        intermediate_dir_path = parent_dir_path.joinpath("intermediates")
        data_design_dir_path.joinpath("dummy.txt").write_text(
            "For creating the target directory."
        )
        intermediate_dir_path.joinpath("dummy.txt").write_text(
            "For creating the target directory."
        )
        filesystem = filesystem_cloudpath_wrapper.FilesystemCloudpathWrapper()

        result = self._execute_experiment_driver_with_sample_experimental_design_using_apache_beam(
            str(data_design_dir_path),
            str(output_file_path),
            str(intermediate_dir_path),
            filesystem,
            use_cloud_path=True,
        )
        self.assertEqual(result.shape[0], 2700)

    def _execute_experiment_driver_with_sample_experimental_design_using_apache_beam(
        self,
        data_design_dir,
        output_file,
        intermediate_dir,
        filesystem,
        use_cloud_path=False,
    ):
        data_design_dir_for_generator = None
        if use_cloud_path:
            data_design_dir_cloud_path = self.client.CloudPath(data_design_dir)
            data_design_dir_for_generator = self.client._cloud_path_to_local(
                data_design_dir_cloud_path
            )
        else:
            data_design_dir_for_generator = data_design_dir

        data_design_generator = SyntheticDataDesignGenerator(
            data_design_dir_for_generator, simple_data_design_example.__file__, 1, False
        )
        data_design_generator()
        experimental_design = sample_experimental_design.__file__

        num_workers = 6
        experiment_driver = ExperimentDriver(
            data_design_dir,
            experimental_design,
            output_file,
            intermediate_dir,
            random_seed=1,
            cores=num_workers,
        )

        temp_location = None
        if use_cloud_path:
            intermediate_dir_cloud_path = self.client.CloudPath(intermediate_dir)
            temp_location = self.client._cloud_path_to_local(
                intermediate_dir_cloud_path
            )
        else:
            temp_location = intermediate_dir

        pipeline_args = []
        pipeline_args.extend(
            [
                "--runner=direct",
                "--direct_running_mode=multi_processing",
                f"--temp_location={temp_location}",
                f"--direct_num_workers={num_workers}",
            ]
        )
        options = pipeline_options.PipelineOptions(pipeline_args)

        logging.disable(logging.CRITICAL)
        result = experiment_driver.execute(
            use_apache_beam=True,
            pipeline_options=options,
            filesystem=filesystem,
        )
        logging.disable(logging.NOTSET)
        return result


if __name__ == "__main__":
    absltest.main()
