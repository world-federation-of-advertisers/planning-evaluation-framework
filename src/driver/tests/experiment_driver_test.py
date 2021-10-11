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
from pathlib import Path
from cloudpathlib.local import LocalGSClient, LocalGSPath
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging

import wfa_planning_evaluation_framework.driver.experiment_driver as experiment_driver
import wfa_planning_evaluation_framework.driver.experimental_design as experimental_design
import wfa_planning_evaluation_framework.data_generators.data_design as data_design
import wfa_planning_evaluation_framework.data_generators.data_set as data_set
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
import wfa_planning_evaluation_framework.data_generators.filesystem_path_client as filesystem_path_client


class FakeExperimentalTrial(ExperimentalTrial):
    def evaluate(self, seed: int):
        return pd.DataFrame(
            {
                "data_set_name": [self._data_set_name],
                "trial_descriptor": [str(self._trial_descriptor)],
            }
        )


class FakeEvaluateTrialDoFn(beam.DoFn):
    def process(self, trial, seed):
        import pandas as pd

        yield pd.DataFrame({"col": [1]})


class ExperimentDriverTest(absltest.TestCase):
    def tearDown(self):
        filesystem_path_client.FilesystemPathClient.reset_default_gs_client()
        LocalGSClient.reset_default_storage_dir()

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

    @patch.object(experimental_design, "EvaluateTrialDoFn", FakeEvaluateTrialDoFn)
    def test_experiment_driver_using_apache_beam_locally(self):
        with TemporaryDirectory() as d:
            data_design_dir = d + "/data"
            output_file = d + "/output.csv"
            intermediate_dir = d + "/intermediates"
            Path(data_design_dir).mkdir(parents=True, exist_ok=True)
            Path(intermediate_dir).mkdir(parents=True, exist_ok=True)

            result = self._execute_experiment_driver_with_sample_experimental_design_using_apache_beam(
                data_design_dir, output_file, intermediate_dir
            )
            self.assertEqual(result.shape[0], 2700)

    @patch.object(filesystem_path_client, "GSClient", LocalGSClient)
    @patch.object(experimental_design, "EvaluateTrialDoFn", FakeEvaluateTrialDoFn)
    def test_experiment_driver_using_apache_beam_and_cloud_path(self):
        fs_path_client = filesystem_path_client.FilesystemPathClient()
        parent_dir_path = fs_path_client.get_fs_path("gs://ExperimentDriverTest/parent")
        data_design_dir_path = parent_dir_path.joinpath("data_design")
        output_file_path = parent_dir_path.joinpath("output.csv")
        intermediate_dir_path = parent_dir_path.joinpath("intermediates")
        data_design_dir_path.joinpath("dummy.txt").write_text(
            "For creating the target directory."
        )
        intermediate_dir_path.joinpath("dummy.txt").write_text(
            "For creating the target directory."
        )

        result = self._execute_experiment_driver_with_sample_experimental_design_using_apache_beam(
            str(data_design_dir_path), str(output_file_path), str(intermediate_dir_path)
        )
        self.assertEqual(result.shape[0], 2700)

    def _execute_experiment_driver_with_sample_experimental_design_using_apache_beam(
        self, data_design_dir, output_file, intermediate_dir
    ):
        data_design_generator = SyntheticDataDesignGenerator(
            data_design_dir, simple_data_design_example.__file__, 1, False
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

        pipeline_args = []
        pipeline_args.extend(
            [
                "--runner=direct",
                "--direct_running_mode=multi_processing",
                f"--temp_location={intermediate_dir}",
                f"--direct_num_workers={num_workers}",
            ]
        )
        pipeline_options = PipelineOptions(pipeline_args)

        logging.disable(logging.CRITICAL)
        result = experiment_driver.execute(
            use_apache_beam=True, pipeline_options=pipeline_options
        )
        logging.disable(logging.NOTSET)
        return result


if __name__ == "__main__":
    absltest.main()
