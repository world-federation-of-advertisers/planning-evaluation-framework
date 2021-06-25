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

from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.synthetic_data_design_generator import (
    SyntheticDataDesignGenerator,
)
from wfa_planning_evaluation_framework.data_generators import simple_data_design_example
from wfa_planning_evaluation_framework.data_generators import (
    analysis_example_data_design,
)
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


class FakeExperimentalTrial(ExperimentalTrial):
    def evaluate(self, rng: np.random.Generator):
        return pd.DataFrame(
            {
                "data_set_name": [self._data_set_name],
                "trial_descriptor": [str(self._trial_descriptor)],
            }
        )


class ExperimentDriverTest(absltest.TestCase):
    def test_sample_experimental_design(self):
        sample_design = sample_experimental_design.generate_experimental_design_config(
            np.random.default_rng(seed=1)
        )
        self.assertLen(list(sample_design), 100)

    def test_m3_first_round_experimental_design(self):
        sample_design = (
            m3_first_round_experimental_design.generate_experimental_design_config(
                np.random.default_rng(seed=1)
            )
        )
        self.assertLen(list(sample_design), 192)

    def test_single_publisher_design(self):
        sp_design = list(
            single_publisher_design.generate_experimental_design_config(
                np.random.default_rng(seed=1)
            )
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
                data_design_dir, 1, simple_data_design_example.__file__, False
            )
            data_design_generator()
            rng = np.random.default_rng(seed=1)
            experimental_design = sample_experimental_design.__file__
            experiment_driver = ExperimentDriver(
                data_design_dir, experimental_design, output_file, intermediate_dir, rng
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
                data_design_dir, 1, simple_data_design_example.__file__, False
            )
            data_design_generator()
            rng = np.random.default_rng(seed=1)
            experimental_design = m3_first_round_experimental_design.__file__
            experiment_driver = ExperimentDriver(
                data_design_dir, experimental_design, output_file, intermediate_dir, rng
            )
            result = experiment_driver.execute()
            self.assertEqual(result.shape[0], 5184)

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
                data_design_dir, 1, analysis_example_data_design.__file__, False
            )
            data_design_generator()
            rng = np.random.default_rng(seed=1)
            experimental_design = analysis_example_experimental_design.__file__
            experiment_driver = ExperimentDriver(
                data_design_dir, experimental_design, output_file, intermediate_dir, rng
            )
            result = experiment_driver.execute()
            self.assertEqual(result.shape[0], 1152)


if __name__ == "__main__":
    absltest.main()
