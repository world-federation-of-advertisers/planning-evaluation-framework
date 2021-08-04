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
"""Runs a collection of experiments.

Runs a collection of experiments against a collection of data sets,
writing the output as a DataFrame in CSV format.

Usage:

  python3 experiment_driver.py \
    --data_design_dir=<data_design_dir> \
    --experimental_design=<experimental_design> \
    --output_file=<output_file> \
    --intermediates_dir=<intermediates_dir> \
    --seed=<random_seed> \
    --cores=<number_of_cores>

where

  data_design_dir is a directory containing a data design.  A data design
    consists of a collection of directories, each representing a data set.
    Each data set in turn contains simulated or actual impression data
    for each of a set of publishers.  For an example of how to create a
    synthetic data design, see the file synthetic_data_design_generator.py
    in the data_generators director.  
  experimental_design is the name of a file containing a specification of 
    the experimental design.  The experimental_design is given as
    Python code.  This file should contain a function with the following
    signature:

      generate_experimental_design_config(
          random_generator: np.random.Generator,
      ) -> Iterable[TrialDescriptor]

    The generate_experimental_design_config function creates a list of
    TrialDescriptor objects, each describing an experiment that should
    be performed against each of the data sets in the data design.
    For an example of an experimental design, see the file
    sample_experimental_design.py in this directory.
  output_file is the name of the file where the output will be written.
    The output is written in CSV format as a DataFrame containing one
    row of evaluation results for each data set and trial descriptor.
    See the file test_point_aggregator.py for a list of the metrics that
    are computed.
  intermediates_dir is the name of a directory where intermediate results
    are written.  If the experiment driver fails part way through, or if
    new experiments are added, the experiment driver can be re-run.  
    Previously computed results will not be re-computed.
  seed is an integer that is used to seed the random number generator.
  cores is an integer specifying the number of cores to be used for
    multithreaded processing.  If cores = 1 (default), then multithreading
    is not used.  If cores < 1, then all available cores are used.
"""

from absl import app
from absl import flags
import importlib.util
import math
import numpy as np
import pandas as pd
import sys
from typing import Iterable
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.driver.experimental_design import (
    ExperimentalDesign,
)
from wfa_planning_evaluation_framework.driver.trial_descriptor import (
    TrialDescriptor,
)


FLAGS = flags.FLAGS

flags.DEFINE_string("data_design_dir", None, "Directory containing the data design")
flags.DEFINE_string(
    "experimental_design", None, "Name of python file containing experimental design."
)
flags.DEFINE_string(
    "output_file", None, "Name of file where output DataFrame will be written."
)
flags.DEFINE_string(
    "intermediates_dir", None, "Directory where intermediate results will be stored."
)
flags.DEFINE_integer("seed", 1, "Seed for the np.random.Generator.")
flags.DEFINE_integer("cores", 1, "Number of cores to use for multithreading.")


class ExperimentDriver:
    """Runs all experiments in an experimental design against a data design."""

    def __init__(
        self,
        data_design_dir: str,
        experimental_design: str,
        output_file: str,
        intermediate_dir: str,
        random_seed: int,
        cores: int,
    ):
        self._data_design_dir = data_design_dir
        self._experimental_design = experimental_design
        self._output_file = output_file
        self._intermediate_dir = intermediate_dir
        self._rng = np.random.default_rng(seed=random_seed)
        np.random.seed(random_seed)
        self._cores = cores

    def execute(self) -> pd.DataFrame:
        """Performs all experiments defined in an experimental design."""
        data_design = DataDesign(self._data_design_dir)
        experiments = list(self._fetch_experiment_list())
        experimental_design = ExperimentalDesign(
            self._intermediate_dir, data_design, experiments, self._rng, self._cores
        )
        experimental_design.generate_trials()
        result = experimental_design.load()
        result.to_csv(self._output_file)
        return result

    def _fetch_experiment_list(self) -> Iterable[TrialDescriptor]:
        """Loads Python module defining the experimental design and fetches it."""
        spec = importlib.util.spec_from_file_location(
            "experiment_generator", self._experimental_design
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["experiment_generator"] = module
        spec.loader.exec_module(module)
        return module.generate_experimental_design_config(self._rng)


def main(argv):
    experiment_driver = ExperimentDriver(
        FLAGS.data_design_dir,
        FLAGS.experimental_design,
        FLAGS.output_file,
        FLAGS.intermediates_dir,
        FLAGS.seed,
        FLAGS.cores,
    )
    experiment_driver.execute()


if __name__ == "__main__":
    app.run(main)
