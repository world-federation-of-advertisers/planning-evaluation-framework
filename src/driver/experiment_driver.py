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
  Local machine without Apache Beam
    python3 experiment_driver.py \
    --data_design_dir=<data_design_dir> \
    --experimental_design=<experimental_design> \
    --output_file=<output_file> \
    --intermediates_dir=<intermediates_dir> \
    --seed=<random_seed> \
    --cores=<number_of_cores> \
    [--analysis_type=single_pub]]

  DirectRuuner:
    python3 experiment_driver.py \
    --data_design_dir=<data_design_dir> \
    --experimental_design=<experimental_design> \
    --output_file=<output_file> \
    --intermediates_dir=<intermediates_dir> \
    --seed=<random_seed> \
    --cores=<number_of_cores> \
    [--analysis_type=single_pub]]
    --use_apache_beam \
    --runner=direct \
    --direct_running_mode=multi_processing

  DataflowRunner:
    python3 experiment_driver.py \
    --data_design_dir=gs://<bucket_name>/<subpath/to/data_design_dir> \
    --experimental_design=gs://<bucket_name>/<subpath/to/experimental_design_dir> \
    --output_file=<output_file> \
    --intermediates_dir=gs://<bucket_name>/<subpath/to/intermediates_dir> \
    --seed=<random_seed> \
    --cores=<number_of_cores> \
    [--analysis_type=single_pub]]
    --use_apache_beam \
    --runner=DataflowRunner \
    --region=<region> \
    --project=<project_id> \
    --staging_location=gs://<bucket_name>/<subpath/to/staging_dir> \
    --setup_file <path/to/setup.py> \
    --extra_package=<path/to/wfa_cardinality_estimation_evaluation_framework-0.0.tar.gz>

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
  analysis_type can be omitted or can be "single_pub".  If "single_pub is
    specified, then additional columns are added to the output data frame
    containing metrics that are specific to single pub analysis.
  cores is an integer specifying the number of cores to be used for
    multithreaded processing.  If cores = 1 (default), then multithreading
    is not used.  If cores < 1, then all available cores are used.
"""

from absl import app
from absl import flags
import argparse
import importlib.util
import math
import numpy as np
import pandas as pd
import sys
from typing import Iterable

from cloudpathlib import CloudPath
from apache_beam.options.pipeline_options import PipelineOptions

from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.driver.experimental_design import (
    ExperimentalDesign,
)
from wfa_planning_evaluation_framework.driver.trial_descriptor import (
    TrialDescriptor,
)


# FLAGS = flags.FLAGS

# flags.DEFINE_string("data_design_dir", None, "Directory containing the data design")
# flags.DEFINE_string(
#     "experimental_design", None, "Name of python file containing experimental design."
# )
# flags.DEFINE_string(
#     "output_file", None, "Name of file where output DataFrame will be written."
# )
# flags.DEFINE_string(
#     "intermediates_dir", None, "Directory where intermediate results will be stored."
# )
# flags.DEFINE_integer("seed", 1, "Seed for the np.random.Generator.")
# flags.DEFINE_integer("cores", 1, "Number of cores to use for multithreading.")
# flags.DEFINE_string(
#     "analysis_type", "", "Specify single_pub if this is a single publisher analysis."
# )


class ExperimentDriver:
    """Runs all experiments in an experimental design against a data design."""

    def __init__(
        self,
        data_design_dir: str,
        experimental_design: str,
        output_file: str,
        intermediate_dir: str,
        random_seed: int,
        cores: int = 1,
        analysis_type: str = "",
    ):
        self._data_design_dir = data_design_dir
        self._experimental_design = experimental_design
        self._output_file = output_file
        self._intermediate_dir = intermediate_dir
        self._seed = random_seed
        self._analysis_type = analysis_type
        self._cores = cores

    def execute(self, use_apache_beam, pipeline_options) -> pd.DataFrame:
        """Performs all experiments defined in an experimental design."""
        from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
        import time
        tic = time.perf_counter()
        data_design = DataDesign(self._data_design_dir)
        experiments = list(self._fetch_experiment_list())
        experimental_design = ExperimentalDesign(
            self._intermediate_dir,
            data_design,
            experiments,
            self._seed,
            self._cores,
            analysis_type=self._analysis_type,
        )
        experimental_design.generate_trials()
        print("=============Generated trials=============")
        print(DataSet.read_data_set.cache_info())

        toc = time.perf_counter()
        print(f"=============Reading all datasets takes {toc - tic:0.4f} seconds=============")
        
        
        result = experimental_design.load(
            use_apache_beam=use_apache_beam,
            pipeline_options=pipeline_options,
        )

        print("=============Loaded=============")
        print(DataSet.read_data_set.cache_info())

        if self._output_file.startswith("gs://"):
            output_cloud_path = CloudPath(self._output_file)
            output_cloud_path.write_text(result.to_csv(index=False))
        else:
            result.to_csv(self._output_file, index=False)

        return result

    def _fetch_experiment_list(self) -> Iterable[TrialDescriptor]:
        """Loads Python module defining the experimental design and fetches it."""
        spec = importlib.util.spec_from_file_location(
            "experiment_generator", self._experimental_design
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["experiment_generator"] = module
        spec.loader.exec_module(module)
        return module.generate_experimental_design_config(seed=self._seed)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_design_dir",
        dest="data_design_dir",
        required=True,
        help="Directory containing the data design.",
    )
    parser.add_argument(
        "--experimental_design",
        dest="experimental_design",
        required=True,
        help="Name of python file containing experimental design.",
    )
    parser.add_argument(
        "--output_file",
        dest="output_file",
        required=True,
        help="Name of file where output DataFrame will be written.",
    )
    parser.add_argument(
        "--intermediates_dir",
        dest="intermediates_dir",
        required=True,
        help="Directory where intermediate results will be stored.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=1,
        help="Seed for the np.random.Generator.",
    )
    parser.add_argument(
        "--cores",
        dest="cores",
        type=int,
        default=1,
        help="Number of cores to use for multithreading.",
    )
    parser.add_argument(
        "--analysis_type",
        dest="analysis_type",
        default="",
        help="Specify single_pub if this is a single publisher analysis.",
    )
    parser.add_argument(
        "--use_apache_beam",
        dest="use_apache_beam",
        action="store_true",
        help="Use Apache Beam.",
    )
    
    return parser


def main(argv):
    parser = create_arg_parser()
    known_args, pipeline_args = parser.parse_known_args(argv)

    experiment_driver = ExperimentDriver(
        known_args.data_design_dir,
        known_args.experimental_design,
        known_args.output_file,
        known_args.intermediates_dir,
        known_args.seed,
        known_args.cores,
        known_args.analysis_type,
    )

    pipeline_args.extend(
        [
            f"--temp_location={known_args.intermediates_dir}",
            f"--direct_num_workers={known_args.cores}",
        ]
    )
    pipeline_options = PipelineOptions(pipeline_args)
    experiment_driver.execute(
        known_args.use_apache_beam, 
        pipeline_options, 
    )


if __name__ == "__main__":
    app.run(main)
