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
"""Generates a DataDesign from SyntheticDataDesignConfig.

Generates a collection of synthetic data sets in a format that can
subsequently be used by the experimental driver for measuring the
performance of reach models.

Example usage:

  python3 synthetic_data_design_generator.py \
    --output_dir=<output_dir>
    --data_design_config=<data_design>
    [--random_seed=<int>]

where

  output_dir is the directory where the DataDesign will be written, and
  data_design_config is the name of a python file that specifies the underlying
    data design.  This file should contain a function with the following
    signature:

       generate_data_design_config(random_generator: np.Generator) ->
         Iterable[DataSetParameters]

    The function returns a list of DataSetParameters objects.  Each
    such object specifies the parameters for one data set that will be
    generated as part of the data design.  For an example of such a
    function, see the files simple_data_design_example.py and
    lhs_data_design_example.py.  The latter file generates a data
    design using a latin hypercube pattern.

Here are some specific examples of usage, assuming that you are in the
data_generators directory.  The following is a simple cartesian product
design:

  python3 synthetic_data_design_generator.py \
     --output_dir=/tmp/simple-data-design \
     --data_design=simple_data_design_example.py 

The following is an example latin hypercube design:

  python3 synthetic_data_design_generator.py \
     --output_dir=/tmp/lhs-example \
     --data_design=lhs_data_design_example.py

"""

from absl import app
from absl import flags
import importlib.util
import math
import numpy as np
import sys
from typing import List
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.data_set_parameters import (
    DataSetParameters,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Directory where data design will be written")
flags.DEFINE_string("data_design_config", None, "Name of data design configuration")
flags.DEFINE_integer("random_seed", 1, "Seed for the random number generator")
flags.DEFINE_bool("verbose", True, "If true, print names of data sets.")


class SyntheticDataDesignGenerator:
    """Generates a DataDesign with synthetic data derived from parameters.

    This class translates a SyntheticDataDesignConfig object to a DataDesign by
    constructing the underlying objects and managing the publisher sizes.
    """

    def __init__(
        self,
        output_dir: str,
        data_design_config: str,
        random_seed: int = 1,
        verbose: bool = True,
    ):
        """Constructor for SyntheticDataGenerator.

        Args:
          output_dir:  String, specifies the directory in the local file
            system where the data design should be written.
          data_design_config:  String, specifies the name of a file in the
            local file system containing Python code that specifies the
            data design.  This file should contain a function with the
            following signature:
              generate_data_design_config(random_generator: np.Generator) ->
                Iterable[DataSetParameters]
          random_seed:  Int, a value used to initialize the random number
            generator.
          verbose:  If True, generates messages as the data design is
            written to disk.
        """
        self._output_dir = output_dir
        self._data_design_config = data_design_config
        self._random_generator = np.random.default_rng(random_seed)
        np.random.seed(random_seed)
        self._verbose = verbose

    def __call__(self) -> DataDesign:
        data_design = DataDesign(dirpath=self._output_dir)
        for data_set_parameters in self._fetch_data_set_parameters_list():
            data_design.add(self._generate_data_set(data_set_parameters))
        return data_design

    def _fetch_data_set_parameters_list(self) -> List[DataSetParameters]:
        spec = importlib.util.spec_from_file_location(
            "data_design_generator", self._data_design_config
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["data_design_generator"] = module
        spec.loader.exec_module(module)
        return module.generate_data_design_config(self._random_generator)

    def _generate_data_set(self, params: DataSetParameters) -> DataSet:
        if self._verbose:
            print(params)
        publishers = []
        publisher_size = params.largest_publisher_size
        publisher_size_decay_rate = (
            1
            if params.num_publishers == 1
            else params.largest_to_smallest_publisher_ratio
            ** (1 / float(params.num_publishers - 1))
        )
        for publisher in range(params.num_publishers):
            publishers.append(
                PublisherData.generate_publisher_data(
                    params.impression_generator_params.generator(
                        **{
                            "n": publisher_size,
                            "random_generator": self._random_generator,
                            **params.impression_generator_params.params,
                        }
                    ),
                    params.pricing_generator_params.generator(
                        **params.pricing_generator_params.params
                    ),
                    str(publisher + 1),
                )
            )
            publisher_size = math.floor(publisher_size * publisher_size_decay_rate)

        overlap_params = {**params.overlap_generator_params.params}
        if "random_generator" in overlap_params:
            overlap_params["random_generator"] = self._random_generator
        if "pricing_generator" in overlap_params:
            overlap_params[
                "pricing_generator"
            ] = params.pricing_generator_params.generator(
                **params.pricing_generator_params.params
            )

        return params.overlap_generator_params.generator(
            publishers, name=str(params), **overlap_params
        )


def main(argv):
    data_design_generator = SyntheticDataDesignGenerator(
        FLAGS.output_dir, FLAGS.data_design_config, FLAGS.random_seed, FLAGS.verbose
    )
    data_design_generator()


if __name__ == "__main__":
    app.run(main)
