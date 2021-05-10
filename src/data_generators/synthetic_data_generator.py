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
"""Generates a DataDesign from DataDesignParameters."""

from absl import app
from absl import flags
import math
from typing import List
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.data_set_parameters import (
    DataSetParameters, GeneratorParameters)
from wfa_planning_evaluation_framework.data_generators.pricing_generator import PricingGenerator
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import FixedPriceGenerator
from wfa_planning_evaluation_framework.data_generators.impression_generator import ImpressionGenerator
from wfa_planning_evaluation_framework.data_generators.homogeneous_impression_generator import HomogeneousImpressionGenerator
from wfa_planning_evaluation_framework.data_generators.publisher_data import PublisherData
from wfa_planning_evaluation_framework.data_generators.test_synthetic_data_design_config import TestSyntheticDataDesignConfig
from wfa_planning_evaluation_framework.data_generators.synthetic_data_design_config import SyntheticDataDesignConfig
from numpy.random import RandomState

FLAGS = flags.FLAGS

flags.DEFINE_string('output_folder', 'TestDataDesign', 'Output Folder.')
flags.DEFINE_string('data_design_config', 'TestConfig', 'Data Desgin Config.')

name_to_config_dict = {'test': TestSyntheticDataDesignConfig}


class SyntheticDataGenerator():
  """Generates a DataDesign with synthetic data derived from parameters.

    This class translates a DataDesignParameters object to a DataDesign by
    constructing the underlying objects for a DataSet and duplicating the
    DataSet with different random seeds.
    """

  def __init__(self, output_folder: str, config: SyntheticDataDesignConfig):
    self._config = config
    self._output_folder = output_folder

  def __call__(self) -> DataDesign:
    data_design = DataDesign(dirpath=self._output_folder)
    data_set_parameters_list = self._config.get_data_set_params_list()
    for data_set_parameters in data_set_parameters_list:
      data_design.add(self.generate_data_set(data_set_parameters))
    return data_design

  def generate_data_set(self, params: DataSetParameters) -> DataSet:
    publishers = []
    publisher_size = params.largest_publisher_size
    for publisher in range(params.num_publishers):
      publishers.append(
          PublisherData.generate_publisher_data(
              params.impression_generator_params.generator(
                  **params.impression_generator_params.params,
                  n=publisher_size),
              params.pricing_generator_params.generator(
                  **params.pricing_generator_params.params),
              self.get_publisher_name(publisher)))
      publisher_size = math.floor(publisher_size *
                                  params.publisher_size_decay_rate)

    return DataSet(publishers, params.name)

  def get_publisher_name(self, publisher_num: str) -> str:
    return 'publisher_' + str(publisher_num + 1)


def main(argv):
  data_generator = SyntheticDataGenerator(
      FLAGS.output_folder, name_to_config_dict[FLAGS.data_design_config])
  data_generator()


if __name__ == '__main__':
  app.run(main)
