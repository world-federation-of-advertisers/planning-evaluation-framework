"""TODO(uakyol): DO NOT SUBMIT without one-line documentation for synthetic_data_generator.

TODO(uakyol): DO NOT SUBMIT without a detailed description of synthetic_data_generator.
"""

from absl import app
from absl import flags

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
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

from typing import List
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.data_design_parameters import (
    DataDesignParameters, DataSetParameters, OverlapModelParameters,
    ImpressionGeneratorParameters, PricingGeneratorParameters)
from wfa_planning_evaluation_framework.data_generators.pricing_generator import PricingGenerator
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import  FixedPriceGenerator
from wfa_planning_evaluation_framework.data_generators.impression_generator import ImpressionGenerator
from wfa_planning_evaluation_framework.data_generators.homogeneous_impression_generator import HomogeneousImpressionGenerator
from wfa_planning_evaluation_framework.data_generators.publisher_data import PublisherData
from numpy.random import RandomState

class SyntheticDataGenerator():
  """Generates a DataDesign with synthetic data derived from parameters.

    This class translates a DataDesignParameters object to a DataDesign by
    constructing the underlying objects for a DataSet and duplicating the
    DataSet with different random seeds.
    """

  def __init__(self, params: DataDesignParameters):
    self._params = params
    self.data_set_name = self.getName(params.data_set_parameters)

  def getName(self, data_set_params: DataSetParameters):
    #TODO(uakyol) : implement this after discussion.
    return "homog_p=10_rep=3"

  def __call__(self) -> DataDesign:
    data_design = DataDesign(dirpath=self._params.output_folder)
    for rep in range(self._params.num_reps):
      data_design.add(
          self.generateDataSet(self._params.data_set_parameters, rep))
    return data_design

  def generateDataSet(self, params: DataSetParameters, seed: int) -> DataSet:
    publishers = []
    for publisher in range(params.num_publishers):
      publishers.append(
          PublisherData.generate_publisher_data(
              self.getImpressionGenerator(params.impression_generator_params,
                                          seed),
              self.getPricingGenerator(params.pricing_generator_params, seed),
              str(publisher)))

    self.executeOverlap(publisher, params.overlap_model_params)
    return DataSet(publishers, self.data_set_name +"_rs="+ str(seed))

  def executeOverlap(self, publisher: List[PublisherData],
                     params: OverlapModelParameters):
    #TODO(uakyol) : Use overlap model here when that class is implemented.
    return

  def getImpressionGenerator(self, params: ImpressionGeneratorParameters,
                             seed: int) -> ImpressionGenerator:
    if (params.impression_generator == "homogenous"):
      return HomogeneousImpressionGenerator(params.num_users,
                                            params.homogenous_lambda,
                                            RandomState(seed))
    else:
      raise ValueError(
          f"Invalid impression generator {params.impression_generator}")

  def getPricingGenerator(self, params: PricingGeneratorParameters,
                          seed: int) -> PricingGenerator:
    if (params.pricing_generator == "fixed"):
      return FixedPriceGenerator(params.fixed_price_cost)
    else:
      raise ValueError(f"Invalid price generator {params.pricin_generator}")
