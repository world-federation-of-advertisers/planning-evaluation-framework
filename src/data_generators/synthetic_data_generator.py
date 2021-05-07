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

from wfa_planning_evaluation_framework.data_generators import (
    DataDesign, DataSet, DataDesignParameters)


class SyntheticDataGenerator():
  """Generates a DataDesign with synthetic data derived from parameters.

    This class translates a DataDesignParameters object to a DataDesign by
    constructing the underlying objects for a DataSet and duplicating the
    DataSet with different random seeds.
    """

  def __init__(self, params: DataDesignParameters):
    self._params = params
    self.data_set_name = getName(params.data_set_parameters)

  def getName(data_set_params: DataSetParameters):
    #TODO(uakyol) : implement this after discussion.
    return "homog_p=10_rep=3"

  def __call__(self) -> DataDesign:
    data_design = DataDesign(dirpath=self._params.output_folder)
    for rep in range(num_reps):
      data_design.add(generateDataSet(self._params.data_set_parameters), rep)
    return data_design

  def generateDataSet(params: DataSetParameters, seed: int) -> DataSet:
    publishers = []
    for publisher in range(data_set_parameters.num_publishers):
      publishers.append(
          PublisherData.generate_publisher_data(
              getImpressionGenerator(params.impression_generator_params, seed),
              getPricingGenerator(params.pricing_generator_paramsm, seed),
              str(publisher)))

    executeOverlap(publisher, params.overlap_model_params)
    return DataSet(publishers, self.data_set_name + str(seed))

  def executeOverlap(publisher: List[PublisherDataSet],
                     params: OverlapModelParams):
    #TODO(uakyol) : Use overlap model here when that class is implemented.
    return

  def getImpressionGenerator(params: ImpressionGeneratorParameters,
                             seed: int) -> ImpressionGenerator:
    if (params.impression_generator == "homogenous"):
      return HomogeneousImpressionGenerator(params.num_users,
                                            params.homogenous_lambda,
                                            RandomState(seed))
    else:
      raise ValueError(
          f"Invalid impression generator {params.impression_generator}")

  def getPriceGenerator(params: PriceGeneratorParameters,
                        seed: int) -> PriceGenerator:
    if (params.price_generator == "fixed"):
      return FixedPriceGenerator(params.fixed_price_cost)
    else:
      raise ValueError(f"Invalid price generator {params.price_generator}")
