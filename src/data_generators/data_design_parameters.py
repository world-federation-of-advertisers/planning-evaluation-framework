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
"""Defines the parameters for generating a DataDesign."""

from numpy.random import Generator
from typing import NamedTuple

class PricingGeneratorParameters(NamedTuple):
  """Parameters to create one PricingGenerator.

    pricing_generator:  Name of the pricing generator. Can be "fixed" or
        "variable".
    fixed_price_cost : Cost argument for the FixedPriceGenerator.
    variable_price_prob : Probability Argument for VariablePriceGenerator.
    variable_price_mean : Mean Argument for VariablePriceGenerator.
    variable_price_std_dev : Standard Deviation Argument for
        VariablePriceGenerator.
    """

  pricing_generator: str
  fixed_price_cost: float = 0.0
  variable_price_prob: float = 0.0
  variable_price_mean: float = 0.0
  variable_price_std_dev: float = 0.0


class ImpressionGeneratorParameters(NamedTuple):
  """Parameters to create one ImpressionGenerator.

    impression_generator:  Name of the impression generator. Can be "homogenous"
    num_users : Number of users for this impression generator.
    homogenous_lambda : Lambda parameter for HomogenousImpressionGenerator.
    """

  impression_generator: str
  num_users: int
  homogenous_lambda: float

class OverlapModelParameters(NamedTuple):
  """Parameters to create one OverlapModel.

    overlap_model:  Name of the overlap model
    """

  overlap_model: str
  # TODO(uakyol) : Add overlap model params when that base class is implemented.

class DataSetParameters(NamedTuple):
  """Parameters to create one DataSet.

  Does not include randomness

    num_publishers: Number of publishers in this DataSet.
    largest_publisher_size: Maximum possible reach of the largest publisher.
    publisher_size_decay_rate: Decay rate for calculating publisher sizes from
        the largest publisher size.
    pricing_generator_params: Parameters for the pricing generator.
    impression_generator_params: Parameters for the impression generator.
    """

  num_publishers: int
  largest_publisher_size: int
  publisher_size_decay_rate: int
  pricing_generator_params: PricingGeneratorParameters
  impression_generator_params: ImpressionGeneratorParameters
  overlap_model_params: OverlapModelParameters

class DataDesignParameters(NamedTuple):
  """Parameters to create one DataDesign.

    num_reps:  Number of repetitions of the DataSet for this DataDesign.
    data_set_parameters:  Recepie of creating a DataSet. Each rep creates a new
        DataSet from these parameters by introducing randomness to it.
    output_folder:  Where this data design folder should be persisted.
    """

  num_reps: int
  data_set_parameters: DataSetParameters
  output_folder: str
