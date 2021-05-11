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
from typing import Type
from typing import Any


class GeneratorParameters(NamedTuple):
  """Parameters to create one Generator.

    generator: Class of the generator (e.g. HomogeneousImpressionGenerator)
    params : Parameters dict used to initialize that class.
    """
  #  TODO(uakyol): create a Generator parent class for impression, pricing and
  #  overlap generators and change this type.
  generator: Type[Any]
  params: dict


class DataSetParameters(NamedTuple):
  """Parameters to create one DataSet.

  Does not include randomness

    num_publishers: Number of publishers in this DataSet.
    largest_publisher_size: Maximum possible reach of the largest publisher.
    largest_to_smallest_publisher_ratio: Ratio of the size of largest publisher
      to smallest size.
    pricing_generator_params: Parameters for the pricing generator.
    impression_generator_params: Parameters for the impression generator.
    overlap_generator_params: Parameters for the overlap generator.
    name: Filename of the data set to be generated when written to disk.
    """

  num_publishers: int
  largest_publisher_size: int
  largest_to_smallest_publisher_ratio : float
  pricing_generator_params: GeneratorParameters
  impression_generator_params: GeneratorParameters
  overlap_generator_params: GeneratorParameters
  name: str
