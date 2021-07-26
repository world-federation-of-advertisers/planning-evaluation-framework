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
"""Defines the parameters for generating a DataSet."""

from typing import NamedTuple
from typing import Type
from typing import Any

from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)
from wfa_planning_evaluation_framework.data_generators.homogeneous_impression_generator import (
    HomogeneousImpressionGenerator,
)


class GeneratorParameters(NamedTuple):
    """Parameters to create one Generator.

    name: String, Name of the generator (e.g., 'Homogeneous')
    generator: Class of the generator (e.g. HomogeneousImpressionGenerator)
    params : Parameters dict used to initialize that class.
    """

    #  TODO(uakyol): create a Generator parent class for impression, pricing and
    #  overlap generators and change this type.
    name: str
    generator: Type[Any]
    params: dict

    def __str__(self):
        if self.params:
            param_str = ",".join([f"{k}={v}" for k, v in self.params.items()])
            return f"{self.name}({param_str})"
        else:
            return self.name


DEFAULT_NUM_PUBLISHERS = 1
DEFAULT_LARGEST_PUBLISHER_SIZE = 10000
DEFAULT_LARGEST_TO_SMALLEST_PUBLISHER_RATIO = 0.9
DEFAULT_PRICING_GENERATOR_PARAMS = GeneratorParameters(
    "Fixed", FixedPriceGenerator, params={"cost_per_impression": 0.1}
)
DEFAULT_IMPRESSION_GENERATOR_PARAMS = GeneratorParameters(
    "Homogeneous", HomogeneousImpressionGenerator, params={"poisson_lambda": 3.0}
)
DEFAULT_OVERLAP_GENERATOR_PARAMS = GeneratorParameters(
    "FullOverlap", DataSet, params={}
)


class DataSetParameters(NamedTuple):
    """Parameters to create one DataSet.

    Does not include randomness

      name: Str, A string describing this parameter configuration.
      num_publishers: Int, Number of publishers in this DataSet.
      largest_publisher_size: Float, Maximum possible reach of the largest publisher.
      largest_to_smallest_publisher_ratio: Float, Ratio of the size of largest publisher
        to smallest size.
      pricing_generator_params: Parameters for the pricing generator.
      impression_generator_params: Parameters for the impression generator.
      overlap_generator_params: Parameters for the overlap generator.
      id: A string that makes this DataSet name unique, if needed
    """

    name: str = ""
    num_publishers: int = DEFAULT_NUM_PUBLISHERS
    largest_publisher_size: int = DEFAULT_LARGEST_PUBLISHER_SIZE
    largest_to_smallest_publisher_ratio: float = (
        DEFAULT_LARGEST_TO_SMALLEST_PUBLISHER_RATIO
    )
    pricing_generator_params: GeneratorParameters = DEFAULT_PRICING_GENERATOR_PARAMS
    impression_generator_params: GeneratorParameters = (
        DEFAULT_IMPRESSION_GENERATOR_PARAMS
    )
    overlap_generator_params: GeneratorParameters = DEFAULT_OVERLAP_GENERATOR_PARAMS
    id: str = ""

    def __str__(self):
        if self.name:
            return self.name
        else:
            return (
                f"P={self.num_publishers},"
                f"N={self.largest_publisher_size},"
                f"pub_ratio={self.largest_to_smallest_publisher_ratio},"
                f"pricing={str(self.pricing_generator_params)},"
                f"impressions={str(self.impression_generator_params)},"
                f"overlaps={str(self.overlap_generator_params)}"
                f"{',id='+self.id if self.id else ''}"
            )
