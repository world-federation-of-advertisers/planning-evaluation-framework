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
"""Encapculates the config for the test DataSet."""

from typing import List

from wfa_planning_evaluation_framework.data_generators.data_set_parameters import (
    DataSetParameters, GeneratorParameters)
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import FixedPriceGenerator
from wfa_planning_evaluation_framework.data_generators.homogeneous_impression_generator import HomogeneousImpressionGenerator
from wfa_planning_evaluation_framework.data_generators.independent_overlap_data_set import IndependentOverlapDataSet
from wfa_planning_evaluation_framework.data_generators.synthetic_data_design_config import SyntheticDataDesignConfig
from numpy.random import RandomState


class TestSyntheticDataDesignConfig(SyntheticDataDesignConfig):
  """Generates a DataDesign with synthetic data derived from parameters.

    This class
    """

  @classmethod
  def get_data_set_params_list(cls) -> List[DataSetParameters]:
    """Generates data set parameters to create a data set from.

    Returns:
       DataSetParameters object. This object can be hard coded or can be
       constructed through some business logic.
    """

    return [cls.get_data_set_params(i) for i in range(3)]

  @classmethod
  def get_data_set_params(cls, seed: int):
    return DataSetParameters(
        num_publishers=3,
        largest_publisher_size=100,
        relative_reach_of_largest_publisher=0.5,
        publisher_size_decay_rate=0.9,
        pricing_generator_params=GeneratorParameters(
            generator=FixedPriceGenerator, params={"cost_per_impression": 0.1}),
        impression_generator_params=GeneratorParameters(
            generator=HomogeneousImpressionGenerator,
            params={
                "poisson_lambda": 0.1,
                "random_state": RandomState(seed)
            }),
        overlap_generator_params=GeneratorParameters(
            generator=IndependentOverlapDataSet,
            params={
                "random_state": RandomState(seed),
            }),
        name="independent_homog_p=10_numpub=3_rs=" + str(seed))
