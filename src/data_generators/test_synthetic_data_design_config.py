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
"""Encapculates the config for the test DataDesign."""

from typing import List
from numpy.random import RandomState
from wfa_planning_evaluation_framework.data_generators.data_set_parameters import (
    DataSetParameters, GeneratorParameters)
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import FixedPriceGenerator
from wfa_planning_evaluation_framework.data_generators.homogeneous_impression_generator import HomogeneousImpressionGenerator
from wfa_planning_evaluation_framework.data_generators.independent_overlap_data_set import IndependentOverlapDataSet
from wfa_planning_evaluation_framework.data_generators.synthetic_data_design_config import SyntheticDataDesignConfig


class TestSyntheticDataDesignConfig(SyntheticDataDesignConfig):
  """Generates an example DataDesign with synthetic data."""

  @classmethod
  def get_data_set_params_list(cls, seed: int) -> List[DataSetParameters]:
    random_state = RandomState(seed=seed)
    return [cls.get_data_set_params(random_state) for i in range(3)]

  @classmethod
  def get_data_set_params(cls, random_state: RandomState):
    largest_publisher_size = 1000
    # This signature is same for all runs with the same seed. It is
    # deterministic, the same value will be generated for the same seed and the
    # same DataSetParameters that construct the same underliying objects.
    random_signature = str(random_state.randint(100000, size=1)[0])
    return DataSetParameters(
        num_publishers=3,
        largest_publisher_size=largest_publisher_size,
        largest_to_smallest_publisher_ratio=0.5,
        pricing_generator_params=GeneratorParameters(
            generator=FixedPriceGenerator, params={"cost_per_impression": 0.1}),
        impression_generator_params=GeneratorParameters(
            generator=HomogeneousImpressionGenerator,
            params={
                "poisson_lambda": 0.1,
                "random_state": random_state
            }),
        overlap_generator_params=GeneratorParameters(
            generator=IndependentOverlapDataSet,
            params={
                "random_state": random_state,
                "universe_size": largest_publisher_size * 10
            }),
        name="independent_homog_p=10_numpub=3_rs=" + random_signature)
