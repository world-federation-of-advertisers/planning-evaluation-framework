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
"""Encapculates the config for a sample LHS DataDesign."""

from typing import List
from typing import Dict
from collections import OrderedDict
import numpy as np
from pyDOE import lhs
from numpy.random import Generator
from wfa_planning_evaluation_framework.data_generators.data_set_parameters import (
    DataSetParameters,
    GeneratorParameters,
)
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)
from wfa_planning_evaluation_framework.data_generators.homogeneous_impression_generator import (
    HomogeneousImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.independent_overlap_data_set import (
    IndependentOverlapDataSet,
)
from wfa_planning_evaluation_framework.data_generators.lhs_synthetic_data_design_config import (
    LHSSyntheticDataDesignConfig,
)

NUM_PUBS = "num_publishers"
LARGEST_PUB_SIZE = "largest_publisher_size"
PUB_RATIO = "largest_to_smallest_publisher_ratio"
IMG_GEN_PARAMS = "impression_generator_params"
PRICE_GEN_PARAMS = "pricing_generator_params"
OVERLAP_GEN_PARAMS = "overlap_generator_params"

RANDOMIZATION_NEEDED_GEN_PARAMS = [IMG_GEN_PARAMS, OVERLAP_GEN_PARAMS]

NUM_SAMPLES_FOR_LHS = 3
NUM_RANDOM_REPLICAS = 2


class TestRoundedLHSSyntheticDataDesignConfig(LHSSyntheticDataDesignConfig):
    """Generates a DataDesign using LHS from a synthetic grid of integers."""

    @classmethod
    def get_lhs_design(cls, variable_grid_axes: Dict[str, List[float]]):
        return cls.get_rounded_lhs_design(variable_grid_axes)

    @classmethod
    def get_grid_axes(cls) -> Dict[str, List[int]]:
        grid_axes = OrderedDict()
        grid_axes[NUM_PUBS] = [1, 2, 5]
        grid_axes[LARGEST_PUB_SIZE] = [100, 1000]
        grid_axes[PUB_RATIO] = [1, 0.5, 0.3, 0.1]
        grid_axes[IMG_GEN_PARAMS] = [
            GeneratorParameters(
                generator=HomogeneousImpressionGenerator, params={"poisson_lambda": 2}
            ),
            GeneratorParameters(
                generator=HomogeneousImpressionGenerator, params={"poisson_lambda": 5}
            ),
        ]
        grid_axes[PRICE_GEN_PARAMS] = [
            GeneratorParameters(
                generator=FixedPriceGenerator, params={"cost_per_impression": 0.1}
            )
        ]
        grid_axes[OVERLAP_GEN_PARAMS] = [
            GeneratorParameters(generator=IndependentOverlapDataSet, params={})
        ]
        return grid_axes
