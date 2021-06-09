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
from wfa_planning_evaluation_framework.data_generators.synthetic_data_design_config import (
    SyntheticDataDesignConfig,
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


class LHSSyntheticDataDesignConfig(SyntheticDataDesignConfig):
    """Generates a DataDesign using LHS from a synthetic grid."""

    @classmethod
    def get_lhs_design(cls, variable_grid_axes: Dict[str, List[float]]):
        pass

    @classmethod
    def get_grid_axes(cls) -> Dict[str, List[float]]:
        pass

    @classmethod
    def get_params_dict(cls, lhs_trial: List[int], grid_axes: OrderedDict):
        keys = list(grid_axes.keys())
        values = list(grid_axes.values())
        params_dict = {}
        for param in range(len(lhs_trial)):
            params_dict[keys[param]] = values[param][lhs_trial[param]]
        return params_dict

    @classmethod
    def get_data_set_params_list(
        cls, random_generator: Generator
    ) -> List[DataSetParameters]:
        grid_axes = cls.get_grid_axes()
        lhs_design = cls.get_lhs_design(grid_axes)

        return [
            cls.get_data_set_params(
                cls.get_params_dict(lhs_trial, grid_axes), random_generator
            )
            for lhs_trial in lhs_design
            for i in range(NUM_RANDOM_REPLICAS)
        ]

    @classmethod
    def get_data_set_params(
        cls, params_dict, random_generator: Generator
    ) -> DataSetParameters:
        for gen_param in RANDOMIZATION_NEEDED_GEN_PARAMS:
            params_dict[gen_param].params["random_generator"] = random_generator
        params_dict[OVERLAP_GEN_PARAMS].params["universe_size"] = (
            params_dict[LARGEST_PUB_SIZE] * 10
        )
        return DataSetParameters(
            num_publishers=params_dict[NUM_PUBS],
            largest_publisher_size=params_dict[LARGEST_PUB_SIZE],
            largest_to_smallest_publisher_ratio=params_dict[PUB_RATIO],
            pricing_generator_params=params_dict[PRICE_GEN_PARAMS],
            impression_generator_params=params_dict[IMG_GEN_PARAMS],
            overlap_generator_params=params_dict[OVERLAP_GEN_PARAMS],
        )
