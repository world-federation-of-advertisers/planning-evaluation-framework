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
"""Data design for a quick eval to establish analysis practice."""

from pyDOE import lhs
from typing import Iterable
import itertools
import numpy as np

from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.data_set_parameters import (
    GeneratorParameters,
    DataSetParameters,
)
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)
from wfa_planning_evaluation_framework.data_generators.heavy_tailed_impression_generator import (
    HeavyTailedImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.heterogeneous_impression_generator import (
    HeterogeneousImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.homogeneous_impression_generator import (
    HomogeneousImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.independent_overlap_data_set import (
    IndependentOverlapDataSet,
)
from wfa_planning_evaluation_framework.data_generators.sequentially_correlated_overlap_data_set import (
    SequentiallyCorrelatedOverlapDataSet,
    OrderOptions,
    CorrelatedSetsOptions,
)

# The following are the parameter sets that are varied in this data design.
# The latin hypercube design constructs a subset of the cartesian product
# of these parameter settings.
NUM_PUBLISHERS = [1, 3]
LARGEST_PUBLISHER = [50, 60]
PUBLISHER_RATIOS = [1, 0.7]
PRICING_GENERATORS = [
    GeneratorParameters(
        "FixedPrice", FixedPriceGenerator, {"cost_per_impression": 0.1}
    ),
]
IMPRESSION_GENERATORS = [
    GeneratorParameters(
        "Homogeneous", HomogeneousImpressionGenerator, {"poisson_lambda": 4.0}
    ),
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 4.0, "gamma_scale": 0.5},
    ),
]

OVERLAP_GENERATORS = [
    GeneratorParameters("FullOverlap", DataSet, {}),
    GeneratorParameters(
        "Independent",
        IndependentOverlapDataSet,
        {"universe_size": 2000, "random_generator": 1},
    ),
]

# Key values should be field names of DataSetParameters
LEVELS = {
    "num_publishers": NUM_PUBLISHERS,
    "largest_publisher_size": LARGEST_PUBLISHER,
    "largest_to_smallest_publisher_ratio": PUBLISHER_RATIOS,
    "pricing_generator_params": PRICING_GENERATORS,
    "impression_generator_params": IMPRESSION_GENERATORS,
    "overlap_generator_params": OVERLAP_GENERATORS,
}


def generate_data_design_config(
    random_generator: np.random.Generator,
) -> Iterable[DataSetParameters]:
    data_design_config = []
    for level_combination in itertools.product(*LEVELS.values()):
        params = dict(zip(LEVELS.keys(), level_combination))
        data_design_config.append(DataSetParameters(**params))
    return data_design_config
