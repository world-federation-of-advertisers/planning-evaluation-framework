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
"""An example of a Latin Hypercube data design."""

from pyDOE import lhs
from typing import Iterable
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
)

# The following are the parameter sets that are varied in this data design.
# The latin hypercube design constructs a subset of the cartesian product
# of these parameter settings.
NUM_PUBLISHERS = [1, 2, 5]
LARGEST_PUBLISHER = [100, 200, 300, 400, 500]
PUBLISHER_RATIOS = [.9, .5, .1]
PRICING_GENERATORS = [
    GeneratorParameters("FixedPrice", FixedPriceGenerator, {"cost_per_impression": 0.1}),
]
IMPRESSION_GENERATORS = [
    GeneratorParameters("Homogeneous", HomogeneousImpressionGenerator,
                        {"poisson_lambda": 3.0}),
    GeneratorParameters("Heterogeneous", HeterogeneousImpressionGenerator,
                        {"gamma_shape": 4.0, "gamma_scale": 0.5}),
    GeneratorParameters("HeavyTailed", HeavyTailedImpressionGenerator,
                        {'zeta_s': 2.1}),
]
OVERLAP_GENERATORS = [
    GeneratorParameters("FullOverlap", DataSet, {}),
    GeneratorParameters("Independent", IndependentOverlapDataSet,
                        {"universe_size": 50_000,
                         "random_generator": 1}),
    GeneratorParameters("Sequential", SequentiallyCorrelatedOverlapDataSet,
                        {"order": "original",
                         "correlated_sets": "all",
                         "shared_prop": 1.0,
                         "random_generator": 1}),
]

# Key values should be field names of DataSetParameters
LEVELS = {
    'num_publishers': NUM_PUBLISHERS,
    'largest_publisher_size': LARGEST_PUBLISHER,
    'largest_to_smallest_publisher_ratio': PUBLISHER_RATIOS,
    'pricing_generator_params': PRICING_GENERATORS,
    'impression_generator_params': IMPRESSION_GENERATORS,
    'overlap_generator_params': OVERLAP_GENERATORS,
}

# Number of samples that will be taken in the latin hypercube design
NUM_SAMPLES_FOR_LHS = 10

def generate_data_design_config(random_generator: np.random.Generator) -> Iterable[DataSetParameters]:
    """Generates a data design configuration.

    This examples illustrates a latin hypercube sampling strategy.
    """
    keys = LEVELS.keys()
    levels = [len(LEVELS[k]) for k in keys]
    for i, sample in enumerate(lhs(
        n=len(levels), samples=NUM_SAMPLES_FOR_LHS, criterion="maximin")):
        design_parameters = {'id': str(i)}
        for key, level in zip(keys, sample):
            design_parameters[key] = LEVELS[key][int(level * len(LEVELS[key]))]
        yield DataSetParameters(**design_parameters)
    
