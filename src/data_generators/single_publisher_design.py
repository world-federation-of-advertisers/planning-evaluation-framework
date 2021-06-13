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
"""Data design for single publisher models."""

from typing import Iterable
import itertools
import numpy as np
from numpy.random import Generator
from wfa_planning_evaluation_framework.data_generators.data_set_parameters import (
    DataSetParameters,
    GeneratorParameters,
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

PUBLISHER_SIZES = [1000 * 2 ** k for k in range(8)]

IMPRESSION_GENERATORS = [
    GeneratorParameters(
        "Homogeneous", HomogeneousImpressionGenerator, {"poisson_lambda": 2.0}
    ),
    GeneratorParameters(
        "Homogeneous", HomogeneousImpressionGenerator, {"poisson_lambda": 5.0}
    ),
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 4.0, "gamma_scale": 0.5},
    ),
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 1.0, "gamma_scale": 2},
    ),
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 4.0, "gamma_scale": 1.0},
    ),
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 1.0, "gamma_scale": 4.0},
    ),
    GeneratorParameters("HeavyTailed", HeavyTailedImpressionGenerator, {"zeta_s": 2.1}),
]


def generate_data_design_config(
    random_generator: np.random.Generator,
) -> Iterable[DataSetParameters]:
    """Generates a data design configuration for single publisher models."""

    data_design_config = []
    for params in itertools.product(PUBLISHER_SIZES, IMPRESSION_GENERATORS):
        publisher_size, impression_generator = params
        data_design_config.append(
            DataSetParameters(
                largest_publisher_size=publisher_size,
                impression_generator_params=impression_generator,
            )
        )
    return data_design_config
