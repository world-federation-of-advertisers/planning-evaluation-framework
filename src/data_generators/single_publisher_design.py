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

PUBLISHER_SIZES = [10_000 * i for i in [1, 2, 3, 4, 5]]

# We consider four basic data generating processes:
#   Poisson, Exponential-Poisson, Gamma-Poisson and Zeta.
# For each of these processes, we consider distributions
# having mean 1.5, 3, 5 and 10.

IMPRESSION_GENERATORS = [
    ## Mean 1.5
    # Shifted Poisson: Mean = 1.5, Var = 0.5
    # Note: For a Poisson distribution the variance and the mean are both
    # equal to the parameter lambda.  For a shifted Poisson, the
    # mean is lambda + 1.
    GeneratorParameters(
        "Homogeneous", HomogeneousImpressionGenerator, {"poisson_lambda": 0.5}
    ),

    # Exponential Poisson: Mean = 1.5, Var = 0.75
    # For the Exponential-Poisson, the mean is beta + 1 and the variance is
    # beta * (beta + 1), where beta is the "gamma_scale" parameter.  This can
    # be worked out from the formulas given below for the Gamma-Poisson by
    # taking alpha=1.
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 1.0, "gamma_scale": 0.5},
    ),

    # Gamma Poisson: Mean = 1.5, Var = 6
    # For the shifted Gamma-Poisson, the mean is alpha * beta + 1, and the
    # variance is alpha * beta * (beta + 1), where alpha = gamma_shape and
    # beta = gamma_scale.  This can be worked out by making
    # use of the equivalence between the Gamma-Poisson and the negative
    # binomial distribution.  Using the formulation for the negative binomial
    # given in Wikipedia, the equivalent negative binomial distribution is
    # obtained by setting p = beta / (1 + beta) and r = alpha.  
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 0.04545, "gamma_scale": 11},
    ),
    
    # Zeta: Mean = 1.5, Var = infinity
    GeneratorParameters(
        "HeavyTailed", HeavyTailedImpressionGenerator, {"zeta_s": 2.8106}
    ),
    
    ## Mean 3
    # Shifted Poisson: Mean = 3, Var = 2
    GeneratorParameters(
        "Homogeneous", HomogeneousImpressionGenerator, {"poisson_lambda": 2.0}
    ),
    # Exponential Poisson: Mean = 3, Var = 6
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 1.0, "gamma_scale": 2.0},
    ),
    # Gamma Poisson: Mean = 3, Var = 12
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 0.4, "gamma_scale": 5.0},
    ),
    # Zeta: Mean = 3, Var = infinity
    GeneratorParameters(
        "HeavyTailed", HeavyTailedImpressionGenerator, {"zeta_s": 2.2662}
    ),
    
    ## Mean 5
    # Shifted Poisson: Mean = 5, Var = 4
    GeneratorParameters(
        "Homogeneous", HomogeneousImpressionGenerator, {"poisson_lambda": 4.0}
    ),
    # Exponential Poisson: Mean = 5, Var = 20
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 1.0, "gamma_scale": 4.0},
    ),
    # Gamma Poisson: Mean = 5, Var = 40
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 0.44444, "gamma_scale": 9.0},
    ),
    # Zeta: Mean = 5, Var = infinity
    GeneratorParameters(
        "HeavyTailed", HeavyTailedImpressionGenerator, {"zeta_s": 2.1416}
    ),

    ## Mean 10
    # Shifted Poisson: Mean = 10, Var = 9
    GeneratorParameters(
        "Homogeneous", HomogeneousImpressionGenerator, {"poisson_lambda": 9.0}
    ),
    # Exponential Poisson: Mean = 10, Var = 90
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 1.0, "gamma_scale": 9.0},
    ),
    # Gamma Poisson: Mean = 10, Var = 200
    GeneratorParameters(
        "Heterogeneous",
        HeterogeneousImpressionGenerator,
        {"gamma_shape": 0.42408, "gamma_scale": 21.2222},
    ),
    # Zeta: Mean = 10, Var = infinity
    GeneratorParameters(
        "HeavyTailed", HeavyTailedImpressionGenerator, {"zeta_s": 2.06539}
    ),
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
