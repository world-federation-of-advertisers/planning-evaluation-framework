# Copyright 2022 The Private Cardinality Estimation Framework Authors
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
from copy import deepcopy
from statsmodels.distributions.copula.elliptical import GaussianCopula, StudentTCopula


from wfa_planning_evaluation_framework.data_generators.copula_data_set import (
    CopulaDataSet,
    CopulaCorrelationMatrixGenerator,
)
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


# Number of samples that will be taken in the latin hypercube design
NUM_SAMPLES_FOR_LHS = 200

# The following are the parameter sets that are varied in this data design.
# The latin hypercube design constructs a subset of the cartesian product
# of these parameter settings.
NUM_PUBLISHERS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
LARGEST_PUBLISHER = [int(1e5), int(2e5), int(3e5)]
PUBLISHER_RATIOS = [1, 0.5, 0.3, 0.1]
PRICING_GENERATORS = [
    GeneratorParameters(
        "FixedPrice", FixedPriceGenerator, {"cost_per_impression": 0.1}
    ),
]
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
        {"gamma_shape": 1.0, "gamma_scale": 2.0},
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
    GeneratorParameters("HeavyTailed", HeavyTailedImpressionGenerator, {"zeta_s": 2.5}),
    GeneratorParameters("HeavyTailed", HeavyTailedImpressionGenerator, {"zeta_s": 5.0}),
]

RNG = np.random.default_rng(0)
OVERLAP_GENERATORS_INDEPENDENT_GIVEN_UNIVERSE_SIZE = [
    GeneratorParameters("FullOverlap", DataSet, {}),
    GeneratorParameters(
        "Independent",
        IndependentOverlapDataSet,
        {
            # Below, "largest_pub_to_universe_ratio" = largest_pub_reach / universe_size.
            # In line 58, PUBLISHER_RATIOS = smallest_pub_reach / largest_pub_reach.
            # Together with LARGEST_PUBLISHER in line 56 they determine all the sizes.
            # Note that "largest_pub_to_universe_ratio" is used only when the universe
            # size is needed.
            "largest_pub_to_universe_ratio": 0.9,
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    ),
    GeneratorParameters(
        "Independent",
        IndependentOverlapDataSet,
        {
            "largest_pub_to_universe_ratio": 0.75,
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    ),
    GeneratorParameters(
        "Independent",
        IndependentOverlapDataSet,
        {
            "largest_pub_to_universe_ratio": 0.5,
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    ),
    GeneratorParameters(
        "Independent",
        IndependentOverlapDataSet,
        {
            "largest_pub_to_universe_ratio": 0.25,
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    ),
    GeneratorParameters(
        "Sequential",
        SequentiallyCorrelatedOverlapDataSet,
        {
            "order": OrderOptions.original,
            "correlated_sets": CorrelatedSetsOptions.all,
            "shared_prop": 0.25,
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    ),
    GeneratorParameters(
        "Sequential",
        SequentiallyCorrelatedOverlapDataSet,
        {
            "order": OrderOptions.original,
            # The random and reversed orders are not supported in the current
            # evaluation framework. Can add them if needed.
            "correlated_sets": CorrelatedSetsOptions.all,
            "shared_prop": 0.75,
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    ),
    GeneratorParameters(
        "Sequential",
        SequentiallyCorrelatedOverlapDataSet,
        {
            "order": OrderOptions.original,
            "correlated_sets": CorrelatedSetsOptions.one,
            "shared_prop": 0.25,
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    ),
    GeneratorParameters(
        "Sequential",
        SequentiallyCorrelatedOverlapDataSet,
        {
            "order": OrderOptions.original,
            "correlated_sets": CorrelatedSetsOptions.one,
            "shared_prop": 0.75,
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    ),
]


# Following this design that was reviewed by WFA:
# https://docs.google.com/document/d/1pRA_fc0RbhRVUPsxbmUDmcrvNRwpNILpO-CGZtG5gYI/edit#
OVERLAP_GENERATORS_COPULA_GAUSSIAN_HOMOGENEOUS = [
    GeneratorParameters(
        "Copula",
        CopulaDataSet,
        {
            "largest_pub_to_universe_ratio": ratio,
            "copula_class": {
                "generator": GaussianCopula,
                "kwargs": {},
            },
            "correlation_matrix": {
                "generator": CopulaCorrelationMatrixGenerator.homogeneous,
                "kwargs": {"rho": rho},
            },
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    )
    for ratio in [0.25, 0.75]
    for rho in [0, 0.25, 0.5, 0.75]
]
OVERLAP_GENERATORS_COPULA_GAUSSIAN_AUTOREGRESSIVE = [
    GeneratorParameters(
        "Copula",
        CopulaDataSet,
        {
            "largest_pub_to_universe_ratio": ratio,
            "copula_class": {
                "generator": GaussianCopula,
                "kwargs": {},
            },
            "correlation_matrix": {
                "generator": CopulaCorrelationMatrixGenerator.autoregressive,
                "kwargs": {"rho": rho},
            },
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    )
    for ratio in [0.25, 0.75]
    for rho in [-0.5, -0.25, 0.25, 0.5]
]
OVERLAP_GENERATORS_COPULA_GAUSSIAN_RANDOM = [
    GeneratorParameters(
        "Copula",
        CopulaDataSet,
        {
            "largest_pub_to_universe_ratio": ratio,
            "copula_class": {
                "generator": GaussianCopula,
                "kwargs": {},
            },
            "correlation_matrix": {
                "generator": CopulaCorrelationMatrixGenerator.random,
                "kwargs": {"rng": np.random.default_rng(seed)},
            },
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    )
    for ratio in [0.25, 0.75]
    for seed in [1, 2, 3, 4]
]
OVERLAP_GENERATORS_COPULA_T_HOMOGENEOUS = [
    GeneratorParameters(
        "Copula",
        CopulaDataSet,
        {
            "largest_pub_to_universe_ratio": ratio,
            "copula_class": {
                "generator": StudentTCopula,
                "kwargs": {"df": df},  # degrees of freedom in t-copula
            },
            "correlation_matrix": {
                "generator": CopulaCorrelationMatrixGenerator.homogeneous,
                "kwargs": {"rho": 0.5},
            },
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    )
    for ratio in [0.25, 0.75]
    for df in [2, 10]
]
OVERLAP_GENERATORS_COPULA_T_AUTOREGRESSIVE = [
    GeneratorParameters(
        "Copula",
        CopulaDataSet,
        {
            "largest_pub_to_universe_ratio": ratio,
            "copula_class": {
                "generator": StudentTCopula,
                "kwargs": {"df": df},
            },
            "correlation_matrix": {
                "generator": CopulaCorrelationMatrixGenerator.autoregressive,
                "kwargs": {"rho": rho},
            },
            "random_generator": np.random.default_rng(RNG.integers(1e9)),
        },
    )
    for ratio in [0.25, 0.75]
    for df in [2, 10]
    for rho in [-0.5, 0.5]
]

OVERLAP_GENERATORS = (
    OVERLAP_GENERATORS_INDEPENDENT_GIVEN_UNIVERSE_SIZE
    + OVERLAP_GENERATORS_COPULA_GAUSSIAN_HOMOGENEOUS
    + OVERLAP_GENERATORS_COPULA_GAUSSIAN_AUTOREGRESSIVE
    + OVERLAP_GENERATORS_COPULA_GAUSSIAN_RANDOM
    + OVERLAP_GENERATORS_COPULA_T_HOMOGENEOUS
    + OVERLAP_GENERATORS_COPULA_T_AUTOREGRESSIVE
)

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
    """Generates the data design configuration for evaluating M3 strategy."""
    keys = LEVELS.keys()
    levels = [len(LEVELS[k]) for k in keys]
    ids = [16, 41, 151, 103, 69, 98, 50, 91, 73]
    for id, sample in zip(
        ids, lhs(n=len(levels), samples=NUM_SAMPLES_FOR_LHS, criterion="maximin")[ids]
    ):
        design_parameters = {"id": str(id)}
        for key, level in zip(keys, sample):
            design_parameters[key] = LEVELS[key][int(level * len(LEVELS[key]))]
        # Specify the universe size for some datasets
        if design_parameters["overlap_generator_params"].name in [
            "Independent",
            "Copula",
        ]:
            raw_overlap_params = design_parameters["overlap_generator_params"]
            kwargs = deepcopy(raw_overlap_params.params)
            kwargs["universe_size"] = int(
                design_parameters["largest_publisher_size"]
                / kwargs["largest_pub_to_universe_ratio"]
            )
            del kwargs["largest_pub_to_universe_ratio"]
            if design_parameters["overlap_generator_params"].name == "Copula":
                pricing_generator_params = design_parameters["pricing_generator_params"]
                kwargs["pricing_generator"] = pricing_generator_params.generator(
                    **pricing_generator_params.params
                )
                correlation_matrix = kwargs["correlation_matrix"]["generator"](
                    p=design_parameters["num_publishers"],
                    **kwargs["correlation_matrix"]["kwargs"]
                )
                kwargs["copula_generator"] = kwargs["copula_class"]["generator"](
                    corr=correlation_matrix, **kwargs["copula_class"]["kwargs"]
                )
                del kwargs["copula_class"]
                del kwargs["correlation_matrix"]
            design_parameters["overlap_generator_params"] = raw_overlap_params._replace(
                params=kwargs
            )
        yield DataSetParameters(**design_parameters)
