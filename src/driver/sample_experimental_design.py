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
"""A sample experimental design."""

from absl import app
from absl import flags
import itertools
import math
import numpy as np
from pyDOE import lhs
from typing import Iterable

from wfa_planning_evaluation_framework.driver.experiment_parameters import (
    ExperimentParameters,
)
from wfa_planning_evaluation_framework.driver.modeling_strategy_descriptor import (
    ModelingStrategyDescriptor,
)
from wfa_planning_evaluation_framework.simulator.modeling_strategy import (
    ModelingStrategy,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    PrivacyBudget,
)
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    LiquidLegionsParameters,
    SystemParameters,
)
from wfa_planning_evaluation_framework.driver.trial_descriptor import (
    TrialDescriptor,
)

MODELING_STRATEGIES = [
    ModelingStrategyDescriptor(
        "m3proposal", {}, "goerg", {}, "restricted_pairwise_union", {}
    ),
    ModelingStrategyDescriptor(
        "m3proposal", {}, "gamma_poisson", {}, "restricted_pairwise_union", {}
    ),
]

# Here we assume all data sets specify exactly two campaigns
# of up to $1M each.
CAMPAIGN_SPENDS = list(
    itertools.product(np.arange(1, 10) * 100_000, np.arange(1, 10) * 100_000)
)

LIQUID_LEGIONS_PARAMS = [
    LiquidLegionsParameters(10, 1e5),
    LiquidLegionsParameters(12, 1e5),
    LiquidLegionsParameters(17, 1e5),
    LiquidLegionsParameters(10, 2e5),
    LiquidLegionsParameters(12, 2e5),
    LiquidLegionsParameters(17, 2e5),
]

PRIVACY_BUDGETS = [
    PrivacyBudget(1.0, 1e-7),
    PrivacyBudget(1.0, 1e-8),
    PrivacyBudget(0.1, 1e-7),
    PrivacyBudget(0.1, 1e-8),
]

REPLICA_IDS = [1, 2, 3]

MAX_FREQUENCIES = [5, 10, 20]

TEST_POINT_STRATEGIES = ["latin_hypercube", "uniformly_random"]

LEVELS = {
    "modeling_strategies": MODELING_STRATEGIES,
    "campaign_spends": CAMPAIGN_SPENDS,
    "liquid_legions_params": LIQUID_LEGIONS_PARAMS,
    "privacy_budgets": PRIVACY_BUDGETS,
    "replica_ids": REPLICA_IDS,
    "max_frequencies": MAX_FREQUENCIES,
    "test_point_strategies": TEST_POINT_STRATEGIES,
}

# Number of experimental trials that should be conducted per dataset
NUM_TRIALS_PER_DATASET = 100


def generate_experimental_design_config(
    random_generator: np.random.Generator,
) -> Iterable[TrialDescriptor]:
    """Generates a list of TrialDescriptors.

    This examples illustrates a latin hypercube sampling strategy.
    """
    keys = LEVELS.keys()
    levels = [len(LEVELS[k]) for k in keys]
    for i, sample in enumerate(
        lhs(n=len(levels), samples=NUM_TRIALS_PER_DATASET, criterion="maximin")
    ):
        design_parameters = {}
        for key, level in zip(keys, sample):
            design_parameters[key] = LEVELS[key][int(level * len(LEVELS[key]))]
        mstrategy = design_parameters["modeling_strategies"]
        sparams = SystemParameters(
            design_parameters["campaign_spends"],
            design_parameters["liquid_legions_params"],
            random_generator,
        )
        eparams = ExperimentParameters(
            design_parameters["privacy_budgets"],
            design_parameters["replica_ids"],
            design_parameters["max_frequencies"],
            design_parameters["test_point_strategies"],
        )
        yield TrialDescriptor(mstrategy, sparams, eparams)
