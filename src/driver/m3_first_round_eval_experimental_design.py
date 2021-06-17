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
"""Experimental design for the first round evaluation of M3 strategy.

At the first round, the proposal M3 strategy will be evaluated against 4 sets of
ExperimentParameters and 1 set of SystemParameters (and 100 DataSets). Later
after implementing parallel evaluation, we plan to evaluate the M3 strategy
against a large experimental design (and also data design). So there will be
m3_second_round_eval_experimental_design.py, etc.
"""

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
        "m3proposal", {}, "gamma_poisson", {}, "restricted_pairwise_union", {}
    ),
]

# Here we assume all data sets specify exactly two campaigns
CAMPAIGN_SPEND_FRACTIONS = list(
    itertools.product(np.arange(1, 10) * 0.1, np.arange(1, 10) * 0.1)
)

LIQUID_LEGIONS_PARAMS = [
    LiquidLegionsParameters(12, 1e5),
]

PRIVACY_BUDGETS = [
    PrivacyBudget(1.0, 1e-7),
    PrivacyBudget(1.0, 1e-9),
    PrivacyBudget(0.1, 1e-7),
    PrivacyBudget(0.1, 1e-9),
]

REPLICA_IDS = [1, 2, 3, 4, 5]

MAX_FREQUENCIES = [5, 20]

TEST_POINT_STRATEGIES = [
    ("latin_hypercube", {"npoints": 100}),
    ("uniformly_random", {"npoints": 500}),
    ("grid", {"grid_size": 5}),
]

LEVELS = {
    "modeling_strategies": MODELING_STRATEGIES,
    "campaign_spend_fractions": CAMPAIGN_SPEND_FRACTIONS,
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
            design_parameters["campaign_spend_fractions"],
            design_parameters["liquid_legions_params"],
            random_generator,
        )
        test_point_generator, test_point_params = design_parameters[
            "test_point_strategies"
        ]
        eparams = ExperimentParameters(
            design_parameters["privacy_budgets"],
            design_parameters["replica_ids"],
            design_parameters["max_frequencies"],
            test_point_generator,
            test_point_params,
        )
        yield TrialDescriptor(mstrategy, sparams, eparams)
