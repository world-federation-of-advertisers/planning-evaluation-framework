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
from itertools import cycle, islice

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
        "m3strategy",
        {"use_ground_truth_for_reach_curves": True},
        "goerg",
        {},
        "restricted_pairwise_union",
        {},
    ),
    ModelingStrategyDescriptor(
        "m3strategy", {}, "goerg", {}, "restricted_pairwise_union", {}
    ),
    ModelingStrategyDescriptor(
        "m3strategy", {}, "gamma_poisson", {}, "restricted_pairwise_union", {}
    ),
]

CAMPAIGN_SPEND_FRACTIONS_GENERATORS = [
    lambda dataset: [0.2] * dataset.publisher_count,
    lambda dataset: list(islice(cycle([0.1, 0.2, 0.3]), dataset.publisher_count)),
]

LIQUID_LEGIONS_PARAMS = [
    LiquidLegionsParameters(12, 1e5),
]

PRIVACY_BUDGETS = [
    PrivacyBudget(1.0, 1e-7),
    PrivacyBudget(1.0, 1e-9),
    PrivacyBudget(0.1, 1e-7),
    PrivacyBudget(0.1, 1e-9),
]

REPLICA_IDS = [1, 2, 3]

MAX_FREQUENCIES = [5, 20]

TEST_POINT_STRATEGIES = [
    ("latin_hypercube", {"npublishers": 1}),
    ("uniformly_random", {"npublishers": 1}),
]

LEVELS = {
    "modeling_strategies": MODELING_STRATEGIES,
    "campaign_spend_fractions_generators": CAMPAIGN_SPEND_FRACTIONS_GENERATORS,
    "liquid_legions_params": LIQUID_LEGIONS_PARAMS,
    "privacy_budgets": PRIVACY_BUDGETS,
    "replica_ids": REPLICA_IDS,
    "max_frequencies": MAX_FREQUENCIES,
    "test_point_strategies": TEST_POINT_STRATEGIES,
}
# A total of 2 * 2 * 1 * 4 * 3 * 2 * 2 = 192 designs. Will evaluate all of them
# per dataset.


def generate_experimental_design_config(seed: int = 1) -> Iterable[TrialDescriptor]:
    """Generates a list of TrialDescriptors for the 1st round eval of M3."""
    for level_combination in itertools.product(*LEVELS.values()):
        design_parameters = dict(zip(LEVELS.keys(), level_combination))
        mstrategy = design_parameters["modeling_strategies"]
        sparams = SystemParameters(
            liquid_legions=design_parameters["liquid_legions_params"],
            generator=np.random.default_rng(seed=seed),
            campaign_spend_fractions_generator=design_parameters[
                "campaign_spend_fractions_generators"
            ],
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
