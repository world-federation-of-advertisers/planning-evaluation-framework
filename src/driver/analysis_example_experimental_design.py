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
"""Experimental design for a quick eval to establish analysis practice."""

import itertools
import math
import numpy as np
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
        "m3strategy", {}, "goerg", {}, "restricted_pairwise_union", {}
    ),
    ModelingStrategyDescriptor(
        "m3strategy", {}, "gamma_poisson", {}, "restricted_pairwise_union", {}
    ),
]

CAMPAIGN_SPEND_FRACTIONS_GENERATORS = [
    lambda dataset: [0.6] * dataset.publisher_count,
    lambda dataset: list(islice(cycle([0.4, 0.8]), dataset.publisher_count)),
]

LIQUID_LEGIONS_PARAMS = [
    LiquidLegionsParameters(10, 8000),
]

PRIVACY_BUDGETS = [
    PrivacyBudget(2, 1e-9),
    PrivacyBudget(0.5, 1e-9),
]

REPLICA_IDS = [1, 2, 3]

MAX_FREQUENCIES = [3, 6]

TEST_POINT_STRATEGIES = [
    ("latin_hypercube", {"npublishers": 1, "minimum_points_per_publisher": 10}),
    ("uniformly_random", {"npublishers": 1, "minimum_points_per_publisher": 10}),
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
# Will evaluate all level combinations


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
