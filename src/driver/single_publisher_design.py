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
"""Experimental design for single publisher models."""

from absl import app
from absl import flags
import itertools
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
    ("goerg", {}),
    ("gamma_poisson", {}),
    ("gamma_poisson", {"extrapolation_multiplier": 2.0}),
    ("kinflated_gamma_poisson", {}),
    ("kinflated_gamma_poisson", {"extrapolation_multiplier": 2.0}),
]

CAMPAIGN_SPEND_FRACTIONS = list(np.arange(1, 10) * 0.1)
LIQUID_LEGIONS_DECAY_RATES = [10]
LIQUID_LEGIONS_SKETCH_SIZES = [100_000]
PRIVACY_BUDGET_EPSILONS = [0.05, 0.10, 0.15, 0.20, 100.0]
PRIVACY_BUDGET_DELTAS = [1e-9]
REPLICA_IDS = [1]
TEST_POINTS = [20]
MAX_FREQUENCIES = [10]

LEVELS = {
    "modeling_strategy": MODELING_STRATEGIES,
    "campaign_spend_fraction": CAMPAIGN_SPEND_FRACTIONS,
    "liquid_legions_decay_rate": LIQUID_LEGIONS_DECAY_RATES,
    "liquid_legions_sketch_size": LIQUID_LEGIONS_SKETCH_SIZES,
    "privacy_budget_epsilon": PRIVACY_BUDGET_EPSILONS,
    "privacy_budget_delta": PRIVACY_BUDGET_DELTAS,
    "replica_id": REPLICA_IDS,
    "test_points": TEST_POINTS,
    "max_frequency": MAX_FREQUENCIES,
}


def generate_experimental_design_config(seed: int = 1) -> Iterable[TrialDescriptor]:
    """Generates a list of TrialDescriptors for a single publisher model."""
    keys = list(LEVELS.keys())
    levels = [LEVELS[k] for k in keys]
    for sample in itertools.product(*levels):
        design_parameters = dict(zip(keys, sample))
        mstrategy = ModelingStrategyDescriptor(
            "single_publisher",
            {},
            design_parameters["modeling_strategy"][0],
            design_parameters["modeling_strategy"][1],
            "none",
            {},
        )
        sparams = SystemParameters(
            [design_parameters["campaign_spend_fraction"]],
            LiquidLegionsParameters(
                design_parameters["liquid_legions_decay_rate"],
                design_parameters["liquid_legions_sketch_size"],
            ),
            np.random.default_rng(seed=seed),
        )
        eparams = ExperimentParameters(
            PrivacyBudget(
                design_parameters["privacy_budget_epsilon"],
                design_parameters["privacy_budget_delta"],
            ),
            design_parameters["replica_id"],
            design_parameters["max_frequency"],
            "grid",
            {"grid_size": design_parameters["test_points"]},
        )
        yield TrialDescriptor(mstrategy, sparams, eparams)
