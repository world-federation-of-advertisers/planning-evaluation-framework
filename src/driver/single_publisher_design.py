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

# MODELING_STRATEGIES = ["goerg", "gamma_poisson"]
MODELING_STRATEGIES = ["goerg"]
CAMPAIGN_SPEND_FRACTIONS = list(np.arange(1,20) * 0.05)
LIQUID_LEGIONS_DECAY_RATES = [10, 12, 15]
LIQUID_LEGIONS_SKETCH_SIZES = [50_000, 100_000]
PRIVACY_BUDGET_EPSILONS = [np.log(3), 0.1 * np.log(3)]
PRIVACY_BUDGET_DELTAS = [1e-5, 1e-9]
REPLICA_IDS = [1]
TEST_POINTS = [20]
MAX_FREQUENCIES = [5, 10, 20]

LEVELS = {
    'modeling_strategy': MODELING_STRATEGIES,
    'campaign_spend_fraction': CAMPAIGN_SPEND_FRACTIONS,
    'liquid_legions_decay_rate': LIQUID_LEGIONS_DECAY_RATES,
    'liquid_legions_sketch_size': LIQUID_LEGIONS_SKETCH_SIZES,
    'privacy_budget_epsilon': PRIVACY_BUDGET_EPSILONS,
    'privacy_budget_delta': PRIVACY_BUDGET_DELTAS,
    'replica_id': REPLICA_IDS,
    'test_points': TEST_POINTS,
    'max_frequency': MAX_FREQUENCIES,
}

# Number of experimental trials that should be conducted per dataset
NUM_TRIALS_PER_DATASET = 100

def generate_experimental_design_config(
    random_generator: np.random.Generator,
) -> Iterable[TrialDescriptor]:
    """Generates a list of TrialDescriptors for a single publisher model."""
    keys = LEVELS.keys()
    levels = [len(LEVELS[k]) for k in keys]
    for sample in (
        lhs(n=len(levels), samples=NUM_TRIALS_PER_DATASET, criterion="maximin")
    ):
        design_parameters = {}
        for key, level in zip(keys, sample):
            design_parameters[key] = LEVELS[key][int(level * len(LEVELS[key]))]
        mstrategy = ModelingStrategyDescriptor(
            "single_publisher", {}, design_parameters['modeling_strategy'], {},
            "none", {})
        sparams = SystemParameters(
            [design_parameters['campaign_spend_fraction']],
            LiquidLegionsParameters(
                design_parameters['liquid_legions_decay_rate'],
                design_parameters['liquid_legions_sketch_size']
            ),
            random_generator)
        eparams = ExperimentParameters(
            PrivacyBudget(
                design_parameters['privacy_budget_epsilon'],
                design_parameters['privacy_budget_delta']
            ),
            design_parameters['replica_id'],
            design_parameters['max_frequency'],
            'grid',
            {'grid_size': design_parameters['test_points']},
        )
        yield TrialDescriptor(mstrategy, sparams, eparams)
