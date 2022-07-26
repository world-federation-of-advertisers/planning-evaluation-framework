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
    # Dirac mixture multi pub
    ModelingStrategyDescriptor(
        "m3strategy",
        {"use_ground_truth_for_reach_curves": True},
        "dirac_mixture_single",
        {},
        "dirac_mixture_multi",
        {
            "dilution": 0.3,
            "largest_pub_to_universe_ratio": 0.25,
            "single_publisher_reach_agreement": False,
        },
    ),
    ModelingStrategyDescriptor(
        "m3strategy",
        {"use_ground_truth_for_reach_curves": False},
        "dirac_mixture_single",
        {"dilution": 0.3, "largest_pub_to_universe_ratio": 0.25},
        "dirac_mixture_multi",
        {
            "dilution": 0.3,
            "largest_pub_to_universe_ratio": 0.25,
            "single_publisher_reach_agreement": False,
        },
    ),
    # Independent multi pub
    # ModelingStrategyDescriptor(
    #     "m3strategy",
    #     {"use_ground_truth_for_reach_curves": True},
    #     "dirac_mixture_single",
    #     {},
    #     "independent",
    #     {"largest_pub_to_universe_ratio": 0.25},
    # ),
    # ModelingStrategyDescriptor(
    #     "m3strategy",
    #     {"use_ground_truth_for_reach_curves": False},
    #     "dirac_mixture_single",
    #     {"dilution": 0.3, "largest_pub_to_universe_ratio": 0.25},
    #     "independent",
    #     {"largest_pub_to_universe_ratio": 0.25},
    # ),
]

CAMPAIGN_SPEND_FRACTIONS_GENERATORS = [
    lambda dataset: [0.1] * dataset.publisher_count,
    lambda dataset: [0.5] * dataset.publisher_count,
    lambda dataset: [0.9] * dataset.publisher_count,
    lambda dataset: list(islice(cycle([0.1, 0.9]), dataset.publisher_count)),
]

LIQUID_LEGIONS_PARAMS = [
    LiquidLegionsParameters(7.75, 1e5),
]

PRIVACY_BUDGETS = [
    PrivacyBudget(1.0, 1e-9),
    # PrivacyBudget(0.1, 1e-9),
    # PrivacyBudget(0.01, 1e-9),
]

REPLICA_IDS = [1, 2, 3]

MAX_FREQUENCIES = [10]

TEST_POINT_STRATEGIES = [
    ("latin_hypercube", {"npublishers": 1}),
    ("subset", {"campaign_spend_fractions": [1]}),
    ("shareshift", {"campaign_spend_fractions": [1]}),
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
# A total of 4 * 4 * 3 * 3 * 3 = 432 configs. Will evaluate all of them
# per dataset.


def generate_experimental_design_config(seed: int = 1) -> Iterable[TrialDescriptor]:
    """Generates a list of TrialDescriptors for the 1st round eval of M3."""
    for id, level_combination in enumerate(itertools.product(*LEVELS.values())):
        print("\n", id, "\n", level_combination)

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
        yield TrialDescriptor(mstrategy, sparams, eparams, id)