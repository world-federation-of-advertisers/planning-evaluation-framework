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
"""Defines the parameters for an experimental trial."""

import copy
import numpy as np
from typing import Dict
from typing import Iterable
from typing import NamedTuple
from typing import Type

from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.models.reach_point import (
    ReachPoint,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    PrivacyBudget,
)
from wfa_planning_evaluation_framework.driver.grid_test_point_generator import (
    GridTestPointGenerator,
)
from wfa_planning_evaluation_framework.driver.latin_hypercube_random_test_point_generator import (
    LatinHypercubeRandomTestPointGenerator,
)
from wfa_planning_evaluation_framework.driver.uniformly_random_test_point_generator import (
    UniformlyRandomTestPointGenerator,
)
from wfa_planning_evaluation_framework.driver.m3_subset_test_point_generator import (
    M3SubsetTestPointGenerator,
)
from wfa_planning_evaluation_framework.driver.shareshift_test_point_generator import (
    ShareShiftTestPointGenerator,
)
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    SystemParameters,
)

TEST_POINT_STRATEGIES = {
    "latin_hypercube": LatinHypercubeRandomTestPointGenerator,
    "uniformly_random": UniformlyRandomTestPointGenerator,
    "grid": GridTestPointGenerator,
    "subset": M3SubsetTestPointGenerator,
    "shareshift": ShareShiftTestPointGenerator,
}


class ExperimentParameters(NamedTuple):
    """Parameters defining an experimental trial.

    Attributes:
      privacy_budget:  The amount of privacy budget that is allocated
        to this experimental trial.
      replica_id:  The replica id of this experimental trial.
      max_frequency:  The maximum frequency that will be modeled.
      test_point_strategy: Name of the strategy to be used for
        generating test points.
      test_point_strategy_kwargs: Keyword args for test point strategy.
    """

    privacy_budget: PrivacyBudget
    replica_id: int
    max_frequency: int
    test_point_strategy: str
    test_point_strategy_kwargs: Dict = {}

    def generate_test_points(
        self, data_set: DataSet, rng: np.random.Generator
    ) -> Iterable[ReachPoint]:
        if not self.test_point_strategy in TEST_POINT_STRATEGIES:
            raise ValueError(
                "Invalid test point strategy: {}".format(self.test_point_strategy)
            )
        test_point_generator = TEST_POINT_STRATEGIES[self.test_point_strategy](
            data_set, rng, **self.test_point_strategy_kwargs
        )

        return test_point_generator.test_points()

    def _kwargs_string(self, dict: Dict) -> str:
        if not dict:
            return ""
        items = ",".join([f"{k}={v}" for k, v in dict.items()])
        return f"({items})"

    def __str__(self) -> str:
        return (
            f"epsilon={self.privacy_budget.epsilon}"
            f",delta={self.privacy_budget.delta}"
            f",replica_id={self.replica_id}"
            f",max_frequency={self.max_frequency}"
            f",test_point_strategy={self.test_point_strategy}"
            f"{self._kwargs_string(self.test_point_strategy_kwargs)}"
        )

    def update_from_dataset(
        self,
        dataset: DataSet,
        system_params: SystemParameters = None,
    ) -> "ExperimentParameters":
        """Uses the dataset to fill in various context-specific items."""
        test_point_strategy_kwargs = copy.deepcopy(self.test_point_strategy_kwargs)
        if "npublishers" in test_point_strategy_kwargs:
            test_point_strategy_kwargs["npublishers"] = dataset.publisher_count
        if "campaign_spend_fractions" in test_point_strategy_kwargs:
            test_point_strategy_kwargs[
                "campaign_spend_fractions"
            ] = system_params.campaign_spend_fractions
        return ExperimentParameters(
            copy.deepcopy(self.privacy_budget),
            copy.deepcopy(self.replica_id),
            copy.deepcopy(self.max_frequency),
            copy.deepcopy(self.test_point_strategy),
            test_point_strategy_kwargs,
        )
