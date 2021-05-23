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

import numpy as np
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Type

from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.models.reach_point import (
    ReachPoint,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    PrivacyBudget,
)

TEST_POINT_STRATEGIES = {
    "latin_hypercube": lambda ds, rng: LatinHypercubeTestPointGenerator(
        ds, rng
    ).test_points(),
    "uniformly_random": lambda ds, rng: UniformlyRandomTestPointGenerator(
        ds, rng
    ).test_points(),
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

    def generate_test_points(
        self, data_set: DataSet, rng: np.random.Generator
    ) -> List[ReachPoint]:
        if not self.test_point_strategy in TEST_POINT_STRATEGIES:
            raise ValueError(
                "Invalid test point strategy: {}".format(self.test_point_strategy)
            )

        return TEST_POINT_STRATEGIES[self.test_point_strategy](data_set, rng)

    def __str__(self) -> str:
        pstrings = ["epsilon={}".format(self.privacy_budget.epsilon)]
        pstrings.append("delta={}".format(self.privacy_budget.delta))
        pstrings.append("replica_id={}".format(self.replica_id))
        pstrings.append("max_frequency={}".format(self.max_frequency))
        pstrings.append("test_point_strategy={}".format(self.test_point_strategy))
        return ":".join(pstrings)
