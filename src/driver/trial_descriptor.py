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
"""Defines the parameters for one experimental trial."""

import copy
from numpy.random import Generator
from typing import Dict
from typing import NamedTuple
from typing import Type

from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.driver.experiment_parameters import (
    ExperimentParameters,
)
from wfa_planning_evaluation_framework.driver.modeling_strategy_descriptor import (
    ModelingStrategyDescriptor,
)
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    SystemParameters,
)


class TrialDescriptor(NamedTuple):
    """Parameters defining a single experimental trial.

    Attributes:
      modeling_strategy: A ModelingStrategyDescriptor
      system_params: System parameters
      experiment_params: Experiment parameters
    """

    modeling_strategy: ModelingStrategyDescriptor
    system_params: SystemParameters
    experiment_params: ExperimentParameters

    def __str__(self) -> str:
        """Returns string representing this trial."""
        return (
            f"{self.modeling_strategy},"
            f"{self.system_params},"
            f"{self.experiment_params}"
        )

    def update_from_dataset(self, dataset: DataSet) -> "TrialDescriptor":
        """Uses the dataset to fill in various context-specific items."""
        system_params = self.system_params.update_from_dataset(dataset)
        experimental_params = self.experiment_params.update_from_dataset(
            dataset, system_params
        )
        modeling_strategy_descriptor = self.modeling_strategy.update_from_dataset(
            dataset
        )
        return TrialDescriptor(
            modeling_strategy_descriptor,
            system_params,
            experimental_params,
        )
