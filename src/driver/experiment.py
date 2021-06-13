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
"""Defines an Experiment.

An experiment consists of running a collection of ModelingStrategies,
SystemParameters and ExperimentParameters against one specific DataSet.
"""

from typing import List
from typing import Tuple

from wfa_planning_evaluation_framework.data_generators.data_design import (
    DataDesign,
)
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    SystemParameters,
)
from wfa_planning_evaluation_framework.driver.experimental_trial import (
    ExperimentalTrial,
)
from wfa_planning_evaluation_framework.driver.modeling_strategy_descriptor import (
    ModelingStrategyDescriptor,
)
from wfa_planning_evaluation_framework.driver.experiment_parameters import (
    ExperimentParameters,
)
from wfa_planning_evaluation_framework.driver.trial_descriptor import (
    TrialDescriptor,
)


class Experiment:
    """Runs of multiple ModelingStrategies against a single DataSet."""

    def __init__(
        self,
        experiment_dir: str,
        data_design: DataDesign,
        data_set_name: str,
        trial_descriptors: List[TrialDescriptor],
    ):
        """Constructs an Experiment object.

        Args:
          experiment_dir:  A string specifying a directory where intermediate
            results will be written.
          data_design:  A DataDesign object specifying the data sets that will
            be used in this experimental design,
          data_set_name:  The name of the data set within the DataDesign that
            will be used for this Experiment.
          trial_descriptors:  A list of tuples of the form
            (ModelingStrategyDescriptor, SystemParameters, ExperimentParameters).
            Each such tuple specifies one configuration of a modeling strategy
            and parameters that is to be tried against each data set.
        """
        self._experiment_dir = experiment_dir
        self._data_design = data_design
        self._data_set_name = data_set_name
        self._trial_descriptors = trial_descriptors

    def generate_trials(self) -> List[ExperimentalTrial]:
        """Generates list of Trial objects associated to this experiement."""
        trials = []
        for desc in self._trial_descriptors:
            trials.append(
                ExperimentalTrial(
                    self._experiment_dir, self._data_design, self._data_set_name, desc
                )
            )
        self._trials = trials
        return trials
