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
"""Defines one experimental trial.

An experimental trial represents running a specific ModelingStrategy
against a specific DataSet, with specific SystemParameters.
"""

from absl import logging
import numpy as np
import pandas as pd
from os.path import isfile, join
from pathlib import Path
from typing import List
from typing import NamedTuple

from wfa_planning_evaluation_framework.data_generators.data_design import (
    DataDesign,
)
from wfa_planning_evaluation_framework.simulator.modeling_strategy import (
    ModelingStrategy,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    PrivacyBudget,
    PrivacyTracker,
)
from wfa_planning_evaluation_framework.simulator.halo_simulator import (
    HaloSimulator,
)
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    SystemParameters,
)
from wfa_planning_evaluation_framework.driver.experiment_parameters import (
    ExperimentParameters,
)
from wfa_planning_evaluation_framework.driver.modeling_strategy_descriptor import (
    ModelingStrategyDescriptor,
)
from wfa_planning_evaluation_framework.driver.trial_descriptor import (
    TrialDescriptor,
)
from wfa_planning_evaluation_framework.driver.test_point_aggregator import (
    aggregate,
    aggregate_on_failure,
)


class ExperimentalTrial:
    """A run of a ModelingStrategy against a DataSet."""

    def __init__(
        self,
        experiment_dir: str,
        data_design: DataDesign,
        data_set_name: str,
        trial_descriptor: TrialDescriptor,
    ):
        """Constructs an object representing a trial.

        A trial represents a run of a specific ModelingStrategy against a
        specific DataSet, with specific SystemParameters and ExperimentParameters.

        Args:
          experiment_dir:  The name of a directory where intermediate results
            are stored.  The results for this specific trial will be stored in
            the file {experiment_dir}/{data_set_name}/{trial_name}.  The
            trial_name is constructed from the ModelingStrategyDescriptor and the
            SystemParameters.
          data_design:  A DataDesign object specifying the source of the data
            that will be used for this trial.
          data_set_name:  The name of the specific DataSet within the DataDesign
            that will be used for this trial.
          trial_descriptor: A descriptor that specifies the configuration
            of this experimental trial.
        """
        self._experiment_dir = experiment_dir
        self._data_design = data_design
        self._data_set_name = data_set_name
        self._trial_descriptor = trial_descriptor

    def evaluate(self, rng: np.random.Generator) -> pd.DataFrame:
        """Executes a trial.

        1. Check if the results for the trial have already been computed.
        2. Load the DataSet.
        3. Instantiate Halo Simulator.
        4. Instantiate Modeling Strategy.
        5. Fit model.
        6. Generate set of test points.
        7. Compute metrics.
        8. Construct output DataFrame.
        9. Save to disk.

        Args:
          rng: An object of type numpy.random.Generator that is used as
            the source of randomness for this experiment.

        Returns:
          A single row DataFrame containing the results of the evaluation
          of this trial.
        """
        trial_results_path = self._compute_trial_results_path()
        if isfile(trial_results_path):
            return pd.read_csv(trial_results_path)

        self._dataset = self._data_design.by_name(self._data_set_name)
        self._privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(
            self._dataset, self._trial_descriptor.system_params, self._privacy_tracker
        )
        privacy_budget = self._trial_descriptor.experiment_params.privacy_budget
        modeling_strategy = (
            self._trial_descriptor.modeling_strategy.instantiate_strategy()
        )
        logging.vlog(2, f"Dataset {self._data_set_name}")
        logging.vlog(2, f"Trial   {self._trial_descriptor}")
        try:
            reach_surface = modeling_strategy.fit(
                halo, self._trial_descriptor.system_params, privacy_budget
            )
            test_points = list(
                self._trial_descriptor.experiment_params.generate_test_points(
                    self._dataset, rng
                )
            )
            true_reach = [
                halo.true_reach_by_spend(
                    t, self._trial_descriptor.experiment_params.max_frequency
                )
                for t in test_points
            ]
            simulated_reach = [
                reach_surface.by_spend(
                    t, self._trial_descriptor.experiment_params.max_frequency
                )
                for t in test_points
            ]
            metrics = aggregate(true_reach, simulated_reach)
        except Exception as inst:
            if not logging.vlog_is_on(2):
                logging.vlog(1, f"Dataset {self._data_set_name}")
                logging.vlog(1, f"Trial   {self._trial_descriptor}")
            logging.vlog(1, inst)
            metrics = aggregate_on_failure()

        independent_vars = self._make_independent_vars_dataframe()
        privacy_tracking_vars = self._make_privacy_tracking_vars_dataframe(
            self._privacy_tracker
        )
        result = pd.concat([independent_vars, privacy_tracking_vars, metrics], axis=1)
        Path(trial_results_path).parent.absolute().mkdir(parents=True, exist_ok=True)
        result.to_csv(trial_results_path)
        return result

    def _compute_trial_results_path(self) -> str:
        """Returns path of file where the results of this trial are stored."""
        return f"{self._experiment_dir}/{self._data_set_name}/{self._trial_descriptor}"

    def _make_independent_vars_dataframe(self) -> pd.DataFrame:
        """Returns a 1-row DataFrame of independent variables for this trial."""
        data_set = self._data_design.by_name(self._data_set_name)
        independent_vars = pd.DataFrame(
            {
                "dataset": [self._data_set_name],
                "replica_id": [self._trial_descriptor.experiment_params.replica_id],
                "single_pub_model": [
                    self._trial_descriptor.modeling_strategy.single_pub_model
                ],
                "multi_pub_model": [
                    self._trial_descriptor.modeling_strategy.multi_pub_model
                ],
                "strategy": [self._trial_descriptor.modeling_strategy.strategy],
                "liquid_legions_sketch_size": [
                    self._trial_descriptor.system_params.liquid_legions.sketch_size
                ],
                "liquid_legions_decay_rate": [
                    self._trial_descriptor.system_params.liquid_legions.decay_rate
                ],
                "maximum_reach": [data_set.maximum_reach],
                "ncampaigns": [data_set.publisher_count],
                "largest_pub_reach": [max([p.max_reach for p in data_set._data])],
                "max_frequency": [
                    self._trial_descriptor.experiment_params.max_frequency
                ],
            }
        )
        return independent_vars

    def _make_privacy_tracking_vars_dataframe(
        self, privacy_tracker: PrivacyTracker
    ) -> pd.DataFrame:
        """Returns a 1-row DataFrame of privacy-related data for this trial."""
        mechanisms_string = "/".join(sorted(set(privacy_tracker.mechanisms)))

        privacy_vars = pd.DataFrame(
            {
                "privacy_budget_epsilon": [
                    self._trial_descriptor.experiment_params.privacy_budget.epsilon
                ],
                "privacy_budget_delta": [
                    self._trial_descriptor.experiment_params.privacy_budget.delta
                ],
                "privacy_used_epsilon": [privacy_tracker.privacy_consumption.epsilon],
                "privacy_used_delta": [privacy_tracker.privacy_consumption.delta],
                "privacy_mechanisms": [mechanisms_string],
            }
        )
        return privacy_vars
