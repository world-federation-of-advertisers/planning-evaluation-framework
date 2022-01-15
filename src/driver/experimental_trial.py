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
from datetime import datetime
import hashlib
import numpy as np
import pandas as pd
from os.path import isfile, join
import traceback
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
    aggregate_on_exception,
)
from wfa_planning_evaluation_framework.filesystem_wrappers import (
    filesystem_wrapper_base,
)
from wfa_planning_evaluation_framework.filesystem_wrappers import (
    filesystem_pathlib_wrapper,
)

FsWrapperBase = filesystem_wrapper_base.FilesystemWrapperBase
FsPathlibWrapper = filesystem_pathlib_wrapper.FilesystemPathlibWrapper

# The output dataframe will contain the estimation error for each of the
# following relative spend fractions.  In other words, if r is one of the
# values below and s is the spend fraction associated to the training point,
# then evaluate the relative error at r * s.
SINGLE_PUBLISHER_FRACTIONS = np.arange(1, 31) * 0.1

SINGLE_PUB_ANALYSIS = "single_pub"


class ExperimentalTrial:
    """A run of a ModelingStrategy against a DataSet."""

    def __init__(
        self,
        experiment_dir: str,
        data_design: DataDesign,
        data_set_name: str,
        trial_descriptor: TrialDescriptor,
        analysis_type: str = "",
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
          analysis_type:  Type of analysis.  Can be empty of "single_pub".  If
            "single_pub" is specified, then additional columns are added to the
            output that are specific to single publisher analysis.
        """
        self._experiment_dir = experiment_dir
        self._data_design = data_design
        self._data_set_name = data_set_name
        self._trial_descriptor = trial_descriptor
        self._analysis_type = analysis_type

    def evaluate(
        self, seed: int, filesystem: FsWrapperBase = FsPathlibWrapper()
    ) -> pd.DataFrame:
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
          seed:  A seed value that is used to initialize the random
            number generator.
          filesystem:  The filesystem object that manages all file operations.

        Returns:
          A single row DataFrame containing the results of the evaluation
          of this trial.
        """
        logging.vlog(2, f"Dataset {self._data_set_name}")
        logging.vlog(2, f"Trial   {self._trial_descriptor}")

        rng = np.random.default_rng(seed=seed)
        np.random.seed(seed)

        trial_results_path = self._compute_trial_results_path()

        if trial_results_path.startswith("gs://"):
            filesystem.set_default_client_to_gs_client()

        if filesystem.is_file(trial_results_path):
            logging.vlog(2, "  --> Returning previously computed result")
            try:
                with filesystem.open(trial_results_path) as file:
                    return pd.read_csv(file)
            except Exception as e:
                filesystem.unlink(trial_results_path)
                logging.vlog(
                    2, f"  --> {e}. Failed reading existing result. Re-evaluate."
                )

        # The pending directory contains one entry for each currently executing
        # experimental trial.  If a computation appears to hang, this can be
        # used to check which evaluations are still pending.
        experiment_dir_parent = filesystem.parent(self._experiment_dir)
        pending_path = f"{experiment_dir_parent}/pending/{hashlib.md5(trial_results_path.encode()).hexdigest()}"
        filesystem.mkdir(filesystem.parent(pending_path), parents=True, exist_ok=True)
        filesystem.write_text(
            pending_path,
            f"{datetime.now()}\n{self._data_set_name}\n{self._trial_descriptor}\n\n",
        )

        dataset = self._data_design.by_name(self._data_set_name)
        privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(
            dataset, self._trial_descriptor.system_params, privacy_tracker
        )
        privacy_budget = self._trial_descriptor.experiment_params.privacy_budget
        modeling_strategy = (
            self._trial_descriptor.modeling_strategy.instantiate_strategy()
        )
        single_publisher_dataframe = pd.DataFrame()
        max_frequency = self._trial_descriptor.experiment_params.max_frequency
        try:
            reach_surface = modeling_strategy.fit(
                halo, self._trial_descriptor.system_params, privacy_budget
            )
            test_points = list(
                self._trial_descriptor.experiment_params.generate_test_points(
                    dataset, rng
                )
            )
            true_reach = [
                halo.true_reach_by_spend(
                    t, self._trial_descriptor.experiment_params.max_frequency
                )
                for t in test_points
            ]
            fitted_reach = [
                reach_surface.by_spend(
                    t, self._trial_descriptor.experiment_params.max_frequency
                )
                for t in test_points
            ]
            metrics = aggregate(true_reach, fitted_reach)
            if self._analysis_type == SINGLE_PUB_ANALYSIS:
                single_publisher_dataframe = (
                    self._compute_single_publisher_fractions_dataframe(
                        halo, reach_surface, max_frequency
                    )
                )
        except Exception as inst:
            if not logging.vlog_is_on(2):
                logging.vlog(1, f"Dataset {self._data_set_name}")
                logging.vlog(1, f"Trial   {self._trial_descriptor}")
            logging.vlog(1, f"Modeling failure: {inst}")
            logging.vlog(2, traceback.format_exc())
            metrics = aggregate_on_exception(inst)
            if self._analysis_type == SINGLE_PUB_ANALYSIS:
                single_publisher_dataframe = (
                    self._single_publisher_fractions_dataframe_on_exception(
                        max_frequency
                    )
                )

        independent_vars = self._make_independent_vars_dataframe()
        privacy_tracking_vars = self._make_privacy_tracking_vars_dataframe(
            privacy_tracker
        )
        result = pd.concat(
            [
                independent_vars,
                privacy_tracking_vars,
                metrics,
                single_publisher_dataframe,
            ],
            axis=1,
        )
        filesystem.mkdir(
            filesystem.parent(trial_results_path), parents=True, exist_ok=True
        )
        filesystem.write_text(trial_results_path, result.to_csv(index=False))
        filesystem.unlink(pending_path, missing_ok=True)

        return result

    def _compute_trial_results_path(self) -> str:
        """Returns path of file where the results of this trial are stored."""
        return (
            f"{self._experiment_dir}/{self._data_set_name}/{self._trial_descriptor}.csv"
        )

    def _make_independent_vars_dataframe(self) -> pd.DataFrame:
        """Returns a 1-row DataFrame of independent variables for this trial."""
        data_set = self._data_design.by_name(self._data_set_name)
        independent_vars = pd.DataFrame(
            {
                "dataset": [self._data_set_name],
                "trial": [f"{self._trial_descriptor}"],
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
                "average_spend_fraction": [
                    np.mean(
                        self._trial_descriptor.system_params.campaign_spend_fractions
                    )
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

    def _compute_single_publisher_fractions_dataframe(
        self, halo, reach_surface, max_frequency
    ) -> pd.DataFrame:
        results = {}
        for r in SINGLE_PUBLISHER_FRACTIONS:
            spend = halo.campaign_spends[0] * r
            true_reach = halo.true_reach_by_spend([spend], 1).reach()
            fitted_reach = reach_surface.by_spend([spend], 1).reach()
            if true_reach:
                relative_error = np.abs((true_reach - fitted_reach) / true_reach)
            else:
                relative_error = np.NaN
            column_name = f"relative_error_at_{int(r*100):03d}"
            results[column_name] = [relative_error]

        for f in range(1, max_frequency):
            for r in SINGLE_PUBLISHER_FRACTIONS:
                spend = halo.campaign_spends[0] * r
                true_reach = halo.true_reach_by_spend([spend], f).reach(f)
                fitted_reach = reach_surface.by_spend([spend], f).reach(f)
                if true_reach:
                    relative_error = np.abs((true_reach - fitted_reach) / true_reach)
                else:
                    relative_error = np.NaN
                column_name = f"freq_{f}_relative_error_at_{int(r*100):03d}"
                results[column_name] = [relative_error]

        # Also, record the maximum frequency in the actual data and the
        # data produced by Halo.
        training_point = reach_surface._data[0]
        results["max_nonzero_frequency_from_halo"] = [
            max(
                [(i + 1) for i, f in enumerate(training_point._kplus_reaches) if f != 0]
                + [0]
            )
        ]
        data_point = halo.true_reach_by_spend(halo.campaign_spends, max_frequency)
        results["max_nonzero_frequency_from_data"] = [
            max(
                [(i + 1) for i, f in enumerate(data_point._kplus_reaches) if f != 0]
                + [0]
            )
        ]
        return pd.DataFrame(results)

    def _single_publisher_fractions_dataframe_on_exception(
        self, max_frequency
    ) -> pd.DataFrame:
        results = {}
        for r in SINGLE_PUBLISHER_FRACTIONS:
            column_name = f"relative_error_at_{int(r*100):03d}"
            results[column_name] = [np.NaN]
        for f in range(1, max_frequency):
            for r in SINGLE_PUBLISHER_FRACTIONS:
                column_name = f"freq_{f}_relative_error_at_{int(r*100):03d}"
                results[column_name] = [np.NaN]
        results["max_nonzero_frequency_from_halo"] = [np.NaN]
        results["max_nonzero_frequency_from_data"] = [np.NaN]
        return pd.DataFrame(results)
