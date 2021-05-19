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
against a specific DataSet, with specific SimulationParameters.
"""

from numpy.random import Generator
from typing import List
from typing import NamedTuple


class Trial:
    """A run of a ModelingStrategy against a DataSet."""

    def __init__(self,
                 experiment_dir: string,
                 data_design: DataDesign,
                 data_set_name: string,
                 modeling_strategy_descriptor: ModelingStrategyDescriptor,
                 simulation_params: SimulationParameters,
                 experiment_params: ExperimentParameters):
        """Constructs an object representing a trial.

        A trial represents a run of a specific ModelingStrategy against a
        specific DataSet, with specific SimulationParameters.

        Args:
          experiment_dir:  The name of a directory where intermediate results
            are stored.  The results for this specific trial will be stored in
            the file {experiment_dir}/{data_set_name}/{trial_name}.  The 
            trial_name is constructed from the ModelingStrategyDescriptor and the
            SimulationParameters.
          data_design:  A DataDesign object specifying the source of the data
            that will be used for this trial.
          data_set_name:  The name of the specific DataSet within the DataDesign
            that will be used for this trial.
          modeling_strategy_descriptor:  A descriptor that specifies the specific
            modeling strategy that will be used as well as any configuration
            parameters specific to that modeling strategy.
          simulation_params:  A descriptor that specifies the configuration of 
            the Halo simulator that will be used for this trial.
          experiment_params:  A descriptor that specifies configuration parameters
            for this experiment.
        """
        self._experiment_dir = experiment_dir
        self._data_design = data_design
        self._data_set_name = data_set_name
        self._modeling_strategy_descriptor = modeling_strategy_descriptor
        self._simulation_params = simulation_params
        self._experiment_params = experiment_params

    def evaluate(self) -> pandas.DataFrame:
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

        Returns:
          A single row DataFrame containing the results of the evaluation
          of this trial.
        """
        trial_results_path = self._compute_trial_results_path()
        if os.path.isfile(trial_results_path):
            return pd.read_csv(trial_results_path)

        self._dataset = self._data_design.by_name(self._data_set_name)
        self._privacy_tracker = PrivacyTracker()
        halo = HaloSimulator(self._dataset, self._simulation_params, self._privacy_tracker)
        privacy_budget = self._experiment_params.privacy_budget
        modeling_strategy = self._modeling_strategy_descriptor.get_strategy()
        modeling_strategy.fit(halo, privacy_budget)

        test_points = self._modeling_strategy_descriptor.generate_test_points()
        true_reach = [halo.true_reach_by_spend(t, self._modeling_strategy_descriptor.max_frequency)
                      for t in test_points]
        simulated_reach = [halo.simulated_reach_by_spend(t, self._modeling_strategy_descriptor.max_frequency)
                           for t in test_points]

        independent_vars = self._make_independent_vars_dataframe()
        privacy_tracking_vars = self._make_privacy_tracking_vars_dataframe()
        metrics = aggregate(true_reach, simulated_reach)
        result = pd.concat([independent_vars, privacy_tracking_vars, metrics], axis=1)
        result.to_csv(trial_results_path)
        return result
    
    def _compute_trial_results_path(self):
        """Returns path of file where the results of this trial are stored."""
        trial_name = (str(self._modeling_strategy_descriptor) +
                      str(self._simulation_params) +
                      str(self._experiment_params))
        return "{}/{}/{}".format(self._experiment_dir, self._data_set_name, trial_name)
        
    def _make_independent_vars_dataframe(self):
        """Returns a 1-row DataFrame of independent variables for this trial."""
        independent_vars = pd.DataFrame({
            'dataset': self._data_set_name,
            'replica': self._experiment_params.replica,
            'single_pub_model': self._modeling_strategy_descriptor.single_pub_model,
            'pricing_model': self._modeling_strategy_descriptor.pricing_model,
            'multi_pub_model': self._modeling_strategy_descriptor.multi_pub_model,
            'strategy': self._modeling_strategy_descriptor.name,
            'liquid_legions_sketch_size': self._simulation_params.liquid_legions.sketch_size,
            'liquid_legions_decay_rate': self._simulation_params.liquid_legions.decay_rate,
            'maximum_reach': self._dataset.maximum_reach,
            'ncampaigns': self._dataset.publisher_count,
            'largest_pub_reach': max([p.reach for p in self._dataset._data]),
            'max_frequency': self._modeling_strategy_descriptor.max_frequency,
        })
        return independent_vars

    def _make_privacy_tracking_vars_dataframe(self, privacy_tracker: PrivacyTracker):
        """Returns a 1-row DataFrame of privacy-related data for this trial."""
        
    
