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
"""Defines an ExperimentalDesign

An experimental design consists of running a collection of ModelingStrategies,
SystemParameters and ExperimentParameters against a collection of DataSets.
"""

from absl import logging
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool, cpu_count
from tqdm import tqdm
from typing import List
from typing import Tuple


from wfa_planning_evaluation_framework.driver.trial_descriptor import TrialDescriptor
from wfa_planning_evaluation_framework.data_generators.data_design import (
    DataDesign,
)
from wfa_planning_evaluation_framework.driver.experiment import (
    Experiment,
)
from wfa_planning_evaluation_framework.driver.experimental_trial import (
    ExperimentalTrial,
)


class ExperimentalDesign:
    """Runs multiple ModelingStrategies against multiple DataSets."""

    def __init__(
        self,
        experiment_dir: str,
        data_design: DataDesign,
        trial_descriptors: List[TrialDescriptor],
        rng: np.random.Generator,
        cores: int = 1,
    ):
        """Constructs an ExperimentalDesign object.

        Args:
          experiment_dir:  A string specifying a directory where intermediate
            results will be written.
          data_design:  A DataDesign object specifying the data sets that will
            be used in this experimental design,
          trial_descriptors:  A list of tuples of the form
            (ModelingStrategyDescriptor, SystemParameters, ExperimentParameters).
            Each such tuple specifies one configuration of a modeling strategy
            and parameters that is to be tried against each data set.
          rng:  The source of randomness that will be used in this
            ExperimentalDesign.
          cores:  Specifies whether to use multithreading, and if so,
            how many cores should be used.  If cores=1, then multithreading
            is not used.  If cores<=0, then all available cores are used.
            If cores > 1, then cores specifies the exact number of cores to use.
        """
        self._experiment_dir = experiment_dir
        self._data_design = data_design
        self._trial_descriptors = trial_descriptors
        self._rng = rng
        self._all_trials = None
        self._cores = cores

    def generate_trials(self) -> List[ExperimentalTrial]:
        """Generates list of Trial objects associated to this experiment."""
        all_trials = []
        for data_set_name in self._data_design.names:
            experiment = Experiment(
                self._experiment_dir,
                self._data_design,
                data_set_name,
                self._trial_descriptors,
            )
            all_trials.extend(experiment.generate_trials())
        self._all_trials = all_trials
        return all_trials

    def _evaluate_all_trials_in_parallel(self) -> None:
        """Evaluates all trials in parallel."""
        if self._cores < 1:
            self._cores = cpu_count()
        logging.vlog(2, f'Using {self._cores} workers')
        ntrials = len(self._all_trials)
        def process_trial(i):
            self._all_trials[i].evaluate(self._rng)
        with ProcessPool(self._cores) as pool:
            list(tqdm(pool.uimap(process_trial, range(ntrials)), total=ntrials))
        
    def load(self) -> pd.DataFrame:
        """Returns a DataFrame of all results from this ExperimentalDesign."""
        if not self._all_trials:
            self.generate_trials()
        if self._cores != 1:
            self._evaluate_all_trials_in_parallel()
        all_results = [trial.evaluate(self._rng) for trial in self._all_trials]
        return pd.concat(all_results)
