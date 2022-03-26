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
import itertools

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

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
from wfa_planning_evaluation_framework.filesystem_wrappers import (
    filesystem_wrapper_base,
)
from wfa_planning_evaluation_framework.filesystem_wrappers import (
    filesystem_pathlib_wrapper,
)


FsWrapperBase = filesystem_wrapper_base.FilesystemWrapperBase
FsPathlibWrapper = filesystem_pathlib_wrapper.FilesystemPathlibWrapper


class EvaluateTrialDoFn(beam.DoFn):
    def process(self, trial, seed, filesystem):
        yield trial.evaluate(seed, filesystem)


class CombineDataFrameFn(beam.CombineFn):
    def create_accumulator(self):
        return []

    def add_input(self, accumulator, input):
        accumulator.append(input)
        return accumulator

    def merge_accumulators(self, accumulators):
        return list(itertools.chain.from_iterable(accumulators))

    def extract_output(self, accumulator):
        return pd.concat(accumulator)


class ExperimentalDesign:
    """Runs multiple ModelingStrategies against multiple DataSets."""

    def __init__(
        self,
        experiment_dir: str,
        data_design: DataDesign,
        trial_descriptors: List[TrialDescriptor],
        seed: int = 1,
        cores: int = 1,
        analysis_type: str = "",
        filesystem: FsWrapperBase = FsPathlibWrapper(),
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
          seed:  The seed that is used for the random number generator.
            Every trial will be initialized with the same seed.
          cores:  Specifies whether to use multithreading, and if so,
            how many cores should be used.  If cores=1, then multithreading
            is not used.  If cores<=0, then all available cores are used.
            If cores > 1, then cores specifies the exact number of cores to use.
          analysis_type:  Type of analysis.  Can be empty of "single_pub".  If
            "single_pub" is specified, then additional columns are added to the
            output that are specific to single publisher analysis.
          filesystem:  The filesystem object that manages all file operations.
        """
        self._experiment_dir = experiment_dir
        self._data_design = data_design
        self._trial_descriptors = trial_descriptors
        self._seed = seed
        self._all_trials = None
        self._cores = cores
        self._analysis_type = analysis_type
        self._filesystem = filesystem

    def _remove_duplicates(
        self, trials: List[ExperimentalTrial]
    ) -> List[ExperimentalTrial]:
        trial_name_set = set()
        output_list = []
        for t in trials:
            trial_name = f"{t._data_set_name}::{t._trial_descriptor}"
            if not trial_name in trial_name_set:
                trial_name_set.add(trial_name)
                output_list.append(t)
        return output_list

    def generate_trials(self) -> List[ExperimentalTrial]:
        """Generates list of Trial objects associated to this experiment."""
        all_trials = []
        for data_set_name in self._data_design.names:
            experiment = Experiment(
                self._experiment_dir,
                self._data_design,
                data_set_name,
                self._trial_descriptors,
                analysis_type=self._analysis_type,
            )
            all_trials.extend(experiment.generate_trials())
        all_trials = self._remove_duplicates(all_trials)
        self._all_trials = all_trials
        return all_trials

    def _evaluate_all_trials_in_parallel(self) -> None:
        """Evaluates all trials in parallel."""
        if self._cores < 1:
            self._cores = cpu_count()
        logging.vlog(2, f"Using {self._cores} workers")
        ntrials = len(self._all_trials)

        def process_trial(i):
            self._all_trials[i].evaluate(self._seed, self._filesystem)

        with ProcessPool(self._cores) as pool:
            list(tqdm(pool.uimap(process_trial, range(ntrials)), total=ntrials))

    def _evaluate_all_trials_using_apache_beam(
        self, pipeline_options: PipelineOptions, result_path: str
    ):
        with beam.Pipeline(options=pipeline_options) as pipeline:
            (
                pipeline
                | "Create trial inputs" >> beam.Create(self._all_trials)
                | "Evaluate trials"
                >> beam.ParDo(EvaluateTrialDoFn(), self._seed, self._filesystem)
                | "Combine results" >> beam.CombineGlobally(CombineDataFrameFn())
                | "Write combined result"
                >> beam.Map(
                    lambda df: self._filesystem.write_text(
                        result_path, df.to_csv(na_rep="NaN", index=False)
                    )
                )
            )

    def load(
        self,
        use_apache_beam: bool = False,
        pipeline_options: PipelineOptions = PipelineOptions(),
    ) -> pd.DataFrame:
        """Returns a DataFrame of all results from this ExperimentalDesign."""
        if not self._all_trials:
            self.generate_trials()

        temp_location = pipeline_options.get_all_options().get("temp_location", "")
        temp_result_path = (
            temp_location + "/temp_result.csv" if temp_location else "/temp_result.csv"
        )

        if use_apache_beam:
            self._evaluate_all_trials_using_apache_beam(
                pipeline_options, temp_result_path
            )
        elif self._cores != 1:
            self._evaluate_all_trials_in_parallel()

        result = None
        if use_apache_beam:
            with self._filesystem.open(temp_result_path) as file:
                result = pd.read_csv(file)
        else:
            result = pd.concat(
                trial.evaluate(self._seed, self._filesystem)
                for trial in self._all_trials
            )

        return result
