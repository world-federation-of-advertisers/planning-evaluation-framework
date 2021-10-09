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
"""Tests for ExperimentalDesign."""

from absl.testing import absltest
from os.path import join
from tempfile import TemporaryDirectory
from typing import Dict
from typing import Iterable
from typing import List
from typing import Type
import numpy as np
import pandas as pd
from unittest.mock import patch
import logging

from cloudpathlib.local import LocalGSClient, LocalGSPath

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from wfa_planning_evaluation_framework.data_generators.heterogeneous_impression_generator import (
    HeterogeneousImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.models.reach_curve import (
    ReachCurve,
)
from wfa_planning_evaluation_framework.models.reach_point import (
    ReachPoint,
)
from wfa_planning_evaluation_framework.models.reach_surface import (
    ReachSurface,
)
from wfa_planning_evaluation_framework.simulator.halo_simulator import (
    HaloSimulator,
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
from wfa_planning_evaluation_framework.driver.experiment_parameters import (
    TEST_POINT_STRATEGIES,
    ExperimentParameters,
)
from wfa_planning_evaluation_framework.driver.experiment import (
    Experiment,
)
import wfa_planning_evaluation_framework.driver.experimental_design as experimental_design
from wfa_planning_evaluation_framework.driver.experimental_design import (
    ExperimentalDesign,
    CombineDataFrameFn,
    EvaluateTrialDoFn,
)
from wfa_planning_evaluation_framework.driver.experimental_trial import (
    ExperimentalTrial,
)
from wfa_planning_evaluation_framework.driver.modeling_strategy_descriptor import (
    MODELING_STRATEGIES,
    ModelingStrategyDescriptor,
)
from wfa_planning_evaluation_framework.driver.test_point_generator import (
    TestPointGenerator,
)
from wfa_planning_evaluation_framework.driver.trial_descriptor import (
    TrialDescriptor,
)
import wfa_planning_evaluation_framework.data_generators.filesystem_path_client as filesystem_path_client


class FakeReachSurface(ReachSurface):
    def __init__(self):
        self._max_reach = 1

    def by_impressions(
        self, impressions: Iterable[int], max_frequency: int = 1
    ) -> ReachPoint:
        return ReachPoint(impressions, [1], impressions)

    def by_spend(self, spend: Iterable[float], max_frequency: int = 1) -> ReachPoint:
        return ReachPoint([1] * len(spend), [1], spend)


class FakeModelingStrategy(ModelingStrategy):
    def __init__(
        self,
        single_pub_model: Type[ReachCurve],
        single_pub_model_kwargs: Dict,
        multi_pub_model: Type[ReachSurface],
        multi_pub_model_kwargs: Dict,
        x: int,
    ):
        self.name = "fake"
        self.x = 1
        super().__init__(
            single_pub_model,
            single_pub_model_kwargs,
            multi_pub_model,
            multi_pub_model_kwargs,
        )

    def fit(
        self, halo: HaloSimulator, params: SystemParameters, budget: PrivacyBudget
    ) -> ReachSurface:
        return FakeReachSurface()


class FakeTestPointGenerator(TestPointGenerator):
    def __init__(self, data_set, rng):
        pass

    def test_points(self) -> Iterable[List[float]]:
        return [[1.0]]


class FakeExperimentalTrial:
    def __init__(self, base):
        self._base = base

    def evaluate(self, seed: int):
        return self._base * seed


class FakeEvaluateTrialDoFn(beam.DoFn):
    def process(self, trial, seed):
        import pandas as pd

        yield pd.DataFrame({"col": [1]})


class ExperimentalDesignTest(absltest.TestCase):
    def tearDown(self):
        LocalGSClient.reset_default_storage_dir()

    def _setup(self, tempdir):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        data_set1 = DataSet([pdf1], "dataset1")
        pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
        data_set2 = DataSet([pdf2], "dataset2")
        data_design_dir = join(tempdir, "data_design")
        self.experiment_dir = join(tempdir, "experiments")
        self.data_design = DataDesign(data_design_dir)
        self.data_design.add(data_set1)
        self.data_design.add(data_set2)

        MODELING_STRATEGIES["fake"] = FakeModelingStrategy
        TEST_POINT_STRATEGIES["fake_tps"] = FakeTestPointGenerator

        msd = ModelingStrategyDescriptor(
            "fake", {"x": 1}, "goerg", {}, "pairwise_union", {}
        )
        sparams1 = SystemParameters(
            [0.03],
            LiquidLegionsParameters(13, 1e6, 1),
            np.random.default_rng(),
        )
        sparams2 = SystemParameters(
            [0.05],
            LiquidLegionsParameters(13, 1e6, 1),
            np.random.default_rng(),
        )
        eparams1 = ExperimentParameters(PrivacyBudget(1.0, 0.01), 1, 5, "fake_tps")
        eparams2 = ExperimentParameters(PrivacyBudget(0.5, 0.001), 1, 5, "fake_tps")

        self.trial_descriptors = [
            TrialDescriptor(msd, sparams1, eparams1),
            TrialDescriptor(msd, sparams1, eparams2),
            TrialDescriptor(msd, sparams2, eparams1),
            TrialDescriptor(msd, sparams2, eparams2),
        ]

    def test_evaluate(self):
        with TemporaryDirectory() as d:
            self._setup(d)

            exp = ExperimentalDesign(
                self.experiment_dir,
                self.data_design,
                self.trial_descriptors,
                seed=1,
            )
            trials = exp.generate_trials()
            self.assertLen(trials, 8)

            results = exp.load()
            self.assertEqual(results.shape[0], 8)

    def test_evaluate_with_two_cores(self):
        with TemporaryDirectory() as d:
            self._setup(d)

            exp = ExperimentalDesign(
                self.experiment_dir,
                self.data_design,
                self.trial_descriptors,
                seed=1,
                cores=2,
            )
            trials = exp.generate_trials()
            self.assertLen(trials, 8)

            results = exp.load()
            self.assertEqual(results.shape[0], 8)

    def test_evaluate_with_apache_beam_locally(self):
        with TemporaryDirectory() as d:
            data_design_dir = join(d, "data_design")
            experiment_dir = join(d, "experiments")

            results = self._evaluate_with_apache_beam(data_design_dir, experiment_dir)
            # We don't check each column in the resulting dataframe, because these have
            # been checked by the preceding unit tests.  However, we make a few strategic
            # probes.
            self.assertEqual(results.shape[0], 1)
            self.assertAlmostEqual(results["relative_error_at_100"][0], 0.0, delta=0.01)
            self.assertGreater(results["max_nonzero_frequency_from_halo"][0], 0)
            self.assertEqual(results["max_nonzero_frequency_from_data"][0], 1)

    @patch.object(filesystem_path_client, "GSClient", LocalGSClient)
    @patch.object(experimental_design, "EvaluateTrialDoFn", FakeEvaluateTrialDoFn)
    def test_evaluate_with_apache_beam_with_cloud_path(self):
        fs_path_client = filesystem_path_client.FilesystemPathClient()
        parent_dir_path = fs_path_client.get_fs_path(
            "gs://ExperimentalDesignTest/parent"
        )
        data_design_dir_path = parent_dir_path.joinpath("data_design")
        experiment_dir_path = parent_dir_path.joinpath("experiments")
        data_design_dir_path.joinpath("dummy.txt").write_text(
            "For creating the target directory."
        )
        experiment_dir_path.joinpath("dummy.txt").write_text(
            "For creating the target directory."
        )

        results = self._evaluate_with_apache_beam(
            str(data_design_dir_path), str(experiment_dir_path)
        )
        # We don't check each column in the resulting dataframe, because these have
        # been checked by the preceding unit tests.  However, we make a few strategic
        # probes.
        self.assertEqual(results.shape[0], 1)
        self.assertEqual(results["col"][0], 1)

    def _evaluate_with_apache_beam(self, data_design_dir, experiment_dir):
        num_workers = 2
        data1 = HeterogeneousImpressionGenerator(
            1000, gamma_shape=1.0, gamma_scale=3.0
        )()
        pdf1 = PublisherData(FixedPriceGenerator(0.1)(data1))
        data_set = DataSet([pdf1], "dataset")
        data_design = DataDesign(data_design_dir)
        data_design.add(data_set)

        msd = ModelingStrategyDescriptor(
            "single_publisher", {}, "goerg", {}, "pairwise_union", {}
        )
        sparams = SystemParameters(
            [0.5],
            LiquidLegionsParameters(13, 1e6, 1),
            np.random.default_rng(),
        )
        eparams = ExperimentParameters(
            PrivacyBudget(1.0, 0.01), 3, 5, "grid", {"grid_size": 5}
        )
        trial_descriptors = [TrialDescriptor(msd, sparams, eparams)]

        exp = ExperimentalDesign(
            experiment_dir,
            data_design,
            trial_descriptors,
            seed=1,
            cores=num_workers,
            analysis_type="single_pub",
        )
        exp.generate_trials()

        pipeline_args = []
        pipeline_args.extend(
            [
                "--runner=direct",
                "--direct_running_mode=multi_processing",
                f"--temp_location={experiment_dir}",
                f"--direct_num_workers={num_workers}",
            ]
        )
        pipeline_options = PipelineOptions(pipeline_args)
        logging.disable(logging.CRITICAL)
        results = exp.load(use_apache_beam=True, pipeline_options=pipeline_options)
        logging.disable(logging.NOTSET)
        return results

    def test_evaluate_trial_do_fn(self):
        num_trials = 10
        seed = 1
        expected = list(range(num_trials))

        logging.disable(logging.CRITICAL)
        with TestPipeline() as p:
            output = (
                p
                | beam.Create([FakeExperimentalTrial(i) for i in range(num_trials)])
                | beam.ParDo(EvaluateTrialDoFn(), seed)
            )
            assert_that(output, equal_to(expected))
        logging.disable(logging.NOTSET)

    def test_combine_dataFrame_fn(self):
        num_dfs = 10
        key = "attr"
        val = 1
        dfs = [pd.DataFrame([{key: val}]) for _ in range(num_dfs)]
        expected = pd.concat(dfs).to_csv()

        logging.disable(logging.CRITICAL)
        with TestPipeline() as p:
            output = (
                p
                | beam.Create(dfs)
                | beam.CombineGlobally(CombineDataFrameFn())
                | beam.Map(lambda df: df.to_csv())
            )
            assert_that(output, equal_to([expected]))
        logging.disable(logging.NOTSET)

    def test_remove_duplicates(self):
        with TemporaryDirectory() as d:
            self._setup(d)

            self.trial_descriptors.append(self.trial_descriptors[0])

            exp = ExperimentalDesign(
                self.experiment_dir,
                self.data_design,
                self.trial_descriptors,
                seed=1,
            )
            trials = exp.generate_trials()
            self.assertLen(trials, 8)

            results = exp.load()
            self.assertEqual(results.shape[0], 8)


if __name__ == "__main__":
    absltest.main()
