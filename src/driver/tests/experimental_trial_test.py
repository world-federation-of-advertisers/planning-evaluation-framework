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
"""Tests for modeling_strategy_descriptor."""

from absl.testing import absltest
from os.path import join
from tempfile import TemporaryDirectory
from typing import Dict
from typing import Iterable
from typing import List
from typing import Type
import math
import numpy as np
import pandas as pd
from unittest.mock import patch

from cloudpathlib.local import LocalGSClient, LocalGSPath

from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
import wfa_planning_evaluation_framework.data_generators.data_design as data_design
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
import wfa_planning_evaluation_framework.data_generators.data_set as data_set
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.heterogeneous_impression_generator import (
    HeterogeneousImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)
from wfa_planning_evaluation_framework.models.goerg_model import (
    GoergModel,
)
from wfa_planning_evaluation_framework.models.reach_curve import (
    ReachCurve,
)
from wfa_planning_evaluation_framework.models.reach_point import (
    ReachPoint,
)
from wfa_planning_evaluation_framework.models.reach_surface import (
    ReachSurface,
)
from wfa_planning_evaluation_framework.models.pairwise_union_reach_surface import (
    PairwiseUnionReachSurface,
)
from wfa_planning_evaluation_framework.simulator.halo_simulator import (
    HaloSimulator,
)
from wfa_planning_evaluation_framework.simulator.modeling_strategy import (
    ModelingStrategy,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    DP_NOISE_MECHANISM_GAUSSIAN,
    DP_NOISE_MECHANISM_LAPLACE,
    NoisingEvent,
    PrivacyBudget,
    PrivacyTracker,
)
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    LiquidLegionsParameters,
    SystemParameters,
)
from wfa_planning_evaluation_framework.driver.experiment_parameters import (
    TEST_POINT_STRATEGIES,
    ExperimentParameters,
)
import wfa_planning_evaluation_framework.driver.experimental_trial as experimental_trial
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
    def __init__(self, dataset, rng):
        pass

    def test_points(self) -> Iterable[List[float]]:
        return [[1.0, 2.0]]


class GoergModelingStrategy(ModelingStrategy):
    """Models a single publisher using Goerg's model."""

    def fit(
        self, halo: HaloSimulator, params: SystemParameters, budget: PrivacyBudget
    ) -> ReachSurface:
        total_reach = ReachPoint(
            [
                2,
            ],
            [
                2,
            ],
            [2.0],
        )
        curve = GoergModel([total_reach])
        curve._fit()
        return curve


class GoergTestPointGenerator(TestPointGenerator):
    def __init__(self, dataset, rng):
        pass

    def test_points(self) -> Iterable[List[float]]:
        return [[1.0]]


class ExperimentalTrialTest(absltest.TestCase):
    def tearDown(self):
        LocalGSClient.reset_default_storage_dir()

    def test_privacy_tracking_vars_dataframe(self):
        tracker = PrivacyTracker()
        eparams = ExperimentParameters(
            PrivacyBudget(1.0, 0.01), 1, 3, "test_point_strategy"
        )
        trial_descriptor = TrialDescriptor(None, None, eparams)
        trial = ExperimentalTrial("", None, "", trial_descriptor)

        actual0 = trial._make_privacy_tracking_vars_dataframe(tracker)
        expected0 = pd.DataFrame(
            {
                "privacy_budget_epsilon": [1.0],
                "privacy_budget_delta": [0.01],
                "privacy_used_epsilon": [0.0],
                "privacy_used_delta": [0.0],
                "privacy_mechanisms": [""],
            }
        )
        pd.testing.assert_frame_equal(actual0, expected0)

        tracker.append(
            NoisingEvent(PrivacyBudget(0.5, 0.005), DP_NOISE_MECHANISM_LAPLACE, {})
        )
        actual1 = trial._make_privacy_tracking_vars_dataframe(tracker)
        expected1 = pd.DataFrame(
            {
                "privacy_budget_epsilon": [1.0],
                "privacy_budget_delta": [0.01],
                "privacy_used_epsilon": [0.5],
                "privacy_used_delta": [0.005],
                "privacy_mechanisms": ["Laplace"],
            }
        )
        pd.testing.assert_frame_equal(actual1, expected1)

        tracker.append(
            NoisingEvent(PrivacyBudget(0.2, 0.002), DP_NOISE_MECHANISM_GAUSSIAN, {})
        )
        actual2 = trial._make_privacy_tracking_vars_dataframe(tracker)
        expected2 = pd.DataFrame(
            {
                "privacy_budget_epsilon": [1.0],
                "privacy_budget_delta": [0.01],
                "privacy_used_epsilon": [0.7],
                "privacy_used_delta": [0.007],
                "privacy_mechanisms": ["Gaussian/Laplace"],
            }
        )
        pd.testing.assert_frame_equal(actual2, expected2)

    def test_make_independent_vars_dataframe(self):
        with TemporaryDirectory() as d:
            pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
            pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
            data_set = DataSet([pdf1, pdf2], "dataset")
            data_design = DataDesign(join(d, "data_design"))
            data_design.add(data_set)

            msd = ModelingStrategyDescriptor(
                "strategy", {}, "single_pub_model", {}, "multi_pub_model", {}
            )
            sparams = SystemParameters(
                [0.03, 0.05],
                LiquidLegionsParameters(13, 1e6, 1),
                np.random.default_rng(),
            )
            eparams = ExperimentParameters(
                PrivacyBudget(1.0, 0.01), 3, 5, "test_point_strategy"
            )
            trial_descriptor = TrialDescriptor(msd, sparams, eparams)
            trial = ExperimentalTrial("edir", data_design, "dataset", trial_descriptor)

            actual = trial._make_independent_vars_dataframe()

            expected_trial_name = "strategy,single_pub_model,multi_pub_model,spends=[0.03,0.05],decay_rate=13,sketch_size=1000000.0,epsilon=1.0,delta=0.01,replica_id=3,max_frequency=5,test_point_strategy=test_point_strategy"

            expected = pd.DataFrame(
                {
                    "dataset": ["dataset"],
                    "trial": [expected_trial_name],
                    "replica_id": [3],
                    "single_pub_model": ["single_pub_model"],
                    "multi_pub_model": ["multi_pub_model"],
                    "strategy": ["strategy"],
                    "liquid_legions_sketch_size": [1e6],
                    "liquid_legions_decay_rate": [13],
                    "maximum_reach": [4],
                    "ncampaigns": [2],
                    "largest_pub_reach": [3],
                    "max_frequency": [5],
                    "average_spend_fraction": [0.04],
                }
            )
            pd.testing.assert_frame_equal(actual, expected)

    def test_compute_trial_results_path(self):
        with TemporaryDirectory() as d:
            pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
            pdf2 = PublisherData([(2, 0.03), (4, 0.06)], "pdf2")
            data_set = DataSet([pdf1, pdf2], "dataset")
            data_design = DataDesign(join(d, "data_design"))
            data_design.add(data_set)

            msd = ModelingStrategyDescriptor(
                "strategy", {}, "single_pub_model", {}, "multi_pub_model", {}
            )
            sparams = SystemParameters(
                [0.03, 0.05],
                LiquidLegionsParameters(13, 1e6, 1),
                np.random.default_rng(),
            )
            eparams = ExperimentParameters(PrivacyBudget(1.0, 0.01), 3, 5, "tps")
            trial_descriptor = TrialDescriptor(msd, sparams, eparams)
            trial = ExperimentalTrial("edir", data_design, "dataset", trial_descriptor)

            actual = trial._compute_trial_results_path()
            expected = "{}/{}/{},{},{},{}".format(
                "edir",
                "dataset",
                "strategy,single_pub_model,multi_pub_model",
                "spends=[0.03,0.05],decay_rate=13,sketch_size=1000000.0",
                "epsilon=1.0,delta=0.01,replica_id=3,max_frequency=5",
                "test_point_strategy=tps.csv",
            )
            self.assertEqual(actual, expected)

    def test_evaluate(self):
        with TemporaryDirectory() as d:
            pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
            pdf2 = PublisherData([(2, 0.02), (2, 0.03), (4, 0.06)], "pdf2")
            data_set = DataSet([pdf1, pdf2], "dataset")
            data_design_dir = join(d, "data_design")
            experiment_dir = join(d, "experiments")
            data_design = DataDesign(data_design_dir)
            data_design.add(data_set)

            MODELING_STRATEGIES["fake"] = FakeModelingStrategy
            TEST_POINT_STRATEGIES["fake_tps"] = FakeTestPointGenerator

            msd = ModelingStrategyDescriptor(
                "fake", {"x": 1}, "goerg", {}, "pairwise_union", {}
            )
            sparams = SystemParameters(
                [0.9, 0.9],
                LiquidLegionsParameters(13, 1e6, 1),
                np.random.default_rng(),
            )
            eparams = ExperimentParameters(PrivacyBudget(1.0, 0.01), 3, 5, "fake_tps")
            trial_descriptor = TrialDescriptor(msd, sparams, eparams)
            trial = ExperimentalTrial(
                experiment_dir, data_design, "dataset", trial_descriptor
            )
            result = trial.evaluate(seed=1)
            # We don't check each column in the resulting dataframe, because these have
            # been checked by the preceding unit tests.  However, we make a few strategic
            # probes.
            self.assertEqual(result.shape[0], 1)
            self.assertEqual(result["dataset"][0], "dataset")
            self.assertEqual(result["replica_id"][0], 3)
            self.assertEqual(result["privacy_budget_epsilon"][0], 1.0)
            self.assertEqual(result["npoints"][0], 1)
            self.assertEqual(result["model_succeeded"][0], 1)
            self.assertEqual(result["model_exception"][0], "")

    @patch.object(experimental_trial, "GSClient", LocalGSClient)
    @patch.object(data_design, "GSPath", LocalGSPath)
    @patch.object(data_set, "GSPath", LocalGSPath)
    def test_evaluate_with_cloud_path(self):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        pdf2 = PublisherData([(2, 0.02), (2, 0.03), (4, 0.06)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "dataset")

        parent_dir_path = LocalGSPath(
            "gs://parallel_planning_evaluation_framework/parent"
        )
        data_design_dir_path = parent_dir_path.joinpath("data_design")
        experiment_dir_path = parent_dir_path.joinpath("experiments")

        data_design_dir_path.joinpath("dummy.txt").write_text(
            "For creating the target directory."
        )
        experiment_dir_path.joinpath("dummy.txt").write_text(
            "For creating the target directory."
        )

        data_design = DataDesign(str(data_design_dir_path))
        data_design.add(data_set)

        MODELING_STRATEGIES["fake"] = FakeModelingStrategy
        TEST_POINT_STRATEGIES["fake_tps"] = FakeTestPointGenerator

        msd = ModelingStrategyDescriptor(
            "fake", {"x": 1}, "goerg", {}, "pairwise_union", {}
        )
        sparams = SystemParameters(
            [0.9, 0.9],
            LiquidLegionsParameters(13, 1e6, 1),
            np.random.default_rng(),
        )
        eparams = ExperimentParameters(PrivacyBudget(1.0, 0.01), 3, 5, "fake_tps")
        trial_descriptor = TrialDescriptor(msd, sparams, eparams)
        trial = ExperimentalTrial(
            str(experiment_dir_path), data_design, "dataset", trial_descriptor
        )
        result = trial.evaluate(seed=1)
        # We don't check each column in the resulting dataframe, because these have
        # been checked by the preceding unit tests.  However, we make a few strategic
        # probes.
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result["dataset"][0], "dataset")
        self.assertEqual(result["replica_id"][0], 3)
        self.assertEqual(result["privacy_budget_epsilon"][0], 1.0)
        self.assertEqual(result["npoints"][0], 1)
        self.assertEqual(result["model_succeeded"][0], 1)
        self.assertEqual(result["model_exception"][0], "")

    def test_evaluate_when_there_is_a_modeling_exception(self):
        with TemporaryDirectory() as d:
            pdf1 = PublisherData([(1, 0.01), (2, 0.02), (3, 0.04), (4, 0.05)], "pdf1")
            data_set = DataSet([pdf1], "dataset")
            data_design_dir = join(d, "data_design")
            experiment_dir = join(d, "experiments")
            data_design = DataDesign(data_design_dir)
            data_design.add(data_set)

            MODELING_STRATEGIES["fake"] = GoergModelingStrategy
            TEST_POINT_STRATEGIES["fake_tps"] = GoergTestPointGenerator

            msd = ModelingStrategyDescriptor(
                "fake", {}, "goerg", {}, "pairwise_union", {}
            )
            sparams = SystemParameters(
                [0.5],
                LiquidLegionsParameters(13, 1e6, 1),
                np.random.default_rng(),
            )
            eparams = ExperimentParameters(PrivacyBudget(1.0, 0.01), 3, 5, "fake_tps")
            trial_descriptor = TrialDescriptor(msd, sparams, eparams)
            trial = ExperimentalTrial(
                experiment_dir, data_design, "dataset", trial_descriptor
            )
            result = trial.evaluate(seed=1)
            # We don't check each column in the resulting dataframe, because these have
            # been checked by the preceding unit tests.  However, we make a few strategic
            # probes.
            self.assertEqual(result.shape[0], 1)
            self.assertEqual(result["dataset"][0], "dataset")
            self.assertEqual(result["replica_id"][0], 3)
            self.assertEqual(result["privacy_budget_epsilon"][0], 1.0)
            self.assertEqual(result["model_succeeded"][0], 0)
            self.assertEqual(
                result["model_exception"][0],
                "Cannot fit Goerg model when impressions <= reach.",
            )

    def test_evaluate_singe_publisher_model_with_exception(self):
        with TemporaryDirectory() as d:
            pdf1 = PublisherData([(1, 0.01), (2, 0.02), (3, 0.04), (4, 0.05)], "pdf1")
            data_set = DataSet([pdf1], "dataset")
            data_design_dir = join(d, "data_design")
            experiment_dir = join(d, "experiments")
            data_design = DataDesign(data_design_dir)
            data_design.add(data_set)

            MODELING_STRATEGIES["fake"] = GoergModelingStrategy
            TEST_POINT_STRATEGIES["fake_tps"] = GoergTestPointGenerator

            msd = ModelingStrategyDescriptor(
                "fake", {}, "goerg", {}, "pairwise_union", {}
            )
            sparams = SystemParameters(
                [0.5],
                LiquidLegionsParameters(13, 1e6, 1),
                np.random.default_rng(),
            )
            eparams = ExperimentParameters(PrivacyBudget(1.0, 0.01), 3, 5, "fake_tps")
            trial_descriptor = TrialDescriptor(msd, sparams, eparams)
            trial = ExperimentalTrial(
                experiment_dir,
                data_design,
                "dataset",
                trial_descriptor,
                analysis_type="single_pub",
            )
            result = trial.evaluate(seed=1)
            # We don't check each column in the resulting dataframe, because these have
            # been checked by the preceding unit tests.  However, we make a few strategic
            # probes.
            self.assertEqual(result.shape[0], 1)
            self.assertTrue(math.isnan(result["relative_error_at_100"][0]))

    def test_evaluate_single_publisher_model(self):
        with TemporaryDirectory() as d:
            data1 = HeterogeneousImpressionGenerator(
                1000, gamma_shape=1.0, gamma_scale=3.0
            )()
            pdf1 = PublisherData(FixedPriceGenerator(0.1)(data1))
            data_set = DataSet([pdf1], "dataset")
            data_design_dir = join(d, "data_design")
            experiment_dir = join(d, "experiments")
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
            trial_descriptor = TrialDescriptor(msd, sparams, eparams)
            trial = ExperimentalTrial(
                experiment_dir,
                data_design,
                "dataset",
                trial_descriptor,
                analysis_type="single_pub",
            )
            result = trial.evaluate(seed=1)
            # We don't check each column in the resulting dataframe, because these have
            # been checked by the preceding unit tests.  However, we make a few strategic
            # probes.
            self.assertEqual(result.shape[0], 1)
            self.assertAlmostEqual(result["relative_error_at_100"][0], 0.0, delta=0.01)
            self.assertGreater(result["max_nonzero_frequency_from_halo"][0], 0)
            self.assertEqual(result["max_nonzero_frequency_from_data"][0], 5)


if __name__ == "__main__":
    absltest.main()
