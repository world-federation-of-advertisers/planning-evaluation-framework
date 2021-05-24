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
"""Tests for privacy_tracker.py."""

from absl.testing import absltest
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.estimators.same_key_aggregator import (
    StandardizedHistogramEstimator,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.simulator.publisher import Publisher
from wfa_planning_evaluation_framework.simulator.privacy_tracker import PrivacyTracker
from wfa_planning_evaluation_framework.simulator.simulation_parameters import (
    LiquidLegionsParameters,
    SimulationParameters,
)


class PublisherTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        pdf = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        cls.params = SimulationParameters(
            [1.0, 0.05, 3.0], LiquidLegionsParameters(), np.random.default_rng(1)
        )
        cls.privacy_tracker = PrivacyTracker()
        cls.publisher = Publisher(pdf, 1, cls.params, cls.privacy_tracker)

    def test_campaign_spend(self):
        self.assertEqual(self.publisher.campaign_spend, 0.05)

    def test_true_reach_by_spend(self):
        reach_point = self.publisher.true_reach_by_spend(0.04, 2)
        self.assertEqual(reach_point.reach(1), 2)
        self.assertEqual(reach_point.reach(2), 1)

    def test_liquid_legions_sketch(self):
        sketch = self.publisher.liquid_legions_sketch(0.04)
        estimator = StandardizedHistogramEstimator()
        cardinality = estimator.estimate_cardinality(sketch)[0]
        self.assertAlmostEqual(cardinality, 2, places=3)


if __name__ == "__main__":
    absltest.main()
