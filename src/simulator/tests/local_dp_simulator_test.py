# Copyright 2022 The Private Cardinality Estimation Framework Authors
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
"""Tests for local_dp_simulator.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.simulator.system_parameters import (
    LiquidLegionsParameters,
    SystemParameters,
)
from wfa_planning_evaluation_framework.simulator.privacy_tracker import (
    PrivacyBudget,
    PrivacyTracker,
)
from wfa_planning_evaluation_framework.simulator.local_dp_simulator import (
    LocalDpSimulator,
)


class LocalDpSimulatorTest(parameterized.TestCase):

    # It suffices to test the simulated_reach_by_spend method.  Other methods
    # were copy-pasted from halo_simulator, so do not need to be tested.
    @parameterized.parameters(
        {
            "spends": [0.05, 0.06, 0.05],
            "expected_reach": 4,
        },
        {
            "spends": [0.02, 0.02, 0.02],
            "expected_reach": 2,
        },
        {
            "spends": [0.03, 0.03, 0.03],
            "expected_reach": 3,
        },
        {
            "spends": [0, 0, 0],
            "expected_reach": 0,
        },
    )
    def test_simulated_reach_by_spend_no_privacy(self, spends, expected_reach):
        pdfs = [
            PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1"),
            PublisherData([(2, 0.03), (4, 0.06)], "pdf2"),
            PublisherData([(2, 0.01), (3, 0.03), (4, 0.05)], "pdf3"),
        ]
        data_set = DataSet(pdfs, "test")
        params = SystemParameters(
            campaign_spend_fractions=[1, 1, 1],
            liquid_legions=LiquidLegionsParameters(
                decay_rate=1,
                sketch_size=10_000,
                random_seed=0,
            ),
            generator=np.random.default_rng(0),
        )
        # Set a large epsilon, so practically zero noise.
        # Minor note: due an math.exp(epsilon) expression in the sketch codes,
        # a too large epsilon will introduce overflow.  So, we set a moderately
        # large epsilon, 500.
        budget = PrivacyBudget(epsilon=500, delta=0)
        simulator = LocalDpSimulator(
            data_set,
            params,
            PrivacyTracker(),
        )
        rp = simulator.simulated_reach_by_spend(spends=spends, budget=budget)
        self.assertAlmostEqual(rp.reach(1)[0], expected_reach, delta=0.01)

    def test_simulated_reach_by_spend_with_privacy(self):
        pdf1 = PublisherData([(i, i * 0.01) for i in range(2_000)], "pdf1")
        pdf2 = PublisherData(
            [(i, (i - 10_000) * 0.01) for i in range(1_000, 3_000)], "pdf2"
        )
        data_set = DataSet([pdf1, pdf2], "test")
        params = SystemParameters(
            campaign_spend_fractions=[1, 1, 1],
            liquid_legions=LiquidLegionsParameters(
                decay_rate=1,
                sketch_size=10_000,
                random_seed=0,
            ),
            generator=np.random.default_rng(0),
        )
        # Set a large epsilon, so practically zero noise.
        # Minor note: due an math.exp(epsilon) expression in the sketch codes,
        # a too large epsilon will introduce overflow.  So, we set a moderately
        # large epsilon, 500.
        budget = PrivacyBudget(epsilon=1, delta=0)
        simulator = LocalDpSimulator(
            data_set,
            params,
            PrivacyTracker(),
        )
        rp = simulator.simulated_reach_by_spend(spends=[200, 200], budget=budget)
        expected = 3000
        # With relatively small reach in the test, check if the relative error
        # is within 20%.
        self.assertAlmostEqual(rp.reach(1)[0] / expected, 1, delta=0.2)
        rp = simulator.simulated_reach_by_spend(spends=[200, 100], budget=budget)
        expected = 2500
        self.assertAlmostEqual(rp.reach(1)[0] / expected, 1, delta=0.2)


if __name__ == "__main__":
    absltest.main()
