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
    def test_obtain_local_dp_sketches_with_no_noise(self):
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
        simulator = LocalDpSimulator(
            data_set,
            params,
            PrivacyTracker(),
        )
        # Set a large epsilon, so practically zero noise.
        # Minor note: due an math.exp(epsilon) expression in the sketch
        # codes, a too large epsilon will introduce overflow.  So, we set
        # a moderately large epsilon, 500.
        budget = PrivacyBudget(epsilon=500, delta=0)
        simulator.obtain_local_dp_sketches(budget)
        # Since sketch_size = 10_000 >> reach, the number of non-empty
        # registers is equal to the reach with nearly 100% chance.
        self.assertEqual(sum(simulator.local_dp_sketches[0].sketch), 3)
        self.assertEqual(sum(simulator.local_dp_sketches[1].sketch), 2)
        self.assertEqual(sum(simulator.local_dp_sketches[2].sketch), 3)

    def test_obtain_local_dp_sketches_with_noise(self):
        pdfs = [
            PublisherData([(1, 0.01)], "pdf1"),
            PublisherData([(i, i * 0.01) for i in range(10_000)], "pdf2"),
        ]
        data_set = DataSet(pdfs, "test")
        params = SystemParameters(
            campaign_spend_fractions=[1, 1],
            liquid_legions=LiquidLegionsParameters(
                decay_rate=1,
                sketch_size=100,
                random_seed=0,
            ),
            generator=np.random.default_rng(0),
        )
        simulator = LocalDpSimulator(
            data_set,
            params,
            PrivacyTracker(),
        )
        # epsilon = ln(3) means the blip probability is 1 / 4.
        budget = PrivacyBudget(epsilon=np.log(3), delta=0)
        simulator.obtain_local_dp_sketches(budget)
        # pdf1 has almost zero reach.  After blipping, approximately
        # 1 / 4 registers are non empty.
        self.assertAlmostEqual(
            sum(simulator.local_dp_sketches[0].sketch) / 100, 1 / 4, delta=0.05
        )
        # pdf2 has reach = 10_000 >> 100 = sketch_size.  So, the raw
        # sketch is full.  After blipping, approximately 3 / 4
        # registers are non empty.
        self.assertAlmostEqual(
            sum(simulator.local_dp_sketches[1].sketch) / 100, 3 / 4, delta=0.05
        )

    # The publisher_count, campaign_spends, max_spends, true_reach_by_spend
    # methods in LocalDpSimulator are copy-pasted from the same methods
    # in HaloSimulator, and thus will not be tested again.

    def test_find_subset(self):
        pdfs = [
            PublisherData([(i, i) for i in range(1, 11)], "pdf1"),
            PublisherData([(i, i) for i in range(1, 21)], "pdf2"),
            PublisherData([(i, i) for i in range(1, 31)], "pdf3"),
        ]
        data_set = DataSet(pdfs, "test")
        params = SystemParameters(
            campaign_spend_fractions=[0.5, 0.5, 0.5],
            liquid_legions=LiquidLegionsParameters(
                decay_rate=1,
                sketch_size=10_000,
                random_seed=0,
            ),
            generator=np.random.default_rng(0),
        )
        simulator = LocalDpSimulator(
            data_set,
            params,
            PrivacyTracker(),
        )
        self.assertTrue(simulator.find_subset([6, 10, 15]) is None)
        self.assertTrue(simulator.find_subset([0, 0, 7]) is None)
        self.assertItemsEqual(simulator.find_subset([0, 10, 0]), {1})
        self.assertItemsEqual(simulator.find_subset([5, 0, 15]), {0, 2})
        self.assertItemsEqual(simulator.find_subset([5, 10, 15]), {0, 1, 2})

    def test_simulated_reach_by_spend_with_no_noise(self):
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
        simulator = LocalDpSimulator(
            data_set,
            params,
            PrivacyTracker(),
        )
        budget = PrivacyBudget(epsilon=500, delta=0)
        simulator.obtain_local_dp_sketches(budget)
        rp = simulator.simulated_reach_by_spend(spends=[0.05, 0.06, 0.05])
        self.assertAlmostEqual(rp.reach(1), 4, delta=0.01)
        rp = simulator.simulated_reach_by_spend(spends=[0, 0.06, 0])
        self.assertAlmostEqual(rp.reach(1), 2, delta=0.01)
        rp = simulator.simulated_reach_by_spend(spends=[0, 0.06, 0.05])
        self.assertAlmostEqual(rp.reach(1), 3, delta=0.01)
        rp = simulator.simulated_reach_by_spend(spends=[0.3, 0.3, 0.3])
        self.assertTrue(np.isnan(rp.reach(1)))

    def test_simulated_reach_by_spend_with_noise(self):
        pdf1 = PublisherData([(i + 1, i + 1) for i in range(2_000)], "pdf1")
        pdf2 = PublisherData(
            [(i + 1, i + 1 - 1000) for i in range(1_000, 3_000)], "pdf2"
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
        simulator = LocalDpSimulator(
            data_set,
            params,
            PrivacyTracker(),
        )
        # epsilon = ln(3) means blip probaility = 1 / 4.  We analytically
        # obtained that the 95% quantile of relative error is smaller than 10%.
        budget = PrivacyBudget(epsilon=np.log(3), delta=0)
        simulator.obtain_local_dp_sketches(budget)
        rp = simulator.simulated_reach_by_spend(spends=[2000, 2000])
        print(rp.reach(1))
        self.assertAlmostEqual(rp.reach(1) / 3000, 1, delta=0.1)
        rp = simulator.simulated_reach_by_spend(spends=[2000, 0])
        print(rp.reach(1))
        self.assertAlmostEqual(rp.reach(1) / 2000, 1, delta=0.1)


if __name__ == "__main__":
    absltest.main()
