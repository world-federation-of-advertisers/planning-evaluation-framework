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
"""Tests for goerg_model.py."""

from absl.testing import absltest
import numpy as np

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.mdi_model import MDITreeNode
from wfa_planning_evaluation_framework.models.mdi_model import MDITreeLeafNode
from wfa_planning_evaluation_framework.models.mdi_model import MDITreeInternalNode

from wfa_planning_evaluation_framework.models.goerg_model import GoergModel


class MDIModelTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        reach0 = [6000, 2400, 1000, 375, 150, 30, 6, 0]
        impressions0 = [10000]
        spend0 = [1000]
        cls.point0 = ReachPoint(impressions0, reach0, spend0)
        cls.curve0 = GoergModel([cls.point0])
        cls.node0 = MDITreeLeafNode(cls.curve0, 20000, 3000, 5)
        cls.node1 = MDITreeLeafNode(cls.curve0, 20000, 3000, 5)

        impressions01 = [10000, 10000]
        spend01 = [1000, 1000]
        reach01 = [10219, 5172, 2455, 1144, 631]
        cls.point01 = ReachPoint(impressions01, reach01, spend01)
        cls.node01 = MDITreeInternalNode(cls.node0, cls.node1, 20000, 5, 3)
        
    def test_leaf_reach(self):
        np.testing.assert_almost_equal(self.node0._freq,
                                       [0.5, 0.167, 0.111, 0.074, 0.049, 0.099],
                                       3)
        np.testing.assert_equal(self.node0._publisher_count, 1)
        np.testing.assert_equal(self.node0._impression_count_vector, [30000])
        np.testing.assert_equal(self.node0._max_spend_vector, [3000])

    def test_leaf_inventory_frequencies(self):
        np.testing.assert_almost_equal(self.node0.inventory_frequencies(),
                                       [0.5, 0.167, 0.111, 0.074, 0.049, 0.099],
                                       3)
        np.testing.assert_almost_equal(self.node0.inventory_frequencies(2),
                                       [0.5, 0.167, 0.111],
                                       3)
        np.testing.assert_almost_equal(self.node0.inventory_frequencies(7),
                                       [0.5, 0.167, 0.111, 0.074, 0.049, 0.099, 0., 0.],
                                       3)

    def test_leaf_inventory_histogram(self):
        np.testing.assert_almost_equal(self.node0.inventory_histogram(),
                                       [10000, 3333, 2222, 1481, 988, 1975],
                                       0)
        np.testing.assert_almost_equal(self.node0.inventory_histogram(2),
                                       [10000, 3333, 2222],
                                       0)
        np.testing.assert_almost_equal(self.node0.inventory_histogram(7),
                                       [10000, 3333, 2222, 1481, 988, 1975, 0, 0],
                                       0)

    def test_leaf_campaign_frequencies(self):
        np.testing.assert_almost_equal(self.node0.campaign_frequencies([1.0], 5),
                                       [0.5, 0.167, 0.111, 0.074, 0.049, 0.099],
                                       3)

        np.testing.assert_almost_equal(self.node0.campaign_frequencies([1.0/3.0], 5),
                                       [0.7, 0.19, 0.076, 0.024, 0.005, 0.004],
                                       2)
        np.testing.assert_almost_equal(self.node0.campaign_frequencies([1.0/3.0], 2),
                                       [0.7, 0.19, 0.076],
                                       2)
        np.testing.assert_almost_equal(self.node0.campaign_frequencies([1.0/3.0], 7),
                                       [0.7, 0.19, 0.076, 0.024, 0.005, 0.004, 0., 0.],
                                       2)

    def test_leaf_campaign_histogram(self):
        np.testing.assert_almost_equal(self.node0.campaign_histogram([1.0/3.0], 5),
                                       [14104., 3798., 1519., 478., 93., 8.],
                                       0)
        np.testing.assert_almost_equal(self.node0.campaign_histogram([1.0/3.0], 3),
                                       [14104., 3798., 1519., 478.],
                                       0)
        np.testing.assert_almost_equal(self.node0.campaign_histogram([1.0/3.0], 7),
                                       [14104., 3798., 1519., 478., 93., 8., 0., 0.],
                                       0)
        
    def test_leaf_by_impression_probability(self):
        rp0 = self.node0.by_impression_probability([1.0], 3)
        np.testing.assert_almost_equal(rp0.impressions, [30000.], 0)
        np.testing.assert_almost_equal(rp0._kplus_reaches, [10000, 6667., 4444.], 0)
        np.testing.assert_almost_equal(rp0.spends, [3000.], 0)

        rp1 = self.node0.by_impression_probability([1.0/3.0], 5)
        np.testing.assert_almost_equal(rp1.impressions, [10000.], 0)
        # Compare the following values to the reach values given in the definition
        # of the original reach point reach0 in setUpClass.  The differences can
        # be attributed to (1) approximation introduced by modeling, (2) the fact
        # that only a small number of frequencies are used.  For reference, here
        # are the estimated reach values from the fitted reach curve:
        #  [(6000, 2400, 960, 384, 154)]
        np.testing.assert_almost_equal(rp1._kplus_reaches,
                                       [5896., 2098., 579., 101., 8.], 0)
        np.testing.assert_almost_equal(rp1.spends, [1000.], 0)

        rp2 = self.node0.by_impression_probability([1.0/3.0], 7)
        np.testing.assert_almost_equal(rp2.impressions, [10000.], 0)
        np.testing.assert_almost_equal(rp2._kplus_reaches,
                                       [5896., 2098., 579., 101., 8., 0., 0.], 0)
        np.testing.assert_almost_equal(rp2.spends, [1000.], 0)

    def test_leaf_by_impressions(self):
        rp1 = self.node0.by_impressions([10000], 5)
        np.testing.assert_almost_equal(rp1.impressions, [10000.], 0)
        np.testing.assert_almost_equal(rp1._kplus_reaches,
                                       [5896., 2098., 579., 101., 8.], 0)
        np.testing.assert_almost_equal(rp1.spends, [1000.], 0)

    def test_leaf_by_spend(self):
        rp1 = self.node0.by_spend([1000.], 5)
        np.testing.assert_almost_equal(rp1.impressions, [10000.], 0)
        np.testing.assert_almost_equal(rp1._kplus_reaches,
                                       [5896., 2098., 579., 101., 8.], 0)
        np.testing.assert_almost_equal(rp1.spends, [1000.], 0)

    def test_leaf_impression_probability(self):
        expected = 0.5
        actual = self.node0.impression_probability([0.5])
        np.testing.assert_almost_equal(actual, expected)

    def test_inventory_frequencies_under_independence(self):
        expected1 = np.array([0.25, 0.16, 0.14, 0.1, 0.07, 0.05])
        actual1 = self.node01.inventory_frequencies()
        np.testing.assert_almost_equal(actual1, expected1, 2)

        expected2 = np.array([0.25, 0.16, 0.14, 0.1])
        actual2 = self.node01.inventory_frequencies(3)
        np.testing.assert_almost_equal(actual2, expected2, 2)

        expected3 = np.array([0.25, 0.16, 0.14, 0.1, 0.07, 0.05, 0.1, 0])
        actual3 = self.node01.inventory_frequencies(7)
        np.testing.assert_almost_equal(actual3, expected3, 2)

    def test_campaign_frequencies_under_independence(self):
        expected1 = np.array([0.25, 0.17, 0.14, 0.11, 0.09, 0.25])
        actual1 = self.node01.campaign_frequencies([1.0, 1.0])
        np.testing.assert_almost_equal(actual1, expected1, 2)

        expected2 = np.array([0.25, 0.17, 0.14, 0.44])
        actual2 = self.node01.campaign_frequencies([1.0, 1.0], 3)
        np.testing.assert_almost_equal(actual2, expected2, 2)

        expected3 = np.array([0.25, 0.17, 0.14, 0.11, 0.09, 0.13, 0.05, 0.07])
        actual3 = self.node01.campaign_frequencies([1.0, 1.0], 7)
        np.testing.assert_almost_equal(actual3, expected3, 2)

        expected4 = np.array([0.5, 0.17, 0.11, 0.07, 0.05, 0.1])
        actual4 = self.node01.campaign_frequencies([1.0, 0.0])
        np.testing.assert_almost_equal(actual4, expected4, 2)

        expected5 = np.array([0.5, 0.17, 0.11, 0.07, 0.05, 0.1])
        actual5 = self.node01.campaign_frequencies([0, 1.0])
        np.testing.assert_almost_equal(actual5, expected5, 2)

        expected6 = np.array([0.75, 0.17, 0.05, 0.01, 0.0, 0.0])
        actual6 = self.node01.campaign_frequencies([0.25, 0.0])
        np.testing.assert_almost_equal(actual6, expected6, 2)

        expected7 = np.array([0.63, 0.19, 0.10, 0.05, 0.02, 0.0])
        actual7 = self.node01.campaign_frequencies([0, 0.5])
        np.testing.assert_almost_equal(actual7, expected7, 2)

        expected8 = np.array([0.47, 0.26, 0.15, 0.08, 0.03, 0.01])
        actual8 = self.node01.campaign_frequencies([0.25, 0.5])
        np.testing.assert_almost_equal(actual8, expected8, 2)
        
    def test_by_impression_probability_under_independence(self):
        expected1 = ReachPoint([7500, 15000],
                               [10504.,  5371.,  2427.],
                               [750., 1500.])
        actual1 = self.node01.by_impression_probability([0.25, 0.5], 3)
        np.testing.assert_almost_equal(actual1.impressions, expected1.impressions, 0)
        np.testing.assert_almost_equal(actual1._kplus_reaches, expected1._kplus_reaches, 0)
        np.testing.assert_almost_equal(actual1.spends, expected1.spends, 0)

    def test_by_impressions_under_independence(self):
        expected1 = ReachPoint([7500, 15000],
                               [10504.,  5371.,  2427.],
                               [750., 1500.])
        actual1 = self.node01.by_impressions([7500, 15000], 3)
        np.testing.assert_almost_equal(actual1.impressions, expected1.impressions, 0)
        np.testing.assert_almost_equal(actual1._kplus_reaches, expected1._kplus_reaches, 0)
        np.testing.assert_almost_equal(actual1.spends, expected1.spends, 0)
        
    def test_by_spend_under_independence(self):
        expected1 = ReachPoint([7500, 15000],
                               [10504.,  5371.,  2427.],
                               [750., 1500.])
        actual1 = self.node01.by_spend([750., 1500.], 3)
        np.testing.assert_almost_equal(actual1.impressions, expected1.impressions, 0)
        np.testing.assert_almost_equal(actual1._kplus_reaches, expected1._kplus_reaches, 0)
        np.testing.assert_almost_equal(actual1.spends, expected1.spends, 0)

    def test_impression_probability_under_independence(self):
        expected1 = 0.375
        actual1 = self.node01.impression_probability([0.25, 0.5])
        np.testing.assert_almost_equal(actual1, expected1)

    def no_test_fit_under_independence(self):
        node = MDITreeInternalNode(self.node0, self.node1, 20000, 10, 10)
        node._fit([self.point01], hlambda=0.0)
        expected1 = self.point01.frequencies
        actual1 = node.by_impressions(self.point01.impressions,
                                      max_frequency=5).frequencies
        np.testing.assert_almost_equal(actual1, expected1, -1)

    def test_fit_disjoint_distributions(self):
        impressions = 15000
        point = self.curve0.by_impressions([impressions], max_frequency=5)
        twice_kplus_reaches = [2*k for k in point._kplus_reaches]
        double_point = ReachPoint([impressions, impressions],
                                  twice_kplus_reaches,
                                  [point.spends[0], point.spends[0]])
        node = MDITreeInternalNode(self.node0, self.node1, 20000, 10, 10)
        node._fit([double_point], hlambda=0.0)
        expected = double_point.frequencies
        actual = node.by_impressions(point.impressions,
                                      max_frequency=5).frequencies
        print(f"expected = {expected}")
        print(f"actual = {actual}")
        np.testing.assert_allclose(actual, expected, rtol=0.1) 

       
if __name__ == "__main__":
    absltest.main()
