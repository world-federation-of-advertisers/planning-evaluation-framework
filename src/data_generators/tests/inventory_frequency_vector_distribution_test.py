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
"""Tests for inventory_frequency_vector_distribution.py."""

from absl.testing import absltest
import numpy as np

from wfa_planning_evaluation_framework.data_generators.inventory_frequency_vector_distribution import (
    InventoryFrequencyVectorDistribution,
)


class InventoryFrequencyVectorDistributionTest(absltest.TestCase):
    def test_truncate_histogram(self):
        hist = np.array([5, 6, 2, 1, 2])
        res = InventoryFrequencyVectorDistribution.truncate_histogram(hist, 2)
        expected = np.array([5, 6, 5])
        np.testing.assert_equal(res, expected)
        res = InventoryFrequencyVectorDistribution.truncate_histogram(hist, 7)
        expected = np.array([5, 6, 2, 1, 2])
        np.testing.assert_equal(res, expected)

    def test_estimate_histogram_after_single_direction_sampling(self):
        hist = np.array([0, 0, 10])
        res = InventoryFrequencyVectorDistribution.estimate_histogram_after_single_direction_sampling(
            histogram=hist, impression_fraction=0
        )
        expected = np.array([10, 0, 0])
        np.testing.assert_equal(res, expected)
        res = InventoryFrequencyVectorDistribution.estimate_histogram_after_single_direction_sampling(
            histogram=hist, impression_fraction=0.5
        )
        expected = np.array([2.5, 5, 2.5])
        np.testing.assert_equal(res, expected)
        res = InventoryFrequencyVectorDistribution.estimate_histogram_after_single_direction_sampling(
            histogram=hist, impression_fraction=1
        )
        expected = np.array([0, 0, 10])
        np.testing.assert_equal(res, expected)

    def test_visualize_single_pub(self):
        try:
            InventoryFrequencyVectorDistribution.visualize_single_pub(
                np.array([0.4, 0.2, 0.3, 0.1])
            )
        except:
            self.fail()

    def visualize_two_pubs(self):
        try:
            InventoryFrequencyVectorDistribution.visualize_two_pubs(
                np.array([[1, 4, 5, 3], [2, 8, 10, 2], [0, 6, 2, 5]])
            )
        except:
            self.fail()


if __name__ == "__main__":
    absltest.main()
