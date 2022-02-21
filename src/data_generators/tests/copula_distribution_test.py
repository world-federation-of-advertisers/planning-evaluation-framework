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
"""Tests for copula_distribution.py."""

from absl.testing import absltest
import numpy as np


from wfa_planning_evaluation_framework.data_generators.copula_distribution import (
    IndependentCopulaDistribution,
)


class IndependentCopulaDistributionTest(absltest.TestCase):
    def test_pairwise_convolution(self):
        this_pmf = np.array([0.2, 0.7, 0.1])
        that_pmf = np.array([0.4, 0.3, 0.3])
        res = IndependentCopulaDistribution.pairwise_convolution(this_pmf, that_pmf)
        expected = np.array([0.08, 0.34, 0.58])
        np.testing.assert_almost_equal(res, expected, 3)

    def test_convolution(self):
        pmf0 = np.array([0.2, 0.7, 0.1])
        pmf1 = np.array([0.4, 0.3, 0.3])
        pmf2 = np.array([0.5, 0.4, 0.1])
        res = IndependentCopulaDistribution.convolution([pmf0, pmf1, pmf2])
        expected = np.array([0.04, 0.202, 0.758])
        np.testing.assert_almost_equal(res, expected, 3)

    def test_estimate_campaign_pmf(self):
        pmf0 = np.array([0.2, 0.7, 0.1])
        pmf1 = np.array([0.4, 0.3, 0.3])
        pmf2 = np.array([0.5, 0.4, 0.1])
        instance = IndependentCopulaDistribution([pmf0, pmf1, pmf2])
        res = instance.estimate_campaign_pmf(np.array([1, 1, 1]))
        expected = np.array([0.04, 0.202, 0.758])
        np.testing.assert_almost_equal(res, expected, 3)

        pmf0 = np.array([0.2, 0.7, 0.1])
        pmf1 = np.array([0.4, 0.3, 0.3])
        pmf2 = np.array([0.5, 0.4, 0.1])
        instance = IndependentCopulaDistribution([pmf0, pmf1, pmf2])
        res = instance.estimate_campaign_pmf(np.array([1, 1, 0]))
        expected = np.array([0.08, 0.34, 0.58])
        np.testing.assert_almost_equal(res, expected, 3)

        pmf0 = np.array([0.2, 0.7, 0.1])
        pmf1 = np.array([0.4, 0.3, 0.3])
        pmf2 = np.array([0.5, 0.4, 0.1])
        instance = IndependentCopulaDistribution([pmf0, pmf1, pmf2])
        res = instance.estimate_campaign_pmf(np.array([0.5, 0.5, 0.5]))
        expected = np.array([0.261, 0.396, 0.343])
        np.testing.assert_almost_equal(res, expected, 3)


if __name__ == "__main__":
    absltest.main()
