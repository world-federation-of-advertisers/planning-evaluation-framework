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


from wfa_planning_evaluation_framework.data_generators.dirac_mixture_distribution import (
    DiracMixtureDistribution,
)


class DiracMixtureDistributionTest(absltest.TestCase):
    def test_estimate_campaign_pmf(self):
        component_matrix = np.array([[1, 2, 3], [2, 0, 1]])
        weights = np.array([0.5, 0.5])
        instance = DiracMixtureDistribution(
            component_matrix=component_matrix, weights=weights
        )
        fractions = np.array([0, 0, 0])
        res = instance.estimate_campaign_pmf(
            campaign_impression_fractions=fractions, max_freq=2
        )
        expected = np.array([1, 0, 0])
        np.testing.assert_almost_equal(res, expected, 2)

        instance = DiracMixtureDistribution(
            component_matrix=component_matrix, weights=weights
        )
        fractions = np.array([1, 1, 1])
        res = instance.estimate_campaign_pmf(
            campaign_impression_fractions=fractions, max_freq=2
        )
        looked_up_poisson_pmf_a = np.array([0.00248, 0.01487])
        # By manual calculation, the first projected component is 6.
        # Above is the pmf for Poisson(6).
        looked_up_poisson_pmf_b = np.array([0.04979, 0.14936])
        # By manual calculation, the second projected component is 3.
        expected = looked_up_poisson_pmf_a * 0.5 + looked_up_poisson_pmf_b * 0.5
        expected = np.concatenate((expected, [1 - sum(expected)]))
        np.testing.assert_almost_equal(res, expected, 2)

        fractions = np.array([0.6, 0.2, 0.4])
        res = instance.estimate_campaign_pmf(
            campaign_impression_fractions=fractions, max_freq=2
        )
        looked_up_poisson_pmf_a = np.array([0.11080, 0.24377])
        # By manual calculation, the first projected component is 2.2.
        looked_up_poisson_pmf_b = np.array([0.20190, 0.32303])
        # By manual calculation, the first projected component is 1.6.
        expected = looked_up_poisson_pmf_a * 0.5 + looked_up_poisson_pmf_b * 0.5
        expected = np.concatenate((expected, [1 - sum(expected)]))
        np.testing.assert_almost_equal(res, expected, 2)


if __name__ == "__main__":
    absltest.main()
