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
"""Tests for gamma_poisson_model.py."""

from absl.testing import absltest
from unittest.mock import patch

import numpy as np
import scipy.stats
import scipy.integrate

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.kinflated_gamma_poisson_model import (
    KInflatedGammaPoissonDistribution,
    KInflatedGammaPoissonModel,
)
from wfa_planning_evaluation_framework.models.gamma_poisson_model import (
    GammaPoissonModel,
)
from wfa_planning_evaluation_framework.models.goerg_model import GoergModel
from wfa_planning_evaluation_framework.data_generators.heterogeneous_impression_generator import HeterogeneousImpressionGenerator
from wfa_planning_evaluation_framework.data_generators.heavy_tailed_impression_generator import HeavyTailedImpressionGenerator
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import FixedPriceGenerator
from wfa_planning_evaluation_framework.data_generators.publisher_data import PublisherData
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet

def gamma_poisson_pmf(k, alpha, beta):
    """Computes Gamma-Poisson density through integration."""
    def gamma_poisson_integrand(k, mu, alpha, beta):
        return scipy.stats.poisson.pmf(k, mu) * scipy.stats.gamma.pdf(
            mu, alpha, scale=1.0 / beta
        )

    return scipy.integrate.quad(
        lambda x: gamma_poisson_integrand(k, x, alpha, beta), 0.0, np.Inf
    )[0]

class KInflatedGammaPoissonModelTest(absltest.TestCase):
    
    def test_pmf_no_inflation(self):
        dist1 = KInflatedGammaPoissonDistribution(1.0, 1.0, [])
        self.assertAlmostEqual(dist1.pmf(1), gamma_poisson_pmf(0, 1.0, 1.0))
        self.assertAlmostEqual(dist1.pmf(2), gamma_poisson_pmf(1, 1.0, 1.0))

        dist2 = KInflatedGammaPoissonDistribution(3.0, 1.0, [])
        self.assertAlmostEqual(dist2.pmf(2), gamma_poisson_pmf(1, 3.0, 1.0))

        dist3 = KInflatedGammaPoissonDistribution(1.0, 4.0, [])
        self.assertAlmostEqual(dist3.pmf(2), gamma_poisson_pmf(1, 1.0, 4.0))

    def test_pmf_with_partial_inflation(self):
        dist1 = KInflatedGammaPoissonDistribution(1.0, 1.0, [0.5])
        self.assertAlmostEqual(dist1.pmf(1), 0.5)

        scale_factor = 0.5 / (1.0 - gamma_poisson_pmf(0, 1.0, 1.0))
        expected = scale_factor * gamma_poisson_pmf(1, 1.0, 1.0)
        self.assertAlmostEqual(dist1.pmf(2), expected)

        dist2 = KInflatedGammaPoissonDistribution(3.0, 1.0, [0.5])
        scale_factor = 0.5 / (1.0 - gamma_poisson_pmf(0, 3.0, 1.0))
        expected = scale_factor * gamma_poisson_pmf(1, 3.0, 1.0)
        self.assertAlmostEqual(dist2.pmf(2), expected)

    def test_pmf_with_full_inflation(self):
        dist = KInflatedGammaPoissonDistribution(1.0, 1.0, [0.75, 0.25])
        self.assertAlmostEqual(dist.pmf(1), 0.75)
        self.assertAlmostEqual(dist.pmf(2), 0.25)
        self.assertAlmostEqual(dist.pmf(3), 0.0)

    def test_knreach_no_inflation(self):
        dist = KInflatedGammaPoissonDistribution(1.0, 1.0, [])
        self.assertAlmostEqual(dist.knreach(1, 1, 0.5), 0.25)
        self.assertAlmostEqual(dist.knreach(2, 1, 0.5), 0)
        self.assertAlmostEqual(dist.knreach(1, 1, 0.5), 1.0 / 4.0)
        self.assertAlmostEqual(dist.knreach(1, np.array([1]), 0.5)[0], 1 / 4)
        self.assertAlmostEqual(dist.knreach(1, np.array([1, 2]), 0.5)[0], 1 / 4)
        self.assertAlmostEqual(dist.knreach(1, np.array([1, 2]), 0.5)[1], 1 / 8)
        self.assertAlmostEqual(
            dist.knreach(1, 3, 1/3), 3 * (1 / 3) * (2 / 3) ** 2 * (1 / 8)
        )

    def test_kreach_no_inflation(self):
        dist = KInflatedGammaPoissonDistribution(1.0, 1.0, [])
        self.assertAlmostEqual(dist.kreach([0], 0.5)[0], 1 / 3)
        self.assertAlmostEqual(dist.kreach([0], 1/3)[0], 1 / 2)
        self.assertAlmostEqual(dist.kreach([1], 1/3)[0], 3 / 8)
        self.assertAlmostEqual(dist.kreach([0, 1, 2], 1/3)[0], 1 / 2)
        self.assertAlmostEqual(dist.kreach([0, 1, 2], 1/3)[1], 3 / 8)
        self.assertAlmostEqual(dist.kreach([0, 1, 2], 1/3)[2], 3 / 32)

    def test_expected_value(self):
        dist1 = KInflatedGammaPoissonDistribution(1.0, 1.0, [])
        self.assertAlmostEqual(dist1.expected_value(), 2.0)

        dist2 = KInflatedGammaPoissonDistribution(2.0, 1.0, [])
        self.assertAlmostEqual(dist2.expected_value(), 3.0)

        dist3 = KInflatedGammaPoissonDistribution(1.0, 2.0, [])
        self.assertAlmostEqual(dist3.expected_value(), 3.0)

        dist4 = KInflatedGammaPoissonDistribution(1.0, 2.0, [1.0])
        self.assertAlmostEqual(dist4.expected_value(), 1.0)
        
        dist5 = KInflatedGammaPoissonDistribution(1.0, 2.0, [0.5, 0.5])
        self.assertAlmostEqual(dist5.expected_value(), 1.5)

    def test_exponential_poisson_reach(self):
        kgpm = KInflatedGammaPoissonModel([ReachPoint([100], [100])])
        self.assertAlmostEqual(kgpm._exponential_poisson_reach(20000, 10000, 3.0), 8000.)
        self.assertAlmostEqual(kgpm._exponential_poisson_reach(19971, 12560, 26.19), 7888.78, delta=1)

    def test_exponential_poisson_beta(self):
        kgpm = KInflatedGammaPoissonModel([ReachPoint([100], [100])])
        self.assertAlmostEqual(kgpm._exponential_poisson_beta(30000, 10000, 10000), 2.0)
        self.assertAlmostEqual(kgpm._exponential_poisson_beta(20000, 10000, 8000), 3.0)

    def test_exponential_poisson_N_from_beta(self):
        kgpm = KInflatedGammaPoissonModel([ReachPoint([100], [100])])
        self.assertAlmostEqual(kgpm._exponential_poisson_N_from_beta(19971, 7992, 2.41), 9416., delta=1)
        self.assertAlmostEqual(kgpm._exponential_poisson_N_from_beta(30000, 10000, 2), 10000.)

    def test_exponential_poisson_N(self):
        kgpm = KInflatedGammaPoissonModel([ReachPoint([100], [100])])
        self.assertAlmostEqual(kgpm._exponential_poisson_N(19971, 7992), 9769, delta=1)
        self.assertAlmostEqual(kgpm._exponential_poisson_N(20000, 10000), 13333.33, delta=1)

    def test_fit_exponential_poisson_model(self):
        p1 = ReachPoint([20000], [10000])
        kgpm1 = KInflatedGammaPoissonModel([p1])
        N1, dist1 = kgpm1._fit_exponential_poisson_model(p1)
        self.assertAlmostEqual(N1, 13333.33, delta=1)
        self.assertAlmostEqual(dist1._alpha, 1.0)
        self.assertAlmostEqual(dist1._beta, 2.0, delta=0.1)

        p2 = ReachPoint([19971], [7993, 4815, 2914, 1759, 1011, 604, 355, 214, 122, 75])
        kgpm2 = KInflatedGammaPoissonModel([p2])
        N2, dist2 = kgpm1._fit_exponential_poisson_model(p2)
        self.assertAlmostEqual(N2, 8384, delta=100)
        self.assertAlmostEqual(dist2._alpha, 1.0)
        self.assertAlmostEqual(dist2._beta, 1.696, delta=0.1)

    def test_fit_frequency_one(self):
        data = HeterogeneousImpressionGenerator(10000, gamma_shape=1., gamma_scale=3)()
        publisher = PublisherData(FixedPriceGenerator(0.1)(data))
        dataset = DataSet([publisher], f"Exponential-poisson")
        spend_fraction = 0.5
        spend = dataset._data[0].max_spend * spend_fraction
        point = dataset.reach_by_spend([spend], max_frequency=1)
        gm = GoergModel([point])
        gm_reach = gm.by_spend([spend]).reach()
        kgpm = KInflatedGammaPoissonModel([point])
        kgpm._fit()
        kgpm_reach = kgpm.by_spend([spend]).reach()
        self.assertAlmostEqual(gm_reach, kgpm_reach)

    def test_fit_exponential_poisson(self):
        data = HeterogeneousImpressionGenerator(10000, gamma_shape=1., gamma_scale=3)()
        publisher = PublisherData(FixedPriceGenerator(0.1)(data))
        dataset = DataSet([publisher], f"Exponential-poisson")
        spend_fraction = 0.5
        spend = dataset._data[0].max_spend * spend_fraction
        point = dataset.reach_by_spend([spend])
        gm = GoergModel([point])
        gm_reach = gm.by_spend([spend]).reach()
        kgpm = KInflatedGammaPoissonModel([point])
        kgpm.print_fit_header()
        kgpm.print_fit("true", 0.0, 10000, 1.0, 3.0, [])
        kgpm._fit()
        kgpm_reach = kgpm.by_spend([spend]).reach()
        self.assertAlmostEqual(gm_reach, kgpm_reach, delta=100)
        
    def test_fit_gamma_poisson(self):
        # mean = 5, var = 6
        N, alpha, beta = 10000, 8, 0.5
        data = HeterogeneousImpressionGenerator(10000, gamma_shape=alpha, gamma_scale=beta)()
        publisher = PublisherData(FixedPriceGenerator(0.1)(data))
        dataset = DataSet([publisher], f"Exponential-poisson")
        spend_fraction = 0.5
        spend = dataset._data[0].max_spend * spend_fraction
        point = dataset.reach_by_spend([spend])
        gm = GoergModel([point])
        gm_reach = gm.by_spend([spend]).reach()
        kgpm = KInflatedGammaPoissonModel([point])
        kgpm.print_fit_header()
        kgpm.print_fit("true", 0.0, N, alpha, beta, [])
        kgpm._fit()
        kgpm_reach = kgpm.by_spend([spend]).reach()
        self.assertAlmostEqual(gm_reach, kgpm_reach, delta=100)
        
    def test_fit_zeta(self):
        # mean = 5, var = 6
        N, zeta_s = 10000, 2.141628
        data = HeavyTailedImpressionGenerator(10000, zeta_s)()
        publisher = PublisherData(FixedPriceGenerator(0.1)(data))
        dataset = DataSet([publisher], f"xxx")
        spend_fraction = 0.5
        spend = dataset._data[0].max_spend * spend_fraction
        point = dataset.reach_by_spend([spend])
        gm = GoergModel([point])
        gm_reach = gm.by_spend([spend]).reach()
        kgpm = KInflatedGammaPoissonModel([point])
        kgpm.print_fit_header()
        kgpm.print_fit("true", 0.0, N, 0.0, 0.0, [])
        kgpm._fit()
        kgpm_reach = kgpm.by_spend([spend]).reach()
        self.assertAlmostEqual(gm_reach, kgpm_reach, delta=100)
        
    def test_fit_histogram_chi2_distance(self):
        # The following point corresponds to
        #  I, Imax, N, alpha0, beta0 = 5000, 30000, 10000, 1.0, 2.0
        # This represents a distribution with mean = 3 and variance = 6.
        # h_actual = [2812, 703, 175, 43, 10, 2, 0, 0, 0, 0]
        # kplus_actual = [3745, 933, 230, 55, 12, 2, 0, 0, 0, 0]
        # rp = ReachPoint([5000], kplus_actual)
        # gpm = GammaPoissonModel([rp])
        # Imax, N, alpha, beta = gpm._fit_histogram_chi2_distance(rp, nstarting_points=1)
        # h_predicted = gpm._expected_histogram(5000, Imax, N, alpha, beta)
        # for i in range(len(h_actual)):
        #     self.assertTrue(
        #         (h_actual[i] - h_predicted[i]) ** 2 / h_predicted[i] < 1,
        #         f"Discrepancy found at position {i}. "
        #         f"Got {h_predicted[i]} Expected {h_actual[i]}",
        #     )
        pass

    def test_fit_histogram_fixed_N(self):
        # The following point corresponds to
        #  I, Imax, N, alpha0, beta0 = 15000, 30000, 10000, 1.0, 2.0
        # This represents a distribution with mean = 3 and variance = 6.
        # h_actual = [3749, 1875, 937, 468, 234, 117, 58, 29, 14, 7]
        # kplus_actual = [7488, 3739, 1864, 927, 459, 225, 108, 50, 21, 7]
        # rp = ReachPoint([15000], kplus_actual)
        # gpm = GammaPoissonModel([rp])
        # Imax, alpha, beta = gpm._fit_histogram_fixed_N(rp, 10000)
        # h_predicted = gpm._expected_histogram(15000, Imax, 10000, alpha, beta)
        # for i in range(len(h_actual)):
        #     self.assertTrue(
        #         (h_actual[i] - h_predicted[i]) ** 2 / h_predicted[i] < 1,
        #         f"Discrepancy found at position {i}. "
        #         f"Got {h_predicted[i]} Expected {h_actual[i]}",
        #     )
        pass

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.GammaPoissonModel._fit_histogram_chi2_distance"
    )
    def test_fit_variable_N(self, mock_gamma_poisson_model):
        # mock_gamma_poisson_model.return_value = (30000, 10000, 1.0, 2.0)
        # h_actual = [2853, 813, 230, 64, 17, 4, 1, 0, 0, 0]
        # rp = ReachPoint([4000], h_actual)
        # gpm = GammaPoissonModel([rp])
        # gpm._fit()
        # self.assertAlmostEqual(gpm._max_reach, 10000, delta=1)
        # self.assertAlmostEqual(gpm._alpha, 1.0, delta=0.01)
        pass

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.GammaPoissonModel._fit_histogram_fixed_N"
    )
    def test_by_impressions(self, mock_gamma_poisson_model):
        # mock_gamma_poisson_model.return_value = (25000, 5.0, 2.0)
        # # Imax = 25000, N = 10000, alpha = 5, beta = 2
        # h_training = [8124, 5464, 3191, 1679, 815, 371, 159, 64, 23, 6, 0]
        # rp = ReachPoint([20000], h_training, [200.0])
        # gpm = GammaPoissonModel([rp], max_reach=10000)
        # gpm._fit()
        # rp = gpm.by_impressions([10000], max_frequency=5)
        # h_expected = np.array([9682, 8765, 7353, 5750, 4233])
        # h_actual = np.array([int(rp.reach(i)) for i in range(1, 6)])
        # total_error = np.sum((h_expected - h_actual) ** 2 / h_expected)
        # self.assertAlmostEqual(rp.spends[0], 100.0)
        # for i in range(len(h_actual)):
        #     self.assertTrue(
        #         (h_actual[i] - h_expected[i]) ** 2 / h_actual[i] < 0.1,
        #         f"Discrepancy found at position {i}. "
        #         f"Got {h_actual[i]} Expected {h_expected[i]}",
        #     )
        pass

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.GammaPoissonModel._fit_histogram_fixed_N"
    )
    def test_by_spend(self, mock_gamma_poisson_model):
        # mock_gamma_poisson_model.return_value = (25000, 5.0, 2.0)
        # # Imax = 25000, N = 10000, alpha = 5, beta = 2
        # h_training = [8124, 5464, 3191, 1679, 815, 371, 159, 64, 23, 6, 0]
        # rp = ReachPoint([20000], h_training, [200.0])
        # gpm = GammaPoissonModel([rp], max_reach=10000)
        # gpm._fit()
        # rp = gpm.by_spend([100.0], max_frequency=5)
        # h_expected = np.array([9682, 8765, 7353, 5750, 4233])
        # h_actual = np.array([int(rp.reach(i)) for i in range(1, 6)])
        # total_error = np.sum((h_expected - h_actual) ** 2 / h_expected)
        # self.assertTrue(total_error < 1)
        # self.assertAlmostEqual(rp.impressions[0], 10000.0)
        # self.assertAlmostEqual(rp.spends[0], 100.0)
        # for i in range(len(h_actual)):
        #     self.assertTrue(
        #         (h_actual[i] - h_expected[i]) ** 2 / h_actual[i] < 0.1,
        #         f"Discrepancy found at position {i}. "
        #         f"Got {h_actual[i]} Expected {h_expected[i]}",
        #     )
        pass


if __name__ == "__main__":
    absltest.main()
