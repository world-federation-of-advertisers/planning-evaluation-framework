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
from wfa_planning_evaluation_framework.models.gamma_poisson_model import (
    GammaPoissonModel,
)


class GammaPoissonModelTest(absltest.TestCase):
    def test_logpmf(self):
        # The coded implementation of the Gamma-Poisson makes use of the fact
        # that a Gamma-Poisson with parameters (alpha, beta) is equivalent
        # to a negative binomial with parameters (p, r) =
        # (beta / (1 + beta), alpha).  In this test, we compute the
        # Gamma-Poisson directly through numerical integration and compare
        # it to the values computed via the negative binomial.
        def gamma_poisson_integrand(k, mu, alpha, beta):
            return scipy.stats.poisson.pmf(k, mu) * scipy.stats.gamma.pdf(
                mu, alpha, scale=1.0 / beta
            )

        def gamma_poisson_pmf(k, alpha, beta):
            return scipy.integrate.quad(
                lambda x: gamma_poisson_integrand(k, x, alpha, beta), 0.0, np.Inf
            )[0]

        gpm = GammaPoissonModel([ReachPoint([20], [10])])
        self.assertAlmostEqual(
            gpm._logpmf(1, 1.0, 1.0), np.log(gamma_poisson_pmf(0, 1.0, 1.0))
        )

        self.assertAlmostEqual(
            gpm._logpmf(2, 1.0, 1.0), np.log(gamma_poisson_pmf(1, 1.0, 1.0))
        )

        self.assertAlmostEqual(
            gpm._logpmf(2, 3.0, 1.0), np.log(gamma_poisson_pmf(1, 3.0, 1.0))
        )

        self.assertAlmostEqual(
            gpm._logpmf(2, 1.0, 4.0), np.log(gamma_poisson_pmf(1, 1.0, 4.0))
        )

    def test_knreach(self):
        gpm = GammaPoissonModel([ReachPoint([20], [10])])
        self.assertAlmostEqual(gpm._knreach(1, 1, 1, 2, 1.0, 1.0), 0.25)
        self.assertAlmostEqual(gpm._knreach(2, 1, 1, 2, 1.0, 1.0), 0)
        self.assertAlmostEqual(gpm._knreach(1, 1, 1, 2, 1.0, 2.0), 1.0 / 6.0)
        self.assertAlmostEqual(gpm._knreach(1, np.array([1]), 1, 2, 1.0, 1.0)[0], 1 / 4)
        self.assertAlmostEqual(
            gpm._knreach(1, np.array([1, 2]), 1, 2, 1.0, 1.0)[0], 1 / 4
        )
        self.assertAlmostEqual(
            gpm._knreach(1, np.array([1, 2]), 1, 2, 1.0, 1.0)[1], 1 / 8
        )
        self.assertAlmostEqual(
            gpm._knreach(1, 3, 1, 3, 1.0, 1.0), 3 * (1 / 3) * (2 / 3) ** 2 * (1 / 8)
        )

    def test_kreach(self):
        gpm = GammaPoissonModel([ReachPoint([20], [10])])
        self.assertAlmostEqual(gpm._kreach([0], 1, 2, 1, 1)[0], 1 / 3)
        self.assertAlmostEqual(gpm._kreach([0], 1, 3, 1, 1)[0], 1 / 2)
        self.assertAlmostEqual(gpm._kreach([1], 1, 3, 1, 1)[0], 3 / 8)
        self.assertAlmostEqual(gpm._kreach([0, 1, 2], 1, 3, 1, 1)[0], 1 / 2)
        self.assertAlmostEqual(gpm._kreach([0, 1, 2], 1, 3, 1, 1)[1], 3 / 8)
        self.assertAlmostEqual(gpm._kreach([0, 1, 2], 1, 3, 1, 1)[2], 3 / 32)

    def test_expected_impressions(self):
        gpm = GammaPoissonModel([ReachPoint([20], [10])])
        self.assertAlmostEqual(gpm._expected_impressions(1, 1, 1), 2.0)
        self.assertAlmostEqual(gpm._expected_impressions(2, 1, 1), 4.0)
        self.assertAlmostEqual(gpm._expected_impressions(1, 2, 1), 3.0)
        self.assertAlmostEqual(gpm._expected_impressions(1, 1, 2), 3.0)

    def test_expected_histogram(self):
        gpm = GammaPoissonModel([ReachPoint([20], [10])])
        h_actual = gpm._expected_histogram(4, 12, 16, 1, 1, max_freq=3)
        self.assertLen(h_actual, 3)
        self.assertAlmostEqual(h_actual[0], 6)
        self.assertAlmostEqual(h_actual[1], 3 / 2)
        self.assertAlmostEqual(h_actual[2], 3 / 8)

    def test_fit_histogram_chi2_distance(self):
        # The following point corresponds to
        #  I, Imax, N, alpha0, beta0 = 5000, 30000, 10000, 1.0, 2.0
        # This represents a distribution with mean = 3 and variance = 6.
        h_actual = [2812, 703, 175, 43, 10, 2, 0, 0, 0, 0]
        kplus_actual = [3745, 933, 230, 55, 12, 2, 0, 0, 0, 0]
        rp = ReachPoint([5000], kplus_actual)
        gpm = GammaPoissonModel([rp])
        Imax, N, alpha, beta = gpm._fit_histogram_chi2_distance(rp, nstarting_points=1)
        h_predicted = gpm._expected_histogram(5000, Imax, N, alpha, beta)
        for i in range(len(h_actual)):
            self.assertTrue(
                (h_actual[i] - h_predicted[i]) ** 2 / h_predicted[i] < 1,
                f"Discrepancy found at position {i}. "
                f"Got {h_predicted[i]} Expected {h_actual[i]}",
            )

    def test_fit_histogram_fixed_N(self):
        # The following point corresponds to
        #  I, Imax, N, alpha0, beta0 = 15000, 30000, 10000, 1.0, 2.0
        # This represents a distribution with mean = 3 and variance = 6.
        h_actual = [3749, 1875, 937, 468, 234, 117, 58, 29, 14, 7]
        kplus_actual = [7488, 3739, 1864, 927, 459, 225, 108, 50, 21, 7]
        rp = ReachPoint([15000], kplus_actual)
        gpm = GammaPoissonModel([rp])
        Imax, alpha, beta = gpm._fit_histogram_fixed_N(rp, 10000)
        h_predicted = gpm._expected_histogram(15000, Imax, 10000, alpha, beta)
        for i in range(len(h_actual)):
            self.assertTrue(
                (h_actual[i] - h_predicted[i]) ** 2 / h_predicted[i] < 1,
                f"Discrepancy found at position {i}. "
                f"Got {h_predicted[i]} Expected {h_actual[i]}",
            )

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.GammaPoissonModel._fit_histogram_chi2_distance"
    )
    def test_fit_variable_N(self, mock_gamma_poisson_model):
        mock_gamma_poisson_model.return_value = (30000, 10000, 1.0, 2.0)
        h_actual = [2853, 813, 230, 64, 17, 4, 1, 0, 0, 0]
        rp = ReachPoint([4000], h_actual)
        gpm = GammaPoissonModel([rp])
        gpm._fit()
        self.assertAlmostEqual(gpm._max_reach, 10000, delta=1)
        self.assertAlmostEqual(gpm._alpha, 1.0, delta=0.01)

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.GammaPoissonModel._fit_histogram_fixed_N"
    )
    def test_fit_fixed_N(self, mock_gamma_poisson_model):
        mock_gamma_poisson_model.return_value = (30000, 1.0, 2.0)
        # Imax = 4000, N = 10000, alpha = 1, beta = 1
        h_actual = [2853, 813, 230, 64, 17, 4, 1, 0, 0, 0]
        rp = ReachPoint([4000], h_actual)
        gpm = GammaPoissonModel([rp], max_reach=10000)
        gpm._fit()
        self.assertAlmostEqual(gpm._max_reach, 10000, delta=1)
        self.assertAlmostEqual(gpm._alpha, 1.0, delta=0.01)

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.GammaPoissonModel._fit_histogram_fixed_N"
    )
    def test_by_impressions(self, mock_gamma_poisson_model):
        mock_gamma_poisson_model.return_value = (25000, 5.0, 2.0)
        # Imax = 25000, N = 10000, alpha = 5, beta = 2
        h_training = [8124, 5464, 3191, 1679, 815, 371, 159, 64, 23, 6, 0]
        rp = ReachPoint([20000], h_training, [200.0])
        gpm = GammaPoissonModel([rp], max_reach=10000)
        gpm._fit()
        rp = gpm.by_impressions([10000], max_frequency=5)
        h_expected = np.array([9682, 8765, 7353, 5750, 4233])
        h_actual = np.array([int(rp.reach(i)) for i in range(1, 6)])
        total_error = np.sum((h_expected - h_actual) ** 2 / h_expected)
        self.assertAlmostEqual(rp.spends[0], 100.0)
        for i in range(len(h_actual)):
            self.assertTrue(
                (h_actual[i] - h_expected[i]) ** 2 / h_actual[i] < 0.1,
                f"Discrepancy found at position {i}. "
                f"Got {h_actual[i]} Expected {h_expected[i]}",
            )

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.GammaPoissonModel._fit_histogram_fixed_N"
    )
    def test_by_spend(self, mock_gamma_poisson_model):
        mock_gamma_poisson_model.return_value = (25000, 5.0, 2.0)
        # Imax = 25000, N = 10000, alpha = 5, beta = 2
        h_training = [8124, 5464, 3191, 1679, 815, 371, 159, 64, 23, 6, 0]
        rp = ReachPoint([20000], h_training, [200.0])
        gpm = GammaPoissonModel([rp], max_reach=10000)
        gpm._fit()
        rp = gpm.by_spend([100.0], max_frequency=5)
        h_expected = np.array([9682, 8765, 7353, 5750, 4233])
        h_actual = np.array([int(rp.reach(i)) for i in range(1, 6)])
        total_error = np.sum((h_expected - h_actual) ** 2 / h_expected)
        self.assertTrue(total_error < 1)
        self.assertAlmostEqual(rp.impressions[0], 10000.0)
        self.assertAlmostEqual(rp.spends[0], 100.0)
        for i in range(len(h_actual)):
            self.assertTrue(
                (h_actual[i] - h_expected[i]) ** 2 / h_actual[i] < 0.1,
                f"Discrepancy found at position {i}. "
                f"Got {h_actual[i]} Expected {h_expected[i]}",
            )

    @patch.object(
        GammaPoissonModel, "_fit_histogram_fixed_N", return_value=(25000, 5.0, 2.0)
    )
    def test_overspend(self, mock_gamma_poisson_model):
        # Imax = 25000, N = 10000, alpha = 5, beta = 2
        h_training = [8124, 5464, 3191, 1679, 815, 371, 159, 64, 23, 6, 0]
        rp = ReachPoint([20000], h_training, [200.0])
        gpm = GammaPoissonModel([rp], max_reach=10000)
        gpm._fit()
        self.assertAlmostEqual(gpm.by_impressions([30000]).reach(1), 10000.0, delta=0.1)
        self.assertAlmostEqual(gpm.by_spend([300]).reach(1), 10000.0, delta=0.1)


if __name__ == "__main__":
    absltest.main()
