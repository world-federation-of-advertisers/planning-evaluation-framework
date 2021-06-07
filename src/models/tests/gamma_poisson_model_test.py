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

        gpm = GammaPoissonModel([ReachPoint([], [])])
        self.assertAlmostEqual(
            gpm._logpmf(1, 1.0, 1.0), np.log(gamma_poisson_pmf(1, 1.0, 1.0))
        )

        self.assertAlmostEqual(
            gpm._logpmf(2, 1.0, 1.0), np.log(gamma_poisson_pmf(2, 1.0, 1.0))
        )

        self.assertAlmostEqual(
            gpm._logpmf(2, 3.0, 1.0), np.log(gamma_poisson_pmf(2, 3.0, 1.0))
        )

        self.assertAlmostEqual(
            gpm._logpmf(2, 1.0, 4.0), np.log(gamma_poisson_pmf(2, 1.0, 4.0))
        )

    def test_knreach(self):
        gpm = GammaPoissonModel([ReachPoint([], [])])
        self.assertAlmostEqual(gpm._knreach(0, 1, 1, 2, 1.0, 1.0), 1 / 8)
        self.assertAlmostEqual(gpm._knreach(1, 1, 1, 2, 1.0, 1.0), 1 / 8)
        self.assertAlmostEqual(gpm._knreach(2, 1, 1, 2, 1.0, 1.0), 0)
        self.assertAlmostEqual(gpm._knreach(1, 1, 1, 2, 1.0, 2.0), 1.0 / 9.0)
        self.assertAlmostEqual(gpm._knreach(1, [1], 1, 2, 1.0, 1.0)[0], 1 / 8)
        self.assertAlmostEqual(gpm._knreach(1, [1, 2], 1, 2, 1.0, 1.0)[0], 1 / 8)
        self.assertAlmostEqual(gpm._knreach(1, [1, 2], 1, 2, 1.0, 1.0)[1], 1 / 16)
        self.assertAlmostEqual(
            gpm._knreach(1, 3, 1, 3, 1.0, 1.0), 3 * (1 / 3) * (2 / 3) ** 2 * (1 / 16)
        )

    def test_kreach(self):
        gpm = GammaPoissonModel([ReachPoint([], [])])
        self.assertAlmostEqual(gpm._kreach([0], 1, 2, 1, 1)[0], 2 / 3)
        self.assertAlmostEqual(gpm._kreach([0], 1, 3, 1, 1)[0], 3 / 4)
        self.assertAlmostEqual(gpm._kreach([1], 1, 3, 1, 1)[0], 3 / 16)
        self.assertAlmostEqual(gpm._kreach([0, 1, 2], 1, 3, 1, 1)[0], 3 / 4)
        self.assertAlmostEqual(gpm._kreach([0, 1, 2], 1, 3, 1, 1)[1], 3 / 16)
        self.assertAlmostEqual(gpm._kreach([0, 1, 2], 1, 3, 1, 1)[2], 3 / 64)

    def test_expected_impressions(self):
        gpm = GammaPoissonModel([ReachPoint([], [])])
        self.assertAlmostEqual(gpm._expected_impressions(1, 1, 1), 1.0)
        self.assertAlmostEqual(gpm._expected_impressions(2, 1, 1), 2.0)
        self.assertAlmostEqual(gpm._expected_impressions(1, 2, 1), 2)
        self.assertAlmostEqual(gpm._expected_impressions(1, 1, 2), 0.5)

    def test_expected_histogram(self):
        gpm = GammaPoissonModel([ReachPoint([], [])])
        h_actual = gpm._expected_histogram(4, 12, 16, 1, 1, max_freq=3)
        self.assertLen(h_actual, 3)
        self.assertAlmostEqual(h_actual[0], 3)
        self.assertAlmostEqual(h_actual[1], 3 / 4.0)
        self.assertAlmostEqual(h_actual[2], 3 / 16.0)

    def test_feasible_point(self):
        gpm = GammaPoissonModel([ReachPoint([], [])])
        alpha, beta, I, N = gpm._feasible_point([14, 8, 4, 2])
        self.assertAlmostEqual(alpha, 1.0)
        self.assertAlmostEqual(beta, 28 / 50)
        self.assertAlmostEqual(I, 100)
        self.assertAlmostEqual(N, 56)

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.MAXIMUM_BASIN_HOPS",
        1,
    )
    def test_fit_histogram_chi2_distance(self):
        gpm = GammaPoissonModel([ReachPoint([], [])])
        I0, Imax0, N0, alpha0, beta0 = 20000, 25000, 10000, 5, 2
        h_actual = gpm._expected_histogram(I0, Imax0, N0, alpha0, beta0)
        Imax, N, alpha, beta = gpm._fit_histogram_chi2_distance(
            h_actual, regularization_param=0.01
        )
        Ih = np.sum([h_actual[i] * (i + 1) for i in range(len(h_actual))])
        h_predicted = gpm._expected_histogram(Ih, Imax, N, alpha, beta)
        for i in range(len(h_actual)):
            self.assertTrue(
                (h_actual[i] - h_predicted[i]) ** 2 < 15,
                f"Discrepancy found at position {i}. "
                f"Got {h_predicted[i]} Expected {h_actual[i]}",
            )

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.MAXIMUM_BASIN_HOPS",
        1,
    )
    def test_fit_histogram_fixed_N(self):
        gpm = GammaPoissonModel([ReachPoint([], [])])
        I0, Imax0, N0, alpha0, beta0 = 4000, 10000, 10000, 1, 1
        h_actual = gpm._expected_histogram(I0, Imax0, N0, alpha0, beta0)
        Imax, alpha, beta = gpm._fit_histogram_fixed_N(
            h_actual, N0, regularization_param=0.01
        )
        Ih = np.sum([h_actual[i] * (i + 1) for i in range(len(h_actual))])
        h_predicted = gpm._expected_histogram(Ih, Imax, N0, alpha, beta)
        for i in range(len(h_actual)):
            self.assertTrue(
                (h_actual[i] - h_predicted[i]) ** 2 < 15,
                f"Discrepancy found at position {i}. "
                f"Got {h_predicted[i]} Expected {h_actual[i]}",
            )

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.MAXIMUM_BASIN_HOPS",
        1,
    )
    def test_fit_variable_N(self):
        h_actual = [2853, 813, 230, 64, 17, 4, 1, 0, 0, 0]
        rp = ReachPoint([4000], h_actual)
        gpm = GammaPoissonModel([rp])
        gpm._fit()
        self.assertAlmostEqual(gpm._max_reach, 10000, delta=100)
        self.assertAlmostEqual(gpm._alpha, 1.0, delta=0.05)

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.MAXIMUM_BASIN_HOPS",
        1,
    )
    def test_fit_fixed_N(self):
        # Imax = 4000, N = 10000, alpha = 1, beta = 1
        h_actual = [2853, 813, 230, 64, 17, 4, 1, 0, 0, 0]
        rp = ReachPoint([4000], h_actual)
        gpm = GammaPoissonModel([rp], max_reach=10000)
        gpm._fit()
        self.assertAlmostEqual(gpm._max_reach, 10000, delta=10)
        self.assertAlmostEqual(gpm._alpha, 1.0, delta=0.05)

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.MAXIMUM_BASIN_HOPS",
        1,
    )
    def test_by_impressions(self):
        # Imax = 25000, N = 10000, alpha = 5, beta = 2
        h_training = [8124, 5464, 3191, 1679, 815, 371, 159, 64, 23, 6, 0]
        rp = ReachPoint([20000], h_training, [200.0])
        gpm = GammaPoissonModel([rp], max_reach=10000)
        gpm._fit()
        he = gpm._expected_histogram(
            20000, gpm._max_impressions, gpm._max_reach, gpm._alpha, gpm._beta
        )
        hp = gpm._expected_histogram(
            10000, gpm._max_impressions, gpm._max_reach, gpm._alpha, gpm._beta
        )
        re = list(reversed(np.cumsum(list(reversed(he)))))
        rx = list(reversed(np.cumsum(list(reversed(hp)))))
        rp = gpm.by_impressions([10000], max_frequency=5)
        h_expected = [5970, 2631, 961, 310, 91]
        h_actual = [int(rp.reach(i)) for i in range(1, 6)]
        self.assertAlmostEqual(rp.spends[0], 100.0)
        for i in range(len(h_actual)):
            self.assertTrue(
                (h_actual[i] - h_expected[i]) ** 2 < 15,
                f"Discrepancy found at position {i}. "
                f"Got {h_actual[i]} Expected {h_expected[i]}",
            )

    @patch(
        "wfa_planning_evaluation_framework.models.gamma_poisson_model.MAXIMUM_BASIN_HOPS",
        1,
    )
    def test_by_spend(self):
        # Imax = 25000, N = 10000, alpha = 5, beta = 2
        h_training = [8124, 5464, 3191, 1679, 815, 371, 159, 64, 23, 6, 0]
        rp = ReachPoint([20000], h_training, [200.0])
        gpm = GammaPoissonModel([rp], max_reach=10000)
        gpm._fit()
        rp = gpm.by_spend([100.0], max_frequency=5)
        h_expected = [5970, 2631, 961, 310, 91]
        h_actual = [int(rp.reach(i)) for i in range(1, 6)]
        self.assertAlmostEqual(rp.impressions[0], 10000.0)
        self.assertAlmostEqual(rp.spends[0], 100.0)
        for i in range(len(h_actual)):
            self.assertTrue(
                (h_actual[i] - h_expected[i]) ** 2 < 15,
                f"Discrepancy found at position {i}. "
                f"Got {h_actual[i]} Expected {h_expected[i]}",
            )


if __name__ == "__main__":
    absltest.main()
