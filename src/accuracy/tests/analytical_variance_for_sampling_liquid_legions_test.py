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
"""Tests for analytical_variance_for_sampling_liquid_legions.py."""


from numpy.lib.polynomial import RankWarning
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from wfa_planning_evaluation_framework.accuracy.analytical_variance_for_sampling_liquid_legions import (
    AnalyticalAccuracyEvaluator,
)
from wfa_cardinality_estimation_evaluation_framework.estimators.same_key_aggregator import (
    ExponentialSameKeyAggregator,
    StandardizedHistogramEstimator,
)
from wfa_cardinality_estimation_evaluation_framework.estimators.estimator_noisers import (
    LaplaceEstimateNoiser,
)

reach_epsilon = 0.1  # dp epsilon for reach
frequency_epsilon = 0.1  # dp epsilon for frequency
a = 10  # decay rate
m = int(1e4)  # number of registers
pi = 0.1  # sampling proportion
aae = AnalyticalAccuracyEvaluator(
    a=a,
    m=m,
    pi=pi,
    ssreach=2 / reach_epsilon ** 2,
    ssfreq=2 / frequency_epsilon ** 2,
)
# Laplace noises have variance = 2 / epsilon^2.
rng = np.random.default_rng(1)


class AnalyticalAccuracyEvaluatorTest(parameterized.TestCase):
    @parameterized.parameters(
        [1e5, 0.02614],
        [1e6, 0.02634],
    )
    def test_relative_std_n_hat_formula(self, n, expected_relative_std):
        """Test if the formula of Var(n-hat) is implemented correctly."""
        self.assertAlmostEqual(aae.relative_std_n_hat(n), expected_relative_std, 3)

    def simulate(self, n, relative_hist):
        """Simulate the reach and frequency estimation using random hash.

        Args:
            n:  True 1+ reach.
            relative_hist:  True relative frequency histogram.

        Returns:
            The estimate of frequency histogram, i.e.,
            estimated_reach * estimated_relative_frequency_histogram.
        """
        sketch = ExponentialSameKeyAggregator(
            length=m,
            decay_rate=a,
            random_seed=rng.integers(low=0, high=1e9),
        )
        y = rng.binomial(int(n), pi)  # the sampled reach
        hist = [0] + [int(y * r) for r in relative_hist]
        cum_hist = np.cumsum(hist)
        for i in range(len(relative_hist)):
            for x in range(cum_hist[i], cum_hist[i + 1]):
                sketch.add_ids([x] * (i + 1))
        estimator = StandardizedHistogramEstimator(
            max_freq=4,
            reach_noiser_class=LaplaceEstimateNoiser,
            frequency_noiser_class=LaplaceEstimateNoiser,
            reach_epsilon=reach_epsilon,
            frequency_epsilon=frequency_epsilon,
            reach_delta=0,
            frequency_delta=0,
            reach_noiser_kwargs={
                "random_state": np.random.RandomState(
                    seed=rng.integers(low=0, high=1e9)
                )
            },
            frequency_noiser_kwargs={
                "random_state": np.random.RandomState(
                    seed=rng.integers(low=0, high=1e9)
                )
            },
        )
        est = np.array(estimator.estimate_cardinality(sketch))
        est[:-1] -= est[1:].copy()  # convert the kplus reaches to k-reach histogram.
        return est / pi

    @parameterized.parameters(
        [1e5, (0.5, 0.25, 0.2, 0.05)],
        [2e5, (0.3, 0.4, 0.15, 0.15)],
    )
    def test_alignment_with_empirical(self, n, relative_hist):
        """Test whether the stds obtained by analytical formulas roughly aligh with simulation.

        Args:
            n:  True 1+ reach.
            relative_hist:  True relative frequency histogram.
        """

        def test_theory_discrepency(empirical, theoretical, type):
            self.assertTrue(
                (abs(empirical - theoretical) / theoretical < 0.5)
                or (abs(empirical - theoretical) < 0.05),
                msg=f"{type} discrepancy. Empirical={empirical}, Theoretical={theoretical}.",
            )
            # Passes the unit test if: relative diff < 0.5, or absolute diff < 0.05.
            # I'm using this very loose criterion because the test is based on a simulation of only 10
            # replicates.  A test with more replicates would be too slow.
            # Pasin will thoroughly compare the analytical and the empirical results in his experiments.

        results = [self.simulate(n, relative_hist) for _ in range(10)]
        test_theory_discrepency(
            np.std([sum(res) for res in results]),
            aae.std_n_hat(n),
            "Reach",
        )
        for k in range(len(relative_hist)):
            test_theory_discrepency(
                np.std([res[k] / sum(res) for res in results]),
                aae.std_rk_hat(n=n, nk=int(n * relative_hist[k]), max_freq=4),
                f"Frequency {k + 1}",
            )


if __name__ == "__main__":
    absltest.main()
