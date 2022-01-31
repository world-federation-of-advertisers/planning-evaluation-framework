"""Tests for distribution_sampler."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from wfa_planning_evaluation_framework.sampling_research import distribution_sampler

# Number of trials for stochastic test cases
NUM_TRIALS = 10000


class DistributionSamplerTest(parameterized.TestCase):
    @parameterized.parameters(
        (10, "dgaussian", 10),
        (15, "dgaussian", 15),
        (0.1, "dlaplace", 14.13),
        (0.15, "dlaplace", 9.42),
    )
    def test_noise_sampling_with_parameter(
        self, noise_parameter, noise_type, expected_standard_deviation
    ):
        noise_sampler = distribution_sampler.noise_sampler_factory(
            np.random.default_rng(seed=1),
            epsilon=0,
            noise_parameter=noise_parameter,
            noise_type=noise_type,
        )
        noise_list = []
        for _ in range(NUM_TRIALS):
            noise = noise_sampler()
            self.assertIsInstance(noise, int)
            noise_list.append(noise)
        self.assertAlmostEqual(np.average(noise_list), 0, delta=0.2)
        self.assertAlmostEqual(
            np.std(noise_list), expected_standard_deviation, delta=0.5
        )

    @parameterized.parameters((1, 10, 14.13), (3, 20, 9.42))
    def test_noise_sampling_dlaplace_basic(
        self, epsilon, num_queries, expected_standard_deviation
    ):
        noise_sampler = distribution_sampler.noise_sampler_factory(
            np.random.default_rng(seed=1),
            epsilon=epsilon,
            num_queries=num_queries,
            noise_type="dlaplace_basic",
        )
        noise_list = []
        for _ in range(NUM_TRIALS):
            noise = noise_sampler()
            self.assertIsInstance(noise, int)
            noise_list.append(noise)
        self.assertAlmostEqual(np.average(noise_list), 0, delta=0.2)
        self.assertAlmostEqual(
            np.std(noise_list), expected_standard_deviation, delta=0.5
        )

    @parameterized.parameters(
        (5, 21, "uniform", [4, 4, 4, 4, 5]),
        (6, 100, "zipf", [60, 15, 6, 3, 2, 14]),
        (10, 1000, "poisson", [6, 33, 84, 140, 175, 175, 146, 104, 65, 72]),
    )
    def test_get_population_histogram(
        self, max_freq, n, distribution, expected_population_histogram
    ):
        population_histogram = list(
            distribution_sampler.get_population_histogram(max_freq, n, distribution)
        )
        self.assertSequenceEqual(population_histogram, expected_population_histogram)

    def test_sample_histogram(self):
        population_histogram = [100, 200, 300, 400]
        sample_size = 100
        sample_histograms = []
        for _ in range(NUM_TRIALS):
            sample_histograms.append(
                distribution_sampler.sample_histogram(
                    population_histogram, sample_size, np.random.default_rng(seed=1)
                )
            )
        mean_sample_histogram = list(np.average(sample_histograms, axis=0))
        self.assertSequenceAlmostEqual(mean_sample_histogram, [10, 20, 30, 40], delta=1)


if __name__ == "__main__":
    absltest.main()
