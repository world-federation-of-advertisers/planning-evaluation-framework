"""Tests for noisy_liquid_legions."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from wfa_planning_evaluation_framework.sampling_research import distribution_sampler
from wfa_planning_evaluation_framework.sampling_research import noisy_liquid_legions


class NoisyLiquidLegionsTest(parameterized.TestCase):
    @parameterized.parameters((10, False), (-1, True))
    def test_pure_sampling(self, m, expected_pure_sampling):
        simulator = noisy_liquid_legions.NoisyLiquidLegionSimulator(m=m)
        self.assertEqual(simulator.pure_sampling(), expected_pure_sampling)

    @parameterized.parameters(
        (2, 4, [0.881, 0.119]),
        (3, 1, [0.448, 0.321, 0.230]),
        (4, 0, [0.25, 0.25, 0.25, 0.25]),
    )
    def test_register_probs(self, m, a, expected_register_probs):
        simulator = noisy_liquid_legions.NoisyLiquidLegionSimulator(m=m, a=a)
        self.assertSequenceAlmostEqual(
            simulator.register_probs, expected_register_probs, delta=0.001
        )

    @parameterized.named_parameters(
        ("pure_sampling", 10, -1, 0, 10),
        ("uniform_bloom_filer", 50, 100, 0, 68.968),
        ("liquid_legions_1", 60, 200, 2, 75.584),
        ("liquid_legions_2", 70, 150, 3, 119.115),
    )
    def test_estimate_cardinality_from_num_non_empty_reg(
        self, num_nonempty_registers, m, a, expected_estimated_cardinality
    ):
        simulator = noisy_liquid_legions.NoisyLiquidLegionSimulator(m=m, a=a)
        self.assertAlmostEqual(
            simulator.estimate_cardinality_from_num_non_empty_reg(
                num_nonempty_registers
            ),
            expected_estimated_cardinality,
            delta=0.01,
        )

    @mock.patch.object(np.random, "binomial", autospec=True)
    def test_get_raw_liquid_legions_nonempty_and_active_registers_pure_sampling(
        self, mock_binomial
    ):
        n = 100
        subsampling_rate = 0.1
        num_runs = 3
        mock_binomial.return_value = [2, 5, 6]
        simulator = noisy_liquid_legions.NoisyLiquidLegionSimulator(m=-1)
        self.assertEqual(
            simulator.get_raw_liquid_legions_nonempty_and_active_registers(
                n, num_runs, subsampling_rate=subsampling_rate
            ),
            ([2, 5, 6], [2, 5, 6]),
        )
        mock_binomial.assert_called_once_with(n, subsampling_rate, size=3)

    @mock.patch.object(np.random, "multinomial", autospec=True)
    def test_get_raw_liquid_legions_nonempty_and_active_registers(
        self, mock_multinomial
    ):
        n = 100
        subsampling_rate = 0.1
        num_runs = 3
        mock_multinomial.return_value = np.array(
            [
                [4, 0, 1, 1],
                [3, 1, 1, 1],
                [1, 0, 2, 0],
            ]
        )
        simulator = noisy_liquid_legions.NoisyLiquidLegionSimulator(m=4, a=1)
        (
            non_empty_registers,
            active_registers,
        ) = simulator.get_raw_liquid_legions_nonempty_and_active_registers(
            n, num_runs, subsampling_rate=subsampling_rate
        )
        self.assertSequenceEqual(list(non_empty_registers), [3, 4, 2])
        self.assertSequenceEqual(list(active_registers), [2, 3, 1])
        self.assertEqual(mock_multinomial.call_args[0][0], n)
        self.assertSequenceAlmostEqual(
            mock_multinomial.call_args[0][1],
            np.array([0.035, 0.027, 0.021, 0.017, 0]),
            delta=0.001,
        )
        self.assertEqual(mock_multinomial.call_args[1]["size"], 3)

    @mock.patch.object(
        noisy_liquid_legions.NoisyLiquidLegionSimulator,
        "get_raw_liquid_legions_nonempty_and_active_registers",
        autospec=True,
    )
    @mock.patch.object(distribution_sampler, "sample_histogram", autospec=True)
    def test_get_batch_estimates(
        self,
        mock_sample_histogram,
        mock_get_raw_liquid_legions_nonempty_and_active_registers,
    ):
        n = 100
        subsampling_rate = 0.1
        num_runs = 3
        max_freq = 4
        noise_samplers = [lambda: 2, lambda: -1]
        mock_get_raw_liquid_legions_nonempty_and_active_registers.return_value = (
            [14, 17, 18],
            [9, 8, 10],
        )
        mock_sample_histogram.side_effect = [[4, 3, 2, 1], [5, 2, 2, 1], [3, 3, 2, 2]]
        simulator = noisy_liquid_legions.NoisyLiquidLegionSimulator(m=25, a=1)
        result_list = simulator.get_batch_estimates(
            n,
            num_runs=num_runs,
            subsampling_rate=subsampling_rate,
            noise_samplers=noise_samplers,
            max_freq=max_freq,
        )

        mock_get_raw_liquid_legions_nonempty_and_active_registers.assert_called_once_with(
            mock.ANY, n, num_runs, subsampling_rate=subsampling_rate
        )
        self.assertLen(mock_sample_histogram.call_args_list, 3)
        true_unnormalized_histogram = [60, 15, 6, 19]
        self.assertSequenceEqual(
            list(mock_sample_histogram.call_args_list[0][0][0]),
            true_unnormalized_histogram,
        )
        self.assertEqual(mock_sample_histogram.call_args_list[0][0][1], 9)
        self.assertSequenceEqual(
            list(mock_sample_histogram.call_args_list[1][0][0]),
            true_unnormalized_histogram,
        )
        self.assertEqual(mock_sample_histogram.call_args_list[1][0][1], 8)
        self.assertSequenceEqual(
            list(mock_sample_histogram.call_args_list[2][0][0]),
            true_unnormalized_histogram,
        )
        self.assertEqual(mock_sample_histogram.call_args_list[2][0][1], 10)

        self.assertLen(result_list, len(noise_samplers))

        self.assertEqual(result_list[0].true_reach, n)
        self.assertSequenceAlmostEqual(
            result_list[0].true_frequency_histogram, [0.6, 0.15, 0.06, 0.19]
        )
        self.assertSequenceAlmostEqual(
            result_list[0].noisy_reaches, [266.570, 378.809, 430.499], delta=0.001
        )
        self.assertLen(result_list[0].noisy_frequency_histograms, num_runs)
        self.assertSequenceAlmostEqual(
            result_list[0].noisy_frequency_histograms[0],
            [0.333, 0.278, 0.222, 0.167],
            delta=0.001,
        )
        self.assertSequenceAlmostEqual(
            result_list[0].noisy_frequency_histograms[1],
            [0.389, 0.222, 0.222, 0.167],
            delta=0.001,
        )
        self.assertSequenceAlmostEqual(
            result_list[0].noisy_frequency_histograms[2],
            [0.278, 0.278, 0.222, 0.222],
            delta=0.001,
        )

        self.assertEqual(result_list[1].true_reach, n)
        self.assertSequenceAlmostEqual(
            result_list[1].true_frequency_histogram, [0.6, 0.15, 0.06, 0.19]
        )
        self.assertSequenceAlmostEqual(
            result_list[1].noisy_reaches, [189.192, 266.570, 298.789], delta=0.001
        )
        self.assertLen(result_list[1].noisy_frequency_histograms, num_runs)
        self.assertSequenceAlmostEqual(
            result_list[1].noisy_frequency_histograms[0],
            [0.5, 0.333, 0.167, 0],
            delta=0.001,
        )
        self.assertSequenceAlmostEqual(
            result_list[1].noisy_frequency_histograms[1],
            [0.667, 0.167, 0.167, 0],
            delta=0.001,
        )
        self.assertSequenceAlmostEqual(
            result_list[1].noisy_frequency_histograms[2],
            [0.333, 0.333, 0.167, 0.167],
            delta=0.001,
        )


if __name__ == "__main__":
    absltest.main()
