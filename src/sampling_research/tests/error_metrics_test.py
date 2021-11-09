"""Tests for error_metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from wfa_planning_evaluation_framework.sampling_research import error_metrics


class ErrorMetricsTest(parameterized.TestCase):
    @parameterized.parameters((10, 9, 0.1), (50, 51, 0.02))
    def test_reach_relative_error(self, true_reach, estimated_reach, expected_error):
        self.assertAlmostEqual(
            error_metrics.ReachRelativeError()._compute_error(
                true_reach, estimated_reach
            ),
            expected_error,
        )

    @parameterized.parameters(
        (
            np.array([0.1, 0.3, 0.6]),
            np.array([0.04, 0.2, 0.76]),
            0.16,
        ),
        (
            np.array([0.4, 0.4, 0.2]),
            np.array([0.21, 0.49, 0.3]),
            0.19,
        ),
        (
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            np.array([0.1, 0.1, 0.1, 0.3, 0.4]),
            0.2,
        ),
    )
    def test_frequency_max_error(
        self, true_frequencies, estimated_frequencies, expected_error
    ):
        self.assertAlmostEqual(
            error_metrics.FrequencyMaxError()._compute_error(
                true_frequencies, estimated_frequencies
            ),
            expected_error,
        )

    @parameterized.parameters(
        (
            np.array([0.1, 0.3, 0.2, 0.4]),
            np.array([0.04, 0.2, 0.26, 0.5]),
            0.16,
        ),
        (
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            np.array([0.1, 0.1, 0.1, 0.3, 0.4]),
            0.3,
        ),
    )
    def test_total_variation_error(
        self, true_frequencies, estimated_frequencies, expected_error
    ):
        self.assertAlmostEqual(
            error_metrics.FrequencyTotalVariationError()._compute_error(
                true_frequencies, estimated_frequencies
            ),
            expected_error,
        )

    @parameterized.parameters(
        (
            0.15,
            np.array([0.1, 0.3, 0.2, 0.4]),
            np.array([0.04, 0.2, 0.26, 0.5]),
            0.26666666,
        ),
        (
            0.5,
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            np.array([0.1, 0.1, 0.1, 0.3, 0.4]),
            0.33333333,
        ),
    )
    def test_frequency_k_plus_above_threshold(
        self, threshold, true_frequencies, estimated_frequencies, expected_error
    ):
        self.assertAlmostEqual(
            error_metrics.FrequencyKPlusAboveThresholdRelativeError(
                threshold
            )._compute_error(true_frequencies, estimated_frequencies),
            expected_error,
        )

    @parameterized.parameters(
        ([2, 1, 3, 5, 4], [20, 30], [1.8, 2.2]), ([7, 5, 3, 9], [75, 25], [7.5, 4.5])
    )
    def test_report_errors_at_percentiles(
        self, error_list, error_percentiles, expected_percenfile_errors
    ):
        self.assertSequenceAlmostEqual(
            error_metrics.ReachRelativeError()
            ._report_errors_at_percentiles(
                error_list, error_percentiles=error_percentiles, print_report=False
            )
            .values(),
            expected_percenfile_errors,
        )


if __name__ == "__main__":
    absltest.main()
