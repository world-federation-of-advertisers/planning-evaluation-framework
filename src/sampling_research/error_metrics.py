"""Error metrics for reach and frequency estimates."""

import abc
import typing

import numpy as np

from wfa_planning_evaluation_framework.sampling_research import noisy_liquid_legions

DEFAULT_ERROR_PERCENTILE = (80, 90, 95)


class ErrorMetric(metaclass=abc.ABCMeta):
    """Metaclass for error metrics."""

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def _report_errors_at_percentiles(
        self,
        error_list: typing.List[float],
        error_percentiles: typing.Optional[typing.List[float]] = None,
        print_report: bool = True,
    ) -> typing.Dict[float, float]:
        """Report errors for given percentile."""
        error_percentiles = error_percentiles or list(DEFAULT_ERROR_PERCENTILE)
        error_report = {
            percentile: np.percentile(error_list, percentile)
            for percentile in error_percentiles
        }
        if print_report:
            print(f"Error report for {self}:")
            for percentile, error in error_report.items():
                print(f"{percentile}-Percentile: {error}.")
        return error_report


class ReachErrorMetric(ErrorMetric, metaclass=abc.ABCMeta):
    """Error metrics for reach evaluation."""

    def report_error(
        self,
        experiment_results: noisy_liquid_legions.ExperimentResults,
        error_percentiles: typing.Optional[typing.List[float]] = None,
        print_report: bool = True,
    ) -> typing.Dict[float, float]:
        """Report errors for given percentile."""
        errors = [
            self._compute_error(experiment_results.true_reach, noisy_reach)
            for noisy_reach in experiment_results.noisy_reaches
        ]
        return self._report_errors_at_percentiles(
            errors, error_percentiles=error_percentiles, print_report=print_report
        )

    @abc.abstractmethod
    def _compute_error(self, true_reach: float, estimated_reach: float) -> float:
        """Compute the error between the true reach and estimate reach.

        Args:
          true_reach: a true (unnoised) reach.
          estimated_reach: an estimate reach computed by an algorithm.

        Returns:
          A number representing the error under this metric.
        """
        raise NotImplementedError


class FrequencyErrorMetric(ErrorMetric, metaclass=abc.ABCMeta):
    """Error metrics for normalized frequency histogram evaluation."""

    def report_error(
        self,
        experiment_results: noisy_liquid_legions.ExperimentResults,
        error_percentiles: typing.Optional[typing.List[float]] = None,
        print_report: bool = True,
    ) -> typing.Dict[float, float]:
        """Report errors for given percentile."""
        errors = [
            self._compute_error(experiment_results.true_frequency_histogram, noisy_freq)
            for noisy_freq in experiment_results.noisy_frequency_histograms
        ]
        return self._report_errors_at_percentiles(
            errors, error_percentiles=error_percentiles, print_report=print_report
        )

    @abc.abstractmethod
    def _compute_error(
        self,
        true_normalized_frequencies: np.ndarray,
        estimated_normalized_frequencies: np.ndarray,
    ) -> float:
        """Compute the error between the true values and estimate values.

        Args:
          true_normalized_frequencies: a true (unnoised) frequency histogram,
            assumed to be normalized (i.e. sum of all entries equal to one).
          estimated_normalized_frequencies: an estimated frequency histogram.

        Returns:
          A number representing the error under this metric.
        """
        raise NotImplementedError


class ReachRelativeError(ReachErrorMetric):
    """Relative error for reach estimation."""

    def __str__(self):
        return "relative error (reach)"

    def _compute_error(self, true_reach: float, estimated_reach: float) -> float:
        """See base class."""
        return abs(true_reach - estimated_reach) / true_reach


class FrequencyMaxError(FrequencyErrorMetric):
    """Maximum error over all buckets of normalized frequency histograms."""

    def __str__(self):
        return "maximum error (frequency)"

    def _compute_error(
        self,
        true_normalized_frequencies: np.ndarray,
        estimated_normalized_frequencies: np.ndarray,
    ) -> float:
        """See base class."""
        return np.max(
            np.abs(true_normalized_frequencies - estimated_normalized_frequencies)
        )


class FrequencyTotalVariationError(FrequencyErrorMetric):
    """Total variation (aka Shuffle) error of normalized frequency histograms."""

    def __str__(self):
        return "TV error (frequency)"

    def _compute_error(
        self,
        true_normalized_frequencies: np.ndarray,
        estimated_normalized_frequencies: np.ndarray,
    ) -> float:
        """See base class."""
        return (
            np.sum(
                np.abs(true_normalized_frequencies - estimated_normalized_frequencies)
            )
            / 2
        )


class FrequencyKPlusAboveThresholdRelativeError(FrequencyErrorMetric):
    """Maximum relative error of k+ reach in the normalized frequency histogram for all k whose k+ reach is at least the threshold.

    Attributes:
      threshold: the value for which each k such that k+ reach is at least
        threshold is included in the computation.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __str__(self):
        return f"k+ reach above {self.threshold} relative error (frequency)"

    def _compute_error(
        self,
        true_normalized_frequencies: np.ndarray,
        estimated_normalized_frequencies: np.ndarray,
    ) -> float:
        """See base class."""
        true_k_plus = 1
        estimated_k_plus = 1
        max_relative_error = 0
        for i in range(0, len(true_normalized_frequencies) - 1):
            true_k_plus -= true_normalized_frequencies[i]
            estimated_k_plus -= estimated_normalized_frequencies[i]
            if true_k_plus < self.threshold:
                break
            max_relative_error = max(
                max_relative_error, abs(true_k_plus - estimated_k_plus) / true_k_plus
            )
        return max_relative_error
