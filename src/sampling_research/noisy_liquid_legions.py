"""Tools to experiment with noisy liquid legions.

For more detail of LiquidLegions, please refer to the following manuscript:
  Craig Wright, Evgeny Skvortsov, Benjamin Kreuter, Yao Wang, Raimundo Mirisola.
  "Privacy-Preserving Secure Cardinality and Frequency Estimation".
  To appear in PoPETs 2022.1.
"""

import dataclasses
import typing

import numpy as np
from scipy import special

from dp_accounting import common
from wfa_planning_evaluation_framework.sampling_research import distribution_sampler


@dataclasses.dataclass
class ExperimentResults(object):
    """Results of experiments on noisy LiquidLegions.

    Attributes:
      true_reach: the true (unnoised) reach.
      true_frequency_histogram: the true (unnoised) frequency histogram, assumed
        to be normalized (i.e. sum of all entries equal to one).
      noisy_reaches: list of estimated reach from each run.
      noisy_frequency_histograms: list of noisy frequency histogram from each run.
    """

    true_reach: int
    true_frequency_histogram: typing.List[float]
    noisy_reaches: typing.List[float]
    noisy_frequency_histograms: typing.List[typing.List[float]]


class NoisyLiquidLegionSimulator(object):
    """Simulator for frequency and reach estimates from noisy LiquidLegions.

    Attributes:
      m: number of registers of the LiquidLegions. If m = -1, then use pure
        subsampling without the LiquidLegions sketch (which can be thought of as a
        sketch with infinite number of registers).
      a: decay rate of the LiquidLegions.
      register_probs: a list of probabilities each item is assigned to each
        register.
      rng: the random number generator.
    """

    def __init__(
        self,
        m: int = 100_000,
        a: float = 12,
        rng: typing.Optional[np.random.Generator] = None,
    ):
        self.m = m
        self.a = a
        if not self.pure_sampling():
            self.register_probs = self.get_register_probs()
        self.rng = rng or np.random.default_rng()

    def pure_sampling(self) -> bool:
        """Returns whether pure sampling is used."""
        return self.m == -1

    def get_register_probs(self) -> np.ndarray:
        """Gets probability that an item is assigned to each register.

        Returns:
          Probability vector with m entries, where the i-th entry is the probability
          that an item is assigned to the i-th register.
        """
        probs = np.exp(-self.a * np.arange(self.m) / self.m)
        return probs / sum(probs)

    def estimate_cardinality_from_num_non_empty_reg(
        self, num_nonempty_registers: int
    ) -> float:
        """Estimates cardinalty from a LiquidLegions with global noise.

        Args:
          num_nonempty_registers: (unnoised) number of non-empty registers of the
            LiquidLegions.

        Returns:
          The estimate cardinality.
        """
        if self.pure_sampling():
            return num_nonempty_registers

        # Sometimes noise added make the number of register out of range.
        # Fixed below; if not fixed, will run into numerical problems.
        if num_nonempty_registers >= self.m:
            return int(1e10)
        if num_nonempty_registers < 0:
            return 0

        def _expected_proportion(n_over_m):
            """Expected proportion of non-empty registers among all the registers."""
            if n_over_m <= 0:
                return 0
            if self.a != 0:
                # The formula below is from Equation (3) of the corresponding
                # manuscript.
                return (
                    1
                    - (
                        -special.expi(-self.a * n_over_m / (np.exp(self.a) - 1))
                        + special.expi(
                            -self.a * np.exp(self.a) * n_over_m / (np.exp(self.a) - 1)
                        )
                    )
                    / self.a
                )
            else:
                # When a = 0 (i.e. uniform bloom filter), each register is identical and
                # there is a probability 1 - (1 - 1 / self.m) ** n that it is non-empty.
                return 1 - (1 - 1 / self.m) ** (n_over_m * self.m)

        result = self.m * common.inverse_monotone_function(
            _expected_proportion,
            num_nonempty_registers / self.m,
            common.BinarySearchParameters(0, 1e10, initial_guess=1),
            increasing=True,
        )
        assert result >= 0, "Negative estimate should not happen."
        return result

    def get_raw_liquid_legions_nonempty_and_active_registers(
        self, n: int, num_runs: int, subsampling_rate: float = 1
    ) -> typing.Tuple[typing.List[int], typing.List[int]]:
        """Generates (unnoised) numbers of non-empty and active registers.

        For each run, this method simulates a procedure where each user is included
        in the sample with probability subsampling_rate. The users in the sample are
        then inserted into the LiquidLegions sketch. We then report the number of
        non-empty registers and active registers from the runs. (Active registers
        are those with exactly one user; they are used to estimate frequency
        histogram.)

        Args:
          n: cardinality.
          num_runs: number of times to generate the numbers (independently).
          subsampling_rate: the probability that each user is included in the
            sketch.

        Returns:
          A tuple l1, l2. l1 is the list of num_runs numbers, each representing the
          number of non-empty registers in a single run. l2 is the list of num_runs
          numbers, each representing the number of active registers in a single run.
        """
        if self.pure_sampling():
            num_samples = np.random.binomial(n, subsampling_rate, size=num_runs)
            return num_samples, num_samples

        # This might lead to one additional non-empty/active register but this
        # should not effect the outcome in any non-trivial way.
        probability_vector = np.append(self.register_probs * subsampling_rate, 0)
        liquid_legions = np.random.multinomial(n, probability_vector, size=num_runs)
        return (
            np.sum(liquid_legions >= 1, axis=1),
            np.sum(liquid_legions == 1, axis=1),
        )

    def get_batch_estimates(
        self,
        n: int,
        num_runs: int = 500,
        max_freq: int = 15,
        distribution: str = "zipf",
        noise_samplers: typing.Optional[
            typing.Sequence[typing.Callable[[], float]]
        ] = None,
        use_destroyed_registers_for_frequency: bool = False,
        subsampling_rate: float = 1,
    ) -> typing.Sequence[ExperimentResults]:
        """Generates experiment results for repeated runs of given parameters.

        For each run, this method simulates a procedure where each user is included
        in the sample with probability subsampling_rate. The users in the sample are
        then inserted into the LiquidLegions sketch. We then add noises using the
        noise sampler given and finally use the noised number of non-empty registers
        and the noised number of active registers to estimate reach and (normalized)
        frequency histogram, respectively.

        Args:
          n: cardinality.
          num_runs: number of times to run the experiments (independently).
          max_freq: maximum frequency, above which we snap to the max_freq-th
            bucket.
          distribution: frequency distribution to run experiment on; can be one of
            (i) 'uniform': the uniform distribution over max_freq buckets,
            (ii) 'zipf': the Zipf distribution with parameter 2 with mass,
            (iii) 'poisson': the Poisson distribution with parameter 5 shifted by
              one.
          noise_samplers: a list of functions, where each call to a function returns
            a noise distribution to be added.
          use_destroyed_registers_for_frequency: whether to also use the destroyed
            (i.e. non-active) registers for the frequency estimation. Note that this
            is hypothetical as it is not yet supported by our protocol.
          subsampling_rate: the probability that each user is included in the
            sketch. Note that if we are using random bucketing approach then this
            would be 1 / (# of buckets); in this case, the number of queries would
            be the number of queries per bucket.

        Returns:
          A list of ExperimentResults objects containing the results, where each
          ExperimentResults object corresponds to each noise sampler.
        """
        print(
            f"Starting {num_runs} experiments for Reach {n}, "
            f"Num registers = {self.m}, Decay rate {self.a}, "
            f"Subsampling rate = {subsampling_rate}."
        )

        (
            ll_non_empty_counts,
            ll_active_counts,
        ) = self.get_raw_liquid_legions_nonempty_and_active_registers(
            n, num_runs, subsampling_rate=subsampling_rate
        )
        print("Finished generating counts")

        # Reach experiments
        print("Computing Reach Estimates")
        true_reach = n
        noisy_reaches_list = []
        for noise_sampler in noise_samplers:
            noisy_non_empty_counts = [
                count + noise_sampler() for count in ll_non_empty_counts
            ]
            noisy_reaches = [
                self.estimate_cardinality_from_num_non_empty_reg(noisy_non_empty_count)
                / subsampling_rate
                for noisy_non_empty_count in noisy_non_empty_counts
            ]
            noisy_reaches_list.append(noisy_reaches)
        print("Finished Computing Reach Estimates")

        # frequency estimates
        print("Computing Frequency Estimates")

        # Compute true population histogram for given distribution.
        hist = distribution_sampler.get_population_histogram(max_freq, n, distribution)

        true_frequency_histogram = hist / sum(hist)
        if use_destroyed_registers_for_frequency:
            register_counts = ll_non_empty_counts
        else:
            register_counts = ll_active_counts

        noisy_frequency_histogram_list = [[] for _ in noise_samplers]
        for sample_size in register_counts:
            # Sample the histogram among the active registers
            sampled_hist = distribution_sampler.sample_histogram(
                hist, sample_size, self.rng
            )
            for i, noise_sampler in enumerate(noise_samplers):
                # Add noise and compute estimated frequency histogram.
                noisy_hist = [v + noise_sampler() for v in sampled_hist]
                noisy_freq = np.array(noisy_hist) / sum(noisy_hist)
                noisy_frequency_histogram_list[i].append(noisy_freq)
        print("Finished Computing Frequency Estimates")

        result_list = []
        for noisy_reaches, noisy_frequency_histogram in zip(
            noisy_reaches_list, noisy_frequency_histogram_list
        ):
            result_list.append(
                ExperimentResults(
                    true_reach,
                    true_frequency_histogram,
                    noisy_reaches,
                    noisy_frequency_histogram,
                )
            )

        return result_list
