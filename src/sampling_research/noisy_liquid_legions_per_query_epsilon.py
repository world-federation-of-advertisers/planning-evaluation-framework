"""Computes the largest number of subsampling buckets noisy liquid legions can support for given accuracy and per-query privacy parameters."""

import os
from typing import Sequence

from absl import app
from absl import flags
import numpy as np

from wfa_planning_evaluation_framework.sampling_research import distribution_sampler
from wfa_planning_evaluation_framework.sampling_research import error_metrics
from wfa_planning_evaluation_framework.sampling_research import noisy_liquid_legions
from dp_accounting.common import DifferentialPrivacyParameters
from dp_accounting.privacy_loss_mechanism import DiscreteGaussianPrivacyLoss

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "result_dir", "./simulation_results/", "Directory to write result to."
)
flags.DEFINE_list(
    "true_reach_list",
    [
        "500_000",
        "1_000_000",
        "5_000_000",
        "10_000_000",
        "50_000_000",
        "100_000_000",
        "500_000_000",
    ],
    "Value of true reaches to experiment on.",
)
flags.DEFINE_integer("max_freq", 10, "Value of maximum frequency to experiment on.")
flags.DEFINE_list(
    "per_query_epsilon_list",
    [
        "0.001",
        "0.002",
        "0.004",
        "0.006",
        "0.008",
        "0.01",
        "0.02",
        "0.03",
        "0.04",
        "0.05",
        "0.06",
        "0.08",
        "0.10",
        "0.12",
        "0.14",
        "0.15",
        "0.16",
        "0.18",
        "0.20",
        "0.25",
        "0.30",
        "0.35",
        "0.40",
        "0.50",
        "0.60",
        "0.80",
        "1.00",
    ],
    "Value of per-query DP parameter epsilon to experiment on.",
)
flags.DEFINE_float("delta", 1e-9, "Value of DP parameter delta to experiment on.")
flags.DEFINE_list(
    "sampling_buckets_list",
    [
        "1",
        "2",
        "5",
        "10",
        "20",
        "30",
        "40",
        "50",
        "75",
        "100",
        "150",
        "200",
        "250",
        "300",
        "400",
        "500",
        "600",
        "800",
        "1000",
    ],
    "Value of sampling buckets to experiment on.",
)
flags.DEFINE_float("decay_rate", 12, "The decay rate a of the liquid legion.")
flags.DEFINE_string(
    "frequency_distribution",
    "zipf",
    "The frequency distribution to run experiment on; can be"
    "one of uniform, zipf or poisson.",
)
flags.DEFINE_integer(
    "sketch_size",
    -1,
    "The sketch size m of the liquid legion. If set to -1, use only sampling.",
)
flags.DEFINE_string(
    "noise_type",
    "dgaussian",
    "Noise type to experiment on. Must be one of dlaplace, dgaussian, no_noise",
)
flags.DEFINE_integer("seed", 1000, "Seed for the numpy random number generator.")


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if not os.path.isdir(FLAGS.result_dir):
        os.mkdir(FLAGS.result_dir)

    per_query_epsilon_list = list(map(float, FLAGS.per_query_epsilon_list))
    true_reach_list = list(map(int, FLAGS.true_reach_list))
    sampling_buckets_list = list(map(int, FLAGS.sampling_buckets_list))
    rng = np.random.default_rng(seed=FLAGS.seed)

    metrics = [
        error_metrics.ReachRelativeError(),
        error_metrics.FrequencyTotalVariationError(),
        error_metrics.FrequencyKPlusAboveThresholdRelativeError(0.15),
    ]
    # Default cutoff used to report maximum number of supported queries
    default_percentiles_cutoffs = [(80, 0.2), (90, 0.1), (95, 0.05)]
    default_percentiles = [i[0] for i in default_percentiles_cutoffs]

    # Compute noise parameters
    noise_samplers = []
    for per_query_epsilon in per_query_epsilon_list:
        if FLAGS.noise_type == "dgaussian":
            noise_parameter = DiscreteGaussianPrivacyLoss.from_privacy_guarantee(
                DifferentialPrivacyParameters(per_query_epsilon, FLAGS.delta)
            ).standard_deviation()
        else:
            noise_parameter = per_query_epsilon
        # Construct noise sampler
        noise_samplers.append(
            distribution_sampler.noise_sampler_factory(
                rng,
                epsilon=per_query_epsilon,
                noise_parameter=noise_parameter,
                noise_type=FLAGS.noise_type,
            )
        )
    print("Finish calculating noise parameters")

    errors_per_reach_and_metric = {}

    # Run noisy Liquid Legions simulations
    simulator = noisy_liquid_legions.NoisyLiquidLegionSimulator(
        a=FLAGS.decay_rate, m=FLAGS.sketch_size
    )
    for subsampling_buckets in sampling_buckets_list:
        for true_reach in true_reach_list:
            results = simulator.get_batch_estimates(
                n=true_reach,
                max_freq=FLAGS.max_freq,
                noise_samplers=noise_samplers,
                subsampling_rate=1.0 / subsampling_buckets,
                distribution=FLAGS.frequency_distribution,
            )

            for i, epsilon in enumerate(FLAGS.per_query_epsilon_list):
                for metric in metrics:
                    error_report = metric.report_error(
                        results[i],
                        error_percentiles=default_percentiles,
                        print_report=False,
                    )
                    for percentile, _ in default_percentiles_cutoffs:
                        errors_per_reach_and_metric[
                            str(metric),
                            percentile,
                            true_reach,
                            subsampling_buckets,
                            epsilon,
                        ] = error_report[percentile]

    # Produce the CSV file.
    if FLAGS.sketch_size == -1:
        # pure sampling
        concatenated_parameters = (
            f"Delta={FLAGS.delta},Distribution={FLAGS.frequency_distribution},"
            f"Max_freq={FLAGS.max_freq},Noise_type={FLAGS.noise_type}"
        )
    else:
        # sampling + LL
        concatenated_parameters = (
            f"Decay_rate={FLAGS.decay_rate},Sketch_size={FLAGS.sketch_size},"
            f"Delta={FLAGS.delta},Distribution={FLAGS.frequency_distribution},"
            f"Max_freq={FLAGS.max_freq},Noise_type={FLAGS.noise_type}"
        )
    # Header for each table; create one table for each accuracy bar.
    csv_header_string = "Reach,Sampling Buckets,Per-query Epsilons,Error\n"
    csv_file_string_list = [concatenated_parameters, "\n"]
    for metric in metrics:
        for percentile, _ in default_percentiles_cutoffs:
            csv_file_string_list_this_metric = []
            table_name = f"{metric} {percentile}-percentile"
            csv_file_string_list.append(f"{table_name}\n")
            csv_file_string_list_this_metric.append(csv_header_string)
            for true_reach in true_reach_list:
                for subsampling_buckets in sampling_buckets_list:
                    for epsilon in FLAGS.per_query_epsilon_list:
                        error_value = errors_per_reach_and_metric[
                            str(metric),
                            percentile,
                            true_reach,
                            subsampling_buckets,
                            epsilon,
                        ]
                        csv_file_string_list_this_metric.append(
                            f"{true_reach},{subsampling_buckets},{epsilon},"
                            f"{error_value:4f}\n"
                        )
            csv_table_string = "".join(csv_file_string_list_this_metric)
            csv_file_string_list.append(csv_table_string)
            table_path = f"{FLAGS.result_dir}{concatenated_parameters}:{table_name}.csv"
            with open(table_path, "w") as f:
                f.write(csv_table_string)
    csv_file_string = "".join(map(str, csv_file_string_list))

    result_path = f"{FLAGS.result_dir}{concatenated_parameters}.csv"
    with open(result_path, "w") as f:
        f.write(csv_file_string)


if __name__ == "__main__":
    app.run(main)
