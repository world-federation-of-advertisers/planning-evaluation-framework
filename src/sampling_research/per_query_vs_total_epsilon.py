"""Computes the number of total queries one can support for a given per-query epsilon and total epsilon."""

import math
import os
from typing import Sequence

from absl import app
from absl import flags

from dp_accounting.common import BinarySearchParameters
from dp_accounting.common import DifferentialPrivacyParameters
from dp_accounting.common import inverse_monotone_function
from dp_accounting.privacy_loss_distribution import PrivacyLossDistribution
from dp_accounting.privacy_loss_mechanism import DiscreteGaussianPrivacyLoss
from dp_accounting.privacy_loss_mechanism import DiscreteLaplacePrivacyLoss

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "result_dir", "./simulation_results/", "Directory to write result to."
)
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
    "Value of per-query DP parameter epsilons.",
)
flags.DEFINE_list(
    "total_epsilon_list",
    [
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "1.0",
        "1.2",
        "1.4",
        "1.6",
        "1.8",
        "2.0",
        "2.5",
        "3.0",
        "3.5",
        "4.0",
        "4.5",
        "5.0",
    ],
    "Value of total DP parameter epsilons.",
)
flags.DEFINE_float(
    "delta", 1e-9, "Value of DP parameter delta (both for per-query and total)."
)
flags.DEFINE_string(
    "noise_type", "dgaussian", "The type of noise to be added to each query."
)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    per_query_epsilon_list = list(map(float, FLAGS.per_query_epsilon_list))
    total_query_epsilon_list = list(map(float, FLAGS.total_epsilon_list))

    # Compute max # of queries we can support
    discretization_interval = 1e-5
    max_num_queries_dict = {}
    for per_query_epsilon in per_query_epsilon_list:
        if FLAGS.noise_type == "dgaussian":
            additive_noise_privacy_loss = (
                DiscreteGaussianPrivacyLoss.from_privacy_guarantee(
                    DifferentialPrivacyParameters(per_query_epsilon, FLAGS.delta)
                )
            )
        elif FLAGS.noise_type == "dlaplace":
            additive_noise_privacy_loss = (
                DiscreteLaplacePrivacyLoss.from_privacy_guarantee(
                    DifferentialPrivacyParameters(per_query_epsilon, FLAGS.delta)
                )
            )

        pld = PrivacyLossDistribution.create_from_additive_noise(
            additive_noise_privacy_loss,
            value_discretization_interval=discretization_interval,
        )
        for total_epsilon in total_query_epsilon_list:
            # pylint: disable=cell-var-from-loop
            max_num_queries = inverse_monotone_function(
                lambda x: pld.self_compose(x).get_epsilon_for_delta(FLAGS.delta),
                total_epsilon,
                BinarySearchParameters(
                    0,
                    math.inf,
                    initial_guess=math.ceil((total_epsilon / per_query_epsilon)),
                    discrete=True,
                ),
                increasing=True,
            )
            print(
                f"Max # queries for per-query epsilon {per_query_epsilon} and "
                f"total epsilon {total_epsilon}: {max_num_queries}"
            )
            max_num_queries_dict[per_query_epsilon, total_epsilon] = max_num_queries

    # Produce the CSV file.
    csv_header_string = "per-query epsilon,total epsilon,num queries\n"
    csv_file_string_list = [csv_header_string]
    for per_query_epsilon in per_query_epsilon_list:
        for total_epsilon in total_query_epsilon_list:
            csv_file_string_list.append(
                f"{per_query_epsilon},{total_epsilon},"
                f"{max_num_queries_dict[per_query_epsilon,total_epsilon]}\n"
            )
    csv_file_string = "".join(map(str, csv_file_string_list))

    if not os.path.isdir(FLAGS.result_dir):
        os.mkdir(FLAGS.result_dir)
    result_path = f"{FLAGS.result_dir}per_query_vs_total_epsilon_{FLAGS.noise_type}.csv"
    with open(result_path, "w") as f:
        f.write(csv_file_string)


if __name__ == "__main__":
    app.run(main)
