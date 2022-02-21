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
"""Base class for describing a multi publisher inventory."""

from __future__ import annotations

import numpy as np
from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.data_set import (
    DataSet,
)


class InventoryFrequencyVectorDistribution:
    """Base class of the inventory distribution of multi-pub frequency vector.

    An inventory can be represented as a distribution of each user's
    frequency vector.  A frequency vector, say, [3, 4, 0] means this user has
    been reached 3 times at pub 1, 4 times at pub 2, and 0 time at pub 3.  A
    toy example of frequency vector is: probability = 2 / 3 at vector [2, 1]
    and probability =  1 / 3 at vector [0, 3], which means (2 / 3) users in
    the universe are reached 2 times at pub 1 and 1 time at pub 2, while the
    remaining (1 / 3) users are reached 0 time at pub 1 and 3 times at pub 2.

    In the rest of this codebase, we represent a multi pub inventory as an
    DataSet which consists of the impression logs of all publishers.  Under
    the fixed price assumption, it is equivalent to the distribution
    representation in this class.  In other words, one can translate an
    InventoryFrequencyVectorDistribution instance to impression logs, and can
    also translate impression logs to a InventoryFrequencyVectorDistribution
    instance.
    """

    def estimate_campaign_pmf(
        self, campaign_impression_fractions: np.ndarray, max_freq: int
    ) -> np.ndarray:
        """Estimate the frequency histogram of any random subset of inventory.

        An InventoryFrequencyVectorDistribution does not need to store the
        probability masses on the high dimensional frequency vectors. An
        InventoryFrequencyVectorDistribution is valid if and only if it
        realizes this method --- tell the frequency distribution along any
        direction, i.e., of any hypothetical campaign as a random subset of
        the inventory.

        Note that many frequency vector distributions (such as the
        independent distribution), calculation of 1 dimensional campaign
        frequency histogram does not need integral of high dimensional pmfs.

        Args:
            campaign_impression_fractions:  A vector v where v[i] equals
                #impressions(campaign on pub i) / #impressions(inventory on
                pub i).
            max_freq:  Maximum frequency of the output probability mass
                function (pmf) vector.

        Returns:
            The pmf vector of the total frequency when a campaign delivers
            these many impressions on different pubs.
        """
        raise NotImplementedError()

    def to_impression_logs(self, fixed_price: float, universe_size: int) -> DataSet:
        """Translate an InventoryFrequencyVectorDistribution to a DataSet.

        Args:
            fixed_price:  The common price of all the impressions.
            universe_size:  Number of users in the multi pub universe.

        Returns:
            A DataSet instance that has the same distribution of frequency vector.
        """
        raise NotImplementedError()

    @classmethod
    def to_single_impression_log(
        cls, pmf: np.ndarray, fixed_price: float, universe_size: int
    ) -> PublisherData:
        """Translate a frequency distribution to a single impression log.

        Args:
            pmf:  Probability mass function of frequency, from 0 to a maximum
                frequency.  It can refer to the frequency at a single publisher
                or along a single direction.
            fixed_price:  The common price of all the impressions.
            universe_size:  Number of users in the multi pub universe.+

        Returns:
            A PublisherData instance that has the same frequency distribution.
        """
        raise NotImplementedError()

    @classmethod
    def truncate_histogram(cls, histogram: np.ndarray, new_max_freq) -> np.ndarray:
        """Truncate a histogram to a new max frequency.

        Args:
            histogram: 1d array that starts from frequency=0 to a max frequency.
            new_max_freq: A new max frequency that we want to truncate the
                histogram up to.

        Returns:
            The truncated histogram.
        """
        if new_max_freq >= len(histogram) - 1:
            return histogram
        new_hist = histogram[: (new_max_freq + 1)].copy()
        new_hist[-1] = sum(histogram[new_max_freq:])
        return new_hist

    @classmethod
    def estimate_histogram_after_single_direction_sampling(
        cls, histogram: np.ndarray, impression_fraction: float
    ) -> np.ndarray:
        """Expected frequency histogram after sampling along one direction.

        Args:
            histogram:  The frequency histogram or pmf before sampling.
            impression_fraction:  Sampling this fraction of impressions.
                - If the input histogram represents N impressions on only one
                pub, then we randomly choosing roughly (N * impression_fraction)
                out of the N impressions.
                - If  the input histogram represents N_1 impressions on pub 1,
                ..., N_p impressions on pub p, then we do proportional sampling,
                i.e., roughly (N_1 * impression_fraction) impressions on pub 1,
                ..., roughly (N_p * impression_fraction) impressions on pub p.

        Returns:
            The expected frequency histogram after sampling.
        """
        if impression_fraction < 0 or impression_fraction > 1:
            raise ValueError("Invalid impression fraction")
        return sum(
            [
                stats.binom.pmf(range(len(histogram)), n=f, p=impression_fraction)
                * histogram[f]
                for f in range(len(histogram))
            ]
        )

    @classmethod
    def visualize_single_pub(cls, pmf_vector: np.ndarray):
        """Visualize marginal pmf for frequency at a single publisher.

        Args:
            pmf_vec: Any 1d array that represents the pmf from 0 to a max frequency.
        """
        plt.figure(figsize=(3, 3))
        plt.stem(
            pmf_vector,
            linefmt="deepskyblue",
            markerfmt="o",
            basefmt="grey",
            use_line_collection=True,
        )
        plt.show()

    @classmethod
    def visualize_two_pubs(cls, pmf_matrix: np.ndarray):
        """Visualize joint pmf for frequencies at two publishers.

        A use case is to check if the two publishers are correlated in an
        expected manner.

        Args:
            pmf_matrix: Any 2d array of which pmf_matrix[i, j] is the probability
                of observing frequency.
        """
        sns.heatmap(pmf_matrix)
        plt.show()
        m, n = pmf_matrix.shape
        x_mat, y_mat = np.meshgrid(range(m), range(n))
        ax = plt.axes(projection="3d")
        ax.plot_surface(x_mat, y_mat, pmf_matrix.transpose(), cmap="Blues", alpha=0.5)
        plt.show()
