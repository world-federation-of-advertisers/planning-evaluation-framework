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
"""Different copula distributions of frequency vector."""

import numpy as np
from typing import List, Dict
from scipy import stats

from wfa_planning_evaluation_framework.data_generators.inventory_frequency_vector_distribution import (
    InventoryFrequencyVectorDistribution,
)
from wfa_planning_evaluation_framework.models.dirac_mixture_single_publisher_model import (
    UnivariateMixedPoissonOptimizer,
)


class CopulaDistribution(InventoryFrequencyVectorDistribution):
    """Construct joint distribution given marginal distributions."""

    def __init__(self, marginal_frequency_histograms: List):
        self.p = len(marginal_frequency_histograms)  # number of publishers
        self.inventory_max_freq = len(marginal_frequency_histograms[0]) - 1
        for hist in marginal_frequency_histograms:
            UnivariateMixedPoissonOptimizer.validate_frequency_histogram(hist)
            if len(hist) != self.inventory_max_freq + 1:
                raise AssertionError("Not all pubs have the same inventory max freq")
        self.marginal_pmfs = [
            hist / sum(hist) for hist in marginal_frequency_histograms
        ]


class IndependentCopulaDistribution(CopulaDistribution):
    """Independently join the single pub inventories."""

    @classmethod
    def pairwise_convolution(
        cls, this_pmf: np.ndarray, that_pmf: np.ndarray
    ) -> np.ndarray:
        """Convolution of two pmf vectors:

        Args:
            this_pmf:  One pmf vector.
            that_pmf:  Another pmf vector of the same max_freq as this_pmf.

        Returns:
            A pmf v of the same length of this_pmf where
            v[f] = sum_{i=0}^f this_pmf[i] that_pmf[j]
            for 0 <= f < max_freq, and the last element
            v[max_freq] = 1 - sum_{i=0}^{max_freq - 1} v[i].
        """
        if len(this_pmf) != len(that_pmf):
            raise ValueError("Please set same max_freq for the two pmfs")
        total_pmf = np.zeros(len(this_pmf))
        for i in range(len(this_pmf) - 1):
            total_pmf[i] = sum(this_pmf[: (i + 1)] * that_pmf[: (i + 1)][::-1])
        total_pmf[-1] = 1 - sum(total_pmf)
        return total_pmf

    @classmethod
    def convolution(cls, pmfs: List) -> np.ndarray:
        """Convolute a list of pmfs sequentially.

        Binary-tree convolution can reduce the complexity, but does not quite
        necessary for <=50 publishers.

        Args:
            pmfs: A list of pmf vectors.

        Returns:
            If the pmfs are of variables X_1, ..., X_p respectively, then
            returns the pmf vector of X_1 + ... + X_p
        """
        total_pmf = np.zeros(len(pmfs[0]))
        total_pmf[0] = 1
        for pmf in pmfs:
            total_pmf = cls.pairwise_convolution(total_pmf, pmf)
        return total_pmf

    def estimate_campaign_pmf(
        self, campaign_impression_fractions: np.ndarray, max_freq: int = None
    ) -> np.ndarray:
        """Estimate the frequency histogram of any random subset of inventory.

        Args:
            campaign_impression_fractions:  A vector v where v[i] equals
                #impressions(campaign on pub i) / #impressions(inventory on
                pub i).
            max_freq:  Maximum frequency of the output pmf vector.

        Returns:
            pmf vector of the total frequency when a campaign delivers these
            many impressions on different pubs.
        """
        if len(campaign_impression_fractions) != self.p:
            raise ValueError("Number of publishers do not match")
        single_pub_pmfs = [
            self.estimate_histogram_after_single_direction_sampling(
                histogram=inventory_pmf, impression_fraction=impression_fraction
            )
            for inventory_pmf, impression_fraction in zip(
                self.marginal_pmfs, campaign_impression_fractions
            )
        ]
        return self.truncate_histogram(
            histogram=self.convolution(single_pub_pmfs),
            new_max_freq=self.inventory_max_freq if max_freq is None else max_freq,
        )


class GaussianCopulaDistribution(CopulaDistribution):
    """Glue single publisher frequency distributions using Gaussian copula."""

    def __init__(
        self, marginal_frequency_histograms: np.ndarray, correlation_matrix: np.ndarray
    ):
        super().__init__(marginal_frequency_histograms)
        raise NotImplementedError()
