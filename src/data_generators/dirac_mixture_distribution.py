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
"""Dirac mixture, i.e., Mixed Poisson distribution of frequency vector."""

import numpy as np

from wfa_planning_evaluation_framework.data_generators.inventory_frequency_vector_distribution import (
    InventoryFrequencyVectorDistribution,
)
from wfa_planning_evaluation_framework.models.dirac_mixture_single_publisher_model import (
    UnivariateMixedPoissonOptimizer,
)


class DiracMixtureDistribution(InventoryFrequencyVectorDistribution):
    def __init__(self, component_matrix: np.ndarray, weights: np.ndarray):
        self.ncomponents, self.p = component_matrix.shape
        if len(weights) != self.ncomponents:
            raise ValueError("Number of components do no match")
        self.component_matrix = component_matrix.astype("float")
        self.weights = weights.astype("float")

    def estimate_campaign_pmf(
        self, campaign_impression_fractions: np.ndarray, max_freq: int = 10
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
        projected_components = self.component_matrix.dot(campaign_impression_fractions)
        project_component_pmfs = np.array(
            [
                UnivariateMixedPoissonOptimizer.truncated_poisson_pmf_vec(
                    poisson_mean=component, max_freq=max_freq
                )
                for component in projected_components
            ]
        )
        return project_component_pmfs.transpose().dot(self.weights)
