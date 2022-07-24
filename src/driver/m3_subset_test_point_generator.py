# Copyright 2022 The Private Cardinality Estimation Framework Authors
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
"""Generates test points for the subset use case with the m3 strategy."""

from typing import Iterable, List

import numpy as np
from itertools import combinations
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.driver.test_point_generator import (
    TestPointGenerator,
)


class M3SubsetTestPointGenerator(TestPointGenerator):
    """Generates test points for the subset use case with the m3 strategy.

    In M3, the training points are the frequency histograms of
    - each of the single publishers,
    - each of the all-but-one subsets of publishers,
    - the union set of all publishers,
    a total of (2p + 1) subsets for training, p being the number of publishers.
    This class generates the test points that represent the remaining
    (2^p - 1) - (2p + 1) subsets, for the use case that customers use the
    planning model to predict the r/f of unobserved subsets.

    When the number of subsets (2^p - 1) - (2p + 1) is too large, randomly
    sample subsets for testing.
    """

    def __init__(
        self,
        dataset: DataSet,
        campaign_spend_fractions: np.ndarray,
        max_num_points: int = 2000,
        rng: np.random.Generator = np.random.default_rng(0),
    ):
        """Constructs an M3SubsetTestPointGenerator.

        Args:
            dataset:  The DataSet for which test points are to be generated.
            campaign_spend_fractions:  The campaign_spend / inventory_spend
                at each publisher, where each inventory_spend is given by
                dataset._max_spends.
            max_num_points:  If (2^p - 1) - (2p + 1) > `max_num_points`, then
                randomly sample `max_num_points` subsets from the
                (2^p - 1) - (2p + 1) subsets.
            rng:  A numpy Generator object that is used to seed the generation
                of random test points.
        """
        super().__init__(dataset)
        self._rng = rng
        self.max_num_points = max_num_points
        self._campaign_spends = self._max_spends * campaign_spend_fractions

    @staticmethod
    def subset_layer(p: int, q: int) -> List[np.ndarray]:
        """Returns a list of subset indicators for a layer of subsets.

        For example, when p = 3, and q = 2, returns
        [[1, 1, 0],  # indicating subset {A, B}
         [1, 0, 1],  # indicating subset {A, C}
         [0, 1, 1]]   # indicating subset {B, C}
        In general, a subset indicator is a binary vector v where v[i] = 1
        if and only if pub i is in the subset.
        A layer of subsets means the collection of subsets with the same
        cardinality.

        Args:
            p: Number of all publishers.
            q: Number of publishers in the subsets.

        Returns:
            A length <p choose q> list of length <p> arrays.
            Each array is a different binary vector that sums up to 1.
        """

        def one_direction(indices: List) -> np.ndarray:
            x = np.zeros(p)
            x[indices] = 1
            return x

        return [one_direction(list(indices)) for indices in combinations(range(p), q)]

    def test_points(self) -> Iterable[List[float]]:
        """Returns a generator for generating a list of test points.

        Returns:
            An iterable of spend vectors representing locations where
            the true reach surface is to be compared to the modeled reach
            surface.
        """
        p = self._npublishers
        if p < 4:
            return
        if (2**p - 1) - (2 * p + 1) > self.max_num_points:
            n = 0
            subset_collection = set()
            while n < self.max_num_points:
                subset_indicator = self._rng.choice([0, 1], p)
                num_involved_subsets = np.count_nonzero(subset_indicator)
                if 1 < num_involved_subsets < p - 1:
                    if tuple(subset_indicator) not in subset_collection:
                        subset_collection.add(tuple(subset_indicator))
                        yield list(self._campaign_spends * subset_indicator)
                        n += 1
        else:
            for q in range(2, p - 1):
                for subset_indicator in self.subset_layer(p, q):
                    yield list(self._campaign_spends * subset_indicator)
