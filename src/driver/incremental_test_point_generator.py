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


class IncrementalTestPointGenerator(TestPointGenerator):
    """Generates a sequence of subset reach from which we can calculate incremental reach."""

    def __init__(
        self,
        dataset: DataSet,
        campaign_spend_fractions: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(0),
    ):
        super().__init__(dataset)
        self._rng = rng
        self._campaign_spends = self._max_spends * campaign_spend_fractions

    @staticmethod
    def sequential_subset_indicator(sequence: List[int]) -> List[np.ndarray]:
        indicator = np.zeros(len(sequence))
        ret = []
        for i in sequence:
            indicator[i] = 1
            ret.append(indicator.copy())
        return ret

    def test_points(self) -> Iterable[List[float]]:
        """Returns a generator for generating a list of test points.

        Returns:
            An iterable of spend vectors representing locations where
            the true reach surface is to be compared to the modeled reach
            surface.
        """
        p = self._npublishers
        num_sequences = int(2 * p)
        sequence = [i for i in range(p)]
        for _ in range(num_sequences):
            self._rng.shuffle(sequence)
            for indicator in self.sequential_subset_indicator(sequence):
                print(
                    "\n\n",
                    indicator,
                    "\n",
                    list(self._campaign_spends * indicator),
                    "\n\n",
                )
                yield list(self._campaign_spends * indicator)
