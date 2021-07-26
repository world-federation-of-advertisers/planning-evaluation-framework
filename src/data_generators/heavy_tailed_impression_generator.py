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
"""Generate a stream of impressions from a zeta distr."""

from typing import List
import numpy as np

from wfa_planning_evaluation_framework.data_generators.impression_generator import (
    ImpressionGenerator,
)


class HeavyTailedImpressionGenerator(ImpressionGenerator):
    """Generate impressions with heavy tailed impressions."""

    def __init__(
        self, n: int, zeta_s: float = 2, random_generator: np.random.Generator = None
    ):
        """Constructor for the HeavyTailedImpressionGenerator.

        For each user, the number of impressions assigned to that user is
        determined by drawing from a zeta distribution which has pmf
        p(k) = k^{-s} / zeta(s), k = 1, 2, ..., zeta(s) being a normalizing factor.

        Args:
          n:  The number of users.
          zeta_s:  The parameter of the zeta distribution.
          random_generator:  An instance of numpy.random.Generator that is
            used for making draws from the Zeta distribution.
        """
        assert zeta_s > 1, "Zeta distribution must have power parameter > 1."
        self._zeta_s = zeta_s
        self._n = n
        if random_generator:
            self._random_generator = random_generator
        else:
            self._random_generator = np.random.default_rng(seed=1)

    def __call__(self) -> List[int]:
        """Generate a random sequence of impressions.

        Returns:
          A list of randomly generated user id's. An id may occur multiple
          times in the output list, representing the fact that the user may
          see multiple ads from the publisher over the course of the campaign.
        """
        impressions = []
        for i in range(self._n):
            impressions.extend([i] * self._random_generator.zipf(self._zeta_s))
        self._random_generator.shuffle(impressions)
        return impressions
