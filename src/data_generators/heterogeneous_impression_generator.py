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
"""Generate a stream of impressions from a Poisson distr with Gamma prior."""

from typing import List
from numpy.random import RandomState

from wfa_planning_evaluation_framework.data_generators.impression_generator import (
    ImpressionGenerator,
)


class HeterogeneousImpressionGenerator(ImpressionGenerator):
    """Generate impressions with heterogeneous impressions."""

    def __init__(
        self,
        n: int,
        gamma_shape: float = 1,
        gamma_scale: float = 1,
        random_state: RandomState = None,
    ):
        """Constructor for the HeterogeneousImpressionGenerator.

        For each user, the number of impressions assigned to that user is
        determined by drawing from a shifted (by +1) Poisson distribution with
        parameter lambda, with lambda drawn from a Gamma distribution. Explicitly,
        lambda is generated from a pdf
        p(x) = x^{shape - 1} * exp(x / scale) / scale^shape / Gamma(shape),
        where shape and gamma are two parameters and Gamma is the Gamma function.
        In this way, lambda has a mean of shape * scale and a variance of
        shape * scale^2.
        Note: this is equivalent to a negative binomial distribution.

        Args:
          n:  The number of users.
          gamma_shape:  The shape parameter of the Gamma distribution.
          gamma_scale:  The scale parameter of the Gamma distribution.
          random_state:  An instance of numpy.random.RandomState that is
            used for making draws from the Poisson distribution.
        """
        self._gamma_shape = gamma_shape
        self._gamma_scale = gamma_scale
        self._n = n
        if random_state:
            self._random_state = random_state
        else:
            self._random_state = RandomState()

    def __call__(self) -> List[int]:
        """Generate a random sequence of impressions.

        Returns:
          A list of randomly generated user id's.  An id may occur multiple
          times in the output list, representing the fact that the user may
          see multiple ads from the publisher over the course of the campaign.
        """
        impressions = []
        for i in range(self._n):
            poisson_lambda = self._random_state.gamma(
                shape=self._gamma_shape, scale=self._gamma_scale
            )
            impressions.extend([i] * (1 + self._random_state.poisson(poisson_lambda)))
        self._random_state.shuffle(impressions)
        return impressions
