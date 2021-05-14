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
"""Generate a stream of impressions from a poisson distr with fixed lambda."""

from typing import List
from numpy.random import Generator

from wfa_planning_evaluation_framework.data_generators.impression_generator import (
    ImpressionGenerator,)


class HomogeneousImpressionGenerator(ImpressionGenerator):
<<<<<<< HEAD
    """Generate ad impressions using Poisson distribution with fixed lambda."""
=======
  """Generate a random sequence of viewer id's of ad impressions.

    This class, along with PricingGenerator, assists in the generation of
    random PublisherDataFiles.  The ImpressionGenerator will generate a
    sequence of random impressions according to specified criteria.
    """
>>>>>>> 79e8ddd (addressed comments)

  def __init__(self,
               n: int,
               poisson_lambda: float,
               random_generator: Generator = None):
    """Constructor for the HomogeneousImpressionGenerator.

        For each user, the number of impressions assigned to that user is
        determined by drawing from a shifted Poisson distribution with fixed
        parameter lambda.  The Poisson distribution is shifted by one.  E.g.,
        the PMF is given by Pr(X=k) = lambda^{k-1} e^{-lambda} / (k-1)!.

        Args:
          n:  The number of users.
          poisson_lambda:  The parameter of the Poisson distribution that
            determines viewing frequencies.
          random_state:  An instance of numpy.random.RandomState that is used
            for making draws from the Poisson distribution.
    """
    self._poisson_lambda = poisson_lambda
    self._n = n
    if random_generator:
      self.random_generator = random_generator
    else:
      self.random_generator = Generator()

<<<<<<< HEAD
    def __call__(self) -> List[int]:
        """Generate impressions using Poisson distribution with fixed lambda.
=======
  def __call__(self) -> List[int]:
    """Generate a random sequence of impressions.
>>>>>>> 79e8ddd (addressed comments)

        Returns:
          A list of randomly generated user id's.  An id may occur multiple
          times in the output list, representing the fact that the user may
          see multiple ads from the publisher over the course of the campaign.
        """
    impressions = []
    for i in range(self._n):
      impressions.extend([i] *
                         (1 + self.random_generator.poisson(self._poisson_lambda)))
    self.random_generator.shuffle(impressions)
    return impressions
