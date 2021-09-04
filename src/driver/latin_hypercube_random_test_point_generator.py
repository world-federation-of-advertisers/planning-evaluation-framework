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
"""Latin Hypercube random test point generator."""

from typing import Iterable
from typing import List
from typing import Callable

import numpy as np
import pyDOE
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.driver.test_point_generator import (
    TestPointGenerator,
)
from wfa_planning_evaluation_framework.driver.test_point_generator import (
    MINIMUM_NUMBER_OF_TEST_POINTS,
    MINIMUM_NUMBER_OF_TEST_POINTS_PER_PUBLISHER,
)

# The maximum number of test points that will be generated.
# The value below was chosen to guarantee that a Latin Hypercube is generated
# in a reasonable amount of time.
MAXIMUM_NUMBER_OF_TEST_POINTS = 2000
# A fact: 2000 points for 50 publishers were generated in 145 CPU seconds.


class LatinHypercubeRandomTestPointGenerator(TestPointGenerator):
    """Generates a collection of test points for a given simulation."""

    def __init__(
        self,
        dataset: DataSet,
        rng: np.random.Generator,
        npoints: int = MINIMUM_NUMBER_OF_TEST_POINTS,
        npublishers: int = 1,
        minimum_points_per_publisher=MINIMUM_NUMBER_OF_TEST_POINTS_PER_PUBLISHER,
    ):
        """Returns a LatinHypercubeRandomTestPointGenerator.

        Args:
          dataset:  The DataSet for which test points are to be generated.
          rng:  A numpy Generator object that is used to seed the generation
            of random test points.
          npoints: Minimum number of points to generate.
          npublishers: int.  The number of publishers in the dataset for which
            this evaluation is being performed.  This is used to determine the
            minimum number of points to generate.
          minimum_points_per_publisher:  The minimum number of test points to
            generate per publisher.  If this value times npublishers is greater
            than npoints, then it overrides npoints.
        """
        super().__init__(dataset)
        self._rng = rng
        self._npoints = max(npoints, npublishers * minimum_points_per_publisher)

    def test_points(self) -> Iterable[List[float]]:
        """Returns a generator for generating a list of test points.

        Returns:
          An iterable of spend vectors representing locations where
          the true reach surface is to be compared to the modeled reach
          surface.  A minimum of 100 points will be generated.  This value
          was chosen heuristically on the belief that this would give an
          acceptably small sampling variance for the modeling errors.
        """
        num_points = min(
            max(self._npoints, MINIMUM_NUMBER_OF_TEST_POINTS),
            MAXIMUM_NUMBER_OF_TEST_POINTS,
        )
        design = pyDOE.lhs(n=self._npublishers, samples=num_points, criterion="maximin")
        design = design[self._rng.permutation(num_points), :][
            :, self._rng.permutation(self._npublishers)
        ]
        for point in design:
            yield list(self._max_spends * point)
