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
"""Grid test point generator.

Generates an evenly spaced grid of test points.
"""

import itertools
import numpy as np
from typing import Iterable
from typing import List
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.driver.test_point_generator import (
    TestPointGenerator,
)


class GridTestPointGenerator(TestPointGenerator):
    """Generates a collection of evenly spaced test points."""

    def __init__(self, dataset: DataSet, rng: np.random.Generator, grid_size: int):
        """Returns a GridTestPointGenerator.

        Args:
          dataset:  The DataSet for which test points are to be generated.
          rng:  A numpy Generator object that is used to seed the generation
            of random test points.
          grid_size: The number of points that should be generated along the
            grid for each dimension.  Thus, if grid_size is 5 and there are
            three publishers, the total number of test points is 5**3 = 125.
        """
        super().__init__(dataset)
        self._rng = rng
        self._grid_size = grid_size

    def test_points(self) -> Iterable[List[float]]:
        """Returns a generator for generating a list of test points.

        Returns:
          An iterable of spend vectors representing locations where
          the true reach surface is to be compared to the modeled reach
          surface.
        """
        points_per_dimension = []
        for i in range(self._npublishers):
            points_per_dimension.append(
                list(
                    self._max_spends[i]
                    * np.arange(1, self._grid_size + 1)
                    / (self._grid_size + 1)
                )
            )
        for point in itertools.product(*points_per_dimension):
            yield point
