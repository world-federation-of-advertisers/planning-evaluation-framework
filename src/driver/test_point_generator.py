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
"""Test point generator.

Generates a collection of test points that can be used for evaluating
a reach curve or surface.

The set of test points depends only on the DataSet for which the
points are to be generated.  It is generated deterministically,
so that different modeling strategies that are evaluated against
the same DataSet will be evaluated at the same test points.  The number
of test points may be a function of the test point generation algorithm
and need not be a configuration parameter.

A good test point generator will generate a minimual number of test 
points that evenly sample the reach surface.
"""

import numpy as np
from typing import Iterable
from typing import List
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet

# The minimum number of test points that will be generated.
# The value below was chosen heuristically on the belief that this would
# give an acceptably small sampling variance for the modeling errors.
MINIMUM_NUMBER_OF_TEST_POINTS = 100


class TestPointGenerator:
    """Generates a collection of test points for a given simulation."""

    def __init__(self, dataset: DataSet):
        """Creates a test point generator for the given DataSet.

        Args:
          dataset:  The underlying data from which the reach surface model
            is constructed, and for which test points should be generated.
        """
        self._npublishers = dataset.publisher_count
        self._max_spends = np.array(
            [dataset._data[i].max_spend for i in range(dataset.publisher_count)]
        )

    def test_points(self) -> Iterable[List[float]]:
        """Returns a generator for generating a list of test points.

        Returns:
          An iterable of spend vectors representing locations where
          the true reach surface is to be compared to the modeled reach
          surface.
        """
        raise NotImplementedError()
