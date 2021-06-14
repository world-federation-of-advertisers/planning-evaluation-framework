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
"""Tests for grid_test_point_generator.py."""

from absl.testing import absltest
import numpy as np
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.driver.grid_test_point_generator import (
    GridTestPointGenerator,
)


class GridTestPointGeneratorTest(absltest.TestCase):
    def test_one_publisher(self):
        pdf = PublisherData([(1, 100.0)], "pdf")
        data_set = DataSet([pdf], "test")
        generator = GridTestPointGenerator(
            data_set, np.random.default_rng(1), grid_size=4
        )
        values = [int(x[0]) for x in generator.test_points()]
        self.assertLen(values, 4)
        self.assertEqual(values, [20, 40, 60, 80])

    def test_two_publishers(self):
        pdf1 = PublisherData([(1, 3.0)], "pdf1")
        pdf2 = PublisherData([(1, 6.0)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "test")
        generator = GridTestPointGenerator(
            data_set, np.random.default_rng(1), grid_size=2
        )
        values = [(int(x[0]), int(x[1])) for x in generator.test_points()]
        self.assertLen(values, 4)
        self.assertEqual(values, [(1, 2), (1, 4), (2, 2), (2, 4)])


if __name__ == "__main__":
    absltest.main()
