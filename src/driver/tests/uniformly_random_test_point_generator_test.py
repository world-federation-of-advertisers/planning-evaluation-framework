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
"""Tests for uniformly_random_test_point_generator.py."""

from absl.testing import absltest
import numpy as np
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.driver.uniformly_random_test_point_generator import (
    UniformlyRandomTestPointGenerator,
)


class UniformlyRandomTestPointGeneratorTest(absltest.TestCase):
    def test_one_publisher(self):
        pdf = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf")
        data_set = DataSet([pdf], "test")
        generator = UniformlyRandomTestPointGenerator(
            data_set, np.random.default_rng(1)
        )
        values = [x for x in generator.test_points()]
        self.assertLen(values, 100)
        for i, v in enumerate(values):
            self.assertLen(v, 1)
            self.assertTrue(v[0] >= 0.0, "Item {} is negative: {}".format(i, v))
            self.assertTrue(v[0] < 0.05, "Item {} is too large: {}".format(i, v))

    def test_two_publishers(self):
        pdf1 = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf1")
        pdf2 = PublisherData([(1, 0.02), (2, 0.04), (1, 0.08), (3, 0.10)], "pdf2")
        data_set = DataSet([pdf1, pdf2], "test")
        generator = UniformlyRandomTestPointGenerator(
            data_set, np.random.default_rng(1)
        )
        values = [x for x in generator.test_points()]
        self.assertLen(values, 100)
        for i, v in enumerate(values):
            self.assertLen(v, 2)
            self.assertTrue(v[0] >= 0.0, "Item {} is negative: {}".format(i, v))
            self.assertTrue(v[0] < 0.05, "Item {} is too large: {}".format(i, v))
            self.assertTrue(v[1] >= 0.0, "Item {} is negative: {}".format(i, v))
            self.assertTrue(v[1] < 0.10, "Item {} is too large: {}".format(i, v))

    def test_fifty_publishers(self):
        pdf_list = []
        for i in range(50):
            pdf = PublisherData(
                [(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)], "pdf{}".format(i)
            )
            pdf_list.append(pdf)
        data_set = DataSet(pdf_list, "test")
        generator = UniformlyRandomTestPointGenerator(
            data_set, np.random.default_rng(1)
        )
        values = [x for x in generator.test_points()]
        self.assertLen(values, 2500)


if __name__ == "__main__":
    absltest.main()
