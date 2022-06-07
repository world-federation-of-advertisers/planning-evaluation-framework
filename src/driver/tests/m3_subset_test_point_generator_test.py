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
"""Tests for m3_subset_test_point_generator.py."""

from typing import ValuesView
from absl.testing import absltest
import numpy as np
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.driver.m3_subset_test_point_generator import (
    M3SubsetTestPointGenerator,
)


class M3SubsetTestPointGeneratorTest(absltest.TestCase):
    def test_subset_layer(self):
        points = M3SubsetTestPointGenerator.subset_layer(5, 3)
        self.assertLen(points, 10)
        for point in points:
            self.assertEqual(np.count_nonzero(point), 3)

    def test_three_publisher(self):
        pdf1 = PublisherData([(1, 3.0)], "pdf1")
        pdf2 = PublisherData([(1, 6.0), (2, 3.0)], "pdf2")
        pdf3 = PublisherData([(1, 3.0), (3, 6.0)], "pdf3")
        dataset = DataSet([pdf1, pdf2, pdf3], "test")
        generator = M3SubsetTestPointGenerator(
            dataset=dataset,
            campaign_spend_fractions=np.array([0.1, 0.2, 0.3]),
        )
        points = [int(x[0]) for x in generator.test_points()]
        # No subset test points for 3 publishers
        self.assertLen(points, 0)

    def test_four_publisher(self):
        pdf1 = PublisherData([(1, 3.0)], "pdf1")
        pdf2 = PublisherData([(1, 6.0), (2, 3.0)], "pdf2")
        pdf3 = PublisherData([(1, 3.0), (3, 6.0)], "pdf3")
        pdf4 = PublisherData([(1, 3.0), (3, 6.0), (4, 9.0)], "pdf4")
        dataset = DataSet([pdf1, pdf2, pdf3, pdf4], "test")
        generator = M3SubsetTestPointGenerator(
            dataset=dataset,
            campaign_spend_fractions=np.array([0.1, 0.2, 0.3, 0.4]),
        )
        points = [x for x in generator.test_points()]
        self.assertLen(points, 6)
        np.testing.assert_almost_equal(
            points[0], [0.1 * 3.0, 0.2 * 6.0, 0, 0], decimal=3
        )

    def test_when_num_subsets_exceeds_max_num_points(self):
        pdfs = []
        for _ in range(4):
            pdfs.append(PublisherData([(1, 1.0)]))
        dataset = DataSet(pdfs, "test")
        generator = M3SubsetTestPointGenerator(
            dataset=dataset,
            campaign_spend_fractions=np.ones(4),
            max_num_points=3,
        )
        points = [x for x in generator.test_points()]
        self.assertLen(points, 3)
        for point in points:
            self.assertEqual(np.count_nonzero(point), 2)
        self.assertTrue(any([x != y for x, y in zip(points[0], points[1])]))
        self.assertTrue(any([x != y for x, y in zip(points[0], points[2])]))
        self.assertTrue(any([x != y for x, y in zip(points[1], points[2])]))


if __name__ == "__main__":
    absltest.main()
