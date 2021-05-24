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
"""Tests for overlap_data_set.py."""

from typing import Iterable
from typing import List
from typing import Tuple

from absl.testing import absltest
import numpy as np
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator import (
    DisjointSetGenerator,
)
from wfa_planning_evaluation_framework.data_generators.overlap_data_set import (
    OverlapDataSet,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)


class OverlapDatSetTest(absltest.TestCase):
    def assert_equal_pub_data_list(
        self,
        res: List[PublisherData],
        expected_len: int,
        expected_data_list: Iterable[Tuple[int, float]],
        expected_name_list: Iterable[str],
    ):
        self.assertLen(res, 3)
        for i in range(expected_len):
            self.assertEqual(set(res[i]._data), set(expected_data_list[i]))
            self.assertEqual(res[i].name, expected_name_list[i])

    def test_map_ids(self):
        set_ids_list = [np.array([3, 4, 5]), np.array([4, 6, 8]), np.array([6, 8, 10])]
        pdf1 = PublisherData([(2, 0.02), (1, 0.01), (1, 0.03), (3, 0.04)], "a")
        pdf2 = PublisherData([(3, 0.04), (1, 0.02), (2, 0.01)], "b")
        pdf3 = PublisherData(
            [(1, 0.01), (2, 0.02), (1, 0.04), (1, 0.01), (3, 0.05)], "c"
        )
        pdf_list = [pdf1, pdf2, pdf3]
        expected_data_list = [
            [(3, 0.01), (3, 0.03), (4, 0.02), (5, 0.04)],
            [(4, 0.02), (6, 0.01), (8, 0.04)],
            [(6, 0.01), (6, 0.01), (6, 0.04), (8, 0.02), (10, 0.05)],
        ]
        expected_name_list = ["a", "b", "c"]
        res = OverlapDataSet._label_ids(set_ids_list, pdf_list)
        self.assert_equal_pub_data_list(res, 3, expected_data_list, expected_name_list)
        res = OverlapDataSet._label_ids(iter(set_ids_list), iter(pdf_list))
        self.assert_equal_pub_data_list(res, 3, expected_data_list, expected_name_list)

    def test_disjoint_overlap_data_set(self):
        pdf1 = PublisherData([(2, 0.02), (1, 0.01), (1, 0.03), (3, 0.04)], "a")
        pdf2 = PublisherData([(3, 0.04), (1, 0.02), (2, 0.01)], "b")
        pdf3 = PublisherData(
            [(1, 0.01), (2, 0.02), (1, 0.04), (1, 0.01), (3, 0.05)], "c"
        )
        res = OverlapDataSet(
            unlabeled_publisher_data_list=[pdf1, pdf2, pdf3],
            overlap_generator=DisjointSetGenerator,
            name="disjoint",
        )
        self.assertEqual(res.name, "disjoint")
        expected_data_list = [
            [(0, 0.01), (0, 0.03), (1, 0.02), (2, 0.04)],
            [(3, 0.02), (4, 0.01), (5, 0.04)],
            [(6, 0.01), (6, 0.01), (6, 0.04), (7, 0.02), (8, 0.05)],
        ]
        expected_name_list = ["a", "b", "c"]
        self.assert_equal_pub_data_list(
            res._data, 3, expected_data_list, expected_name_list
        )


if __name__ == "__main__":
    absltest.main()
