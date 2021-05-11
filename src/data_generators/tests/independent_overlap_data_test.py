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
"""Tests for independent_overlap_data_set.py."""

from typing import Iterable
from typing import List
from typing import Tuple

from absl.testing import absltest
import numpy as np
from wfa_planning_evaluation_framework.data_generators.independent_overlap_data_set import (
    IndependentOverlapDataSet,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)


class IndependentOverlapDataSetTest(absltest.TestCase):
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

    def test_independent_overlap_data_set(self):
        pdf1 = PublisherData([(2, 0.02), (1, 0.01), (1, 0.03), (3, 0.04)], "a")
        pdf2 = PublisherData([(3, 0.04), (1, 0.02), (2, 0.01)], "b")
        pdf3 = PublisherData(
            [(1, 0.01), (2, 0.02), (1, 0.04), (1, 0.01), (3, 0.05)], "c"
        )
        res = IndependentOverlapDataSet(
            unlabeled_publisher_data_list=[pdf1, pdf2, pdf3],
            universe_size=5,
            random_state=np.random.RandomState(1),
        )
        expected_data_list = [
            [(0, 0.01), (0, 0.03), (1, 0.02), (2, 0.04)],
            [(0, 0.02), (3, 0.01), (4, 0.04)],
            [(0, 0.01), (0, 0.01), (0, 0.04), (1, 0.02), (4, 0.05)],
        ]
        expected_name_list = ["a", "b", "c"]
        self.assert_equal_pub_data_list(
            res._data, 3, expected_data_list, expected_name_list
        )


if __name__ == "__main__":
    absltest.main()
