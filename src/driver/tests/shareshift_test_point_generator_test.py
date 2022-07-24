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
"""Tests for shareshift_test_point_generator.py."""

from absl.testing import absltest
import numpy as np
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.driver.shareshift_test_point_generator import (
    ShareShiftTestPointGenerator,
)


class ShareShiftTestPointGeneratorTest(absltest.TestCase):
    def test_two_publishers(self):
        pdf1 = PublisherData([(1, 10.0)], "pdf1")
        pdf2 = PublisherData([(1, 20.0)], "pdf2")
        dataset = DataSet([pdf1, pdf2], "test")
        generator = ShareShiftTestPointGenerator(
            dataset=dataset,
            campaign_spend_fractions=np.array([0.6, 0.2]),
            shift_fraction_choices=[-0.5, 0.5],
        )
        points = [x for x in generator.test_points()]
        self.assertLen(points, 4)
        np.testing.assert_almost_equal(points[0], [9, 1], decimal=3)
        np.testing.assert_almost_equal(points[1], [3, 7], decimal=3)
        np.testing.assert_almost_equal(points[2], [4, 6], decimal=3)
        np.testing.assert_almost_equal(points[3], [8, 2], decimal=3)


if __name__ == "__main__":
    absltest.main()
