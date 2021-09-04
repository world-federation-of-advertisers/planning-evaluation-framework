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
"""Tests for test_point_aggregator.py."""

from absl.testing import absltest
import numpy as np
import pandas as pd
from wfa_planning_evaluation_framework.driver.test_point_aggregator import (
    AGGREGATORS,
    _reach,
    _shuffle_distance,
    aggregate,
)
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint


class TestPointAggregatorTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rp110 = ReachPoint([1000], [0.0, 0.0, 0.0], [10.0])
        cls.rp111 = ReachPoint([1000], [200.0, 100.0, 50.0], [10.0])
        cls.rp112 = ReachPoint([1000], [210.0, 110.0, 60.0], [10.0])

        cls.rp121 = ReachPoint([2000], [300.0, 150.0, 75.0], [30.0])
        cls.rp122 = ReachPoint([2000], [320.0, 170.0, 95.0], [30.0])

        cls.rp131 = ReachPoint([3000], [400.0, 200.0, 100.0], [40.0])
        cls.rp132 = ReachPoint([3000], [430.0, 230.0, 130.0], [40.0])
        cls.test_points1 = [cls.rp111, cls.rp121, cls.rp131]
        cls.model_points1 = [cls.rp112, cls.rp122, cls.rp132]

    def test_npoints(self):
        self.assertEqual(
            AGGREGATORS["npoints"](self.test_points1, self.model_points1), 3
        )

    def test_mean_error(self):
        self.assertEqual(
            AGGREGATORS["mean_error"](self.test_points1, self.model_points1), -20
        )
        self.assertEqual(
            AGGREGATORS["mean_error"](self.model_points1, self.test_points1), 20
        )

    def test_mean_abs_error(self):
        self.assertEqual(
            AGGREGATORS["mean_abs_error"](self.test_points1, self.model_points1), 20
        )
        self.assertEqual(
            AGGREGATORS["mean_abs_error"](self.model_points1, self.test_points1), 20
        )

    def test_mean_squared_error(self):
        self.assertEqual(
            AGGREGATORS["mean_squared_error"](self.test_points1, self.model_points1),
            1400 / 3,
        )
        self.assertEqual(
            AGGREGATORS["mean_squared_error"](self.model_points1, self.test_points1),
            1400 / 3,
        )

    def test_mean_abs_relative_error(self):
        self.assertEqual(
            AGGREGATORS["mean_abs_relative_error"](
                self.test_points1, self.model_points1
            ),
            1.0 / 3 * (10.0 / 200.0 + 20.0 / 300.0 + 30.0 / 400.0),
        )

    def test_mean_abs_relative_error_at_higher_frequencies(self):
        self.assertEqual(
            AGGREGATORS["mare_freq_at_least_2"](self.test_points1, self.model_points1),
            1.0 / 3 * (10.0 / 100.0 + 20.0 / 150.0 + 30.0 / 200.0),
        )
        self.assertEqual(
            AGGREGATORS["mare_freq_at_least_3"](self.test_points1, self.model_points1),
            1.0 / 3 * (10.0 / 50.0 + 20.0 / 75.0 + 30.0 / 100.0),
        )
        self.assertTrue(
            np.isnan(
                AGGREGATORS["mare_freq_at_least_4"](
                    self.test_points1, self.model_points1
                )
            )
        )

    def test_mean_squared_relative_error(self):
        self.assertEqual(
            AGGREGATORS["mean_squared_relative_error"](
                self.test_points1, self.model_points1
            ),
            1.0
            / 3
            * (
                10.0 ** 2 / 200.0 ** 2 + 20.0 ** 2 / 300.0 ** 2 + 30.0 ** 2 / 400.0 ** 2
            ),
        )

    def test_var_error(self):
        self.assertEqual(
            AGGREGATORS["var_error"](self.test_points1, self.model_points1), 200.0 / 3
        )

    def test_var_relative_error(self):
        self.assertAlmostEqual(
            AGGREGATORS["var_relative_error"](self.test_points1, self.model_points1),
            0.00010802,
        )

    def test_relative_error_quantiles(self):
        xlist = []
        ylist = []
        for i in range(11):
            xlist.append(ReachPoint([i], [1]))
            ylist.append(ReachPoint([i], [i + 1]))
        self.assertEqual(AGGREGATORS["relative_error_q10"](xlist, ylist), 1.0)
        self.assertEqual(AGGREGATORS["relative_error_q20"](xlist, ylist), 2.0)
        self.assertEqual(AGGREGATORS["relative_error_q30"](xlist, ylist), 3.0)
        self.assertEqual(AGGREGATORS["relative_error_q40"](xlist, ylist), 4.0)
        self.assertEqual(AGGREGATORS["relative_error_q50"](xlist, ylist), 5.0)
        self.assertEqual(AGGREGATORS["relative_error_q60"](xlist, ylist), 6.0)
        self.assertEqual(AGGREGATORS["relative_error_q70"](xlist, ylist), 7.0)
        self.assertEqual(AGGREGATORS["relative_error_q80"](xlist, ylist), 8.0)
        self.assertEqual(AGGREGATORS["relative_error_q90"](xlist, ylist), 9.0)

    def test_mean_shuffle_distance(self):
        self.assertEqual(
            AGGREGATORS["mean_shuffle_distance"](self.test_points1, self.model_points1),
            1.0,
        )
        xlist = [ReachPoint([1], [6, 5, 4, 3, 2, 1], [1])]
        self.assertEqual(AGGREGATORS["mean_shuffle_distance"](xlist, xlist), 0.0)
        ylist = [ReachPoint([1], [7, 6, 6, 6, 6, 6], [1])]
        self.assertAlmostEqual(AGGREGATORS["mean_shuffle_distance"](xlist, ylist), 0.8)

    def test_mean_squared_shuffle_distance(self):
        self.assertEqual(
            AGGREGATORS["mean_squared_shuffle_distance"](
                self.test_points1, self.model_points1
            ),
            1.0,
        )
        xlist = [ReachPoint([1], [6, 5, 4, 3, 2, 1], [1])]
        self.assertEqual(
            AGGREGATORS["mean_squared_shuffle_distance"](xlist, xlist), 0.0
        )
        ylist = [ReachPoint([1], [7, 6, 6, 6, 6, 6], [1])]
        self.assertAlmostEqual(
            AGGREGATORS["mean_squared_shuffle_distance"](xlist, ylist), 0.64
        )

    def test_var_shuffle_distance(self):
        xlist = [ReachPoint([1], [6, 5, 4, 3, 2, 1], [1])] * 2
        ylist = [
            ReachPoint([1], [7, 6, 6, 6, 6, 6], [1]),
            ReachPoint([1], [6, 5, 4, 3, 2, 1], [1]),
        ]
        self.assertAlmostEqual(
            AGGREGATORS["var_shuffle_distance"](xlist, ylist), 0.16, places=3
        )

    def test__reach(self):
        np.testing.assert_array_equal(_reach([]), np.array([]))
        np.testing.assert_array_equal(_reach([self.rp111]), np.array([200.0]))
        np.testing.assert_array_equal(
            _reach([self.rp111, self.rp112]), np.array([200.0, 210.0])
        )
        np.testing.assert_array_equal(
            _reach([self.rp111, self.rp112], 2), np.array([100.0, 110.0])
        )

    def test__shuffle_distance(self):
        self.assertEqual(
            _shuffle_distance(ReachPoint([1], [1]), ReachPoint([1], [2])), 1.0
        )
        self.assertEqual(
            _shuffle_distance(
                ReachPoint([1], [6, 5, 4, 3, 2, 1]), ReachPoint([1], [5, 5, 5, 5, 5, 5])
            ),
            0.5,
        )
        self.assertEqual(
            _shuffle_distance(
                ReachPoint([1], [5, 4, 3, 2, 2, 2]), ReachPoint([1], [5, 5, 4, 3, 2, 2])
            ),
            1.0 / 3.0,
        )

    def test_aggregate(self):
        pd = aggregate(self.test_points1, self.model_points1)
        self.assertEqual(pd["npoints"][0], 3)
        self.assertEqual(len(pd.columns), len(AGGREGATORS) + 2)


if __name__ == "__main__":
    absltest.main()
