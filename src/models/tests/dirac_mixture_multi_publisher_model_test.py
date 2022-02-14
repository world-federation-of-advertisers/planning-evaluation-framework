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
"""Tests for dirac_mixture_multi_publisher_model.py."""

from absl.testing import absltest
import numpy as np
from typing import List, Tuple
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.models.dirac_mixture_single_publisher_model import (
    DiracMixtureSinglePublisherModel,
)
from wfa_planning_evaluation_framework.models.dirac_mixture_multi_publisher_model import (
    MultivariateMixedPoissonOptimizer,
    DiracMixtureMultiPublisherModel,
)


class FakeRandomGenerator:
    def choice(self, a: List, p: np.ndarray, size: int) -> np.ndarray:
        return np.array([len(a) - 1] * size)

    def random(self, size: Tuple) -> np.ndarray:
        return np.zeros(shape=size)


class MultivariateMixedPoissonOptimizerTest(absltest.TestCase):

    cls = MultivariateMixedPoissonOptimizer

    def test_normalize_rows(self):
        matrix = np.array([[1, 2, 1], [2, 3, 5]])
        res = self.cls.normalize_rows(matrix)
        expected = np.array([[0.25, 0.5, 0.25], [0.2, 0.3, 0.5]])
        np.testing.assert_almost_equal(res, expected, decimal=2)
        # For 2d arrays, absltest.assertSequenceAlmostEqual no longer works.

    def test_check_dimension_campatibility(self):
        try:
            MultivariateMixedPoissonOptimizer(
                observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
                frequency_histograms_on_observable_directions=np.array(
                    [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7]]
                ),
                prior_marginal_frequency_histograms=np.array(
                    [[0.75, 0.25], [0.65, 0.35]]
                ),
            )
            # Note that check_dimension_campatibility() is already
            # executed in __init__().
        except:
            self.fail("Rejected compatible dimensions")
        # number of pubs don't match
        with self.assertRaises(ValueError):
            MultivariateMixedPoissonOptimizer(
                observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
                frequency_histograms_on_observable_directions=np.array(
                    [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7]]
                ),
                prior_marginal_frequency_histograms=np.array(
                    [[0.75, 0.25], [0.65, 0.35], [0.5, 0.5]]
                ),
            )
        # number of observable directions don't match
        with self.assertRaises(ValueError):
            MultivariateMixedPoissonOptimizer(
                observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
                frequency_histograms_on_observable_directions=np.array(
                    [[0.7, 0.3], [0.6, 0.4]]
                ),
                prior_marginal_frequency_histograms=np.array(
                    [[0.75, 0.25], [0.65, 0.35]]
                ),
            )

    def test_weighted_random_sampling(self):
        res = self.cls.weighted_random_sampling(
            ncomponents=1,
            marginal_pmfs=np.array([[0.5, 0.5], [0.7, 0.3]]),
            rng=FakeRandomGenerator(),
        )
        expected = np.array([[0, 0], [1, 1]])
        np.testing.assert_almost_equal(res, expected)

    def test_obtain_pmf_matrix(self):
        res = self.cls.obtain_pmf_matrix(
            component_vector=np.array([0, 0]),
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            max_freq=1,
        )
        expected = np.array([[1, 0], [1, 0], [1, 0]])
        np.testing.assert_almost_equal(
            res, expected, decimal=2, err_msg="Unexpected for zero component"
        )
        res = self.cls.obtain_pmf_matrix(
            component_vector=np.array([1, 2]),
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            max_freq=1,
        )
        expected = np.array([[0.368, 0.632], [0.135, 0.865], [0.050, 0.950]])
        np.testing.assert_almost_equal(
            res, expected, decimal=2, err_msg="Unexpected for non-zero component"
        )

    def test_fit(self):
        # An obvious solution under zero reach
        optimizer = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            frequency_histograms_on_observable_directions=np.array(
                [[1, 0], [1, 0], [1, 0]]
            ),
            prior_marginal_frequency_histograms=np.array([[0.5, 0.5], [0.7, 0.3]]),
            ncomponents=1,
            rng=FakeRandomGenerator(),
        )
        optimizer.fit()
        expected_ws = np.array([1, 0])
        np.testing.assert_almost_equal(
            optimizer.ws, expected_ws, decimal=2, err_msg="Unexpected for zero reach"
        )
        # A more complicated case of which the solution has been manually computed
        optimizer = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            frequency_histograms_on_observable_directions=np.array(
                [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7]]
            ),
            prior_marginal_frequency_histograms=np.array([[0.5, 0.5], [0.7, 0.3]]),
            ncomponents=1,
            rng=FakeRandomGenerator(),
        )
        optimizer.fit()
        expected_components = np.array([[0, 0], [1, 1]])
        expected_ws = np.array([0.322, 0.678])
        np.testing.assert_almost_equal(
            optimizer.components,
            expected_components,
            decimal=2,
            err_msg="Unexpected components for non-zero reach",
        )
        np.testing.assert_almost_equal(
            optimizer.ws,
            expected_ws,
            decimal=2,
            err_msg="Unexpected weights for non-zero reach",
        )

    def test_predict(self):
        optimizer = MultivariateMixedPoissonOptimizer(
            observable_directions=np.array([[1, 0], [0, 1], [1, 1]]),
            frequency_histograms_on_observable_directions=np.array(
                [[0.7, 0.3, 0], [0.6, 0.4, 0], [0.3, 0.7, 0]]
            ),
            prior_marginal_frequency_histograms=np.array([[0.5, 0.5], [0.7, 0.3]]),
            ncomponents=1,
            rng=FakeRandomGenerator(),
        )

        # Set arbitrary parameters
        optimizer.fitted = True
        optimizer.components = np.array([[0, 0], [1, 0], [1, 1]])
        optimizer.ws = np.array([0.3, 0.2, 0.5])

        # Zero impression
        res = optimizer.predict(hypothetical_direction=np.array([0, 0]))
        expected = np.array([1, 0, 0])
        np.testing.assert_almost_equal(
            res,
            expected,
            decimal=2,
            err_msg="Unexpected for zero impression, no customized_max_freq",
        )
        res = optimizer.predict(
            hypothetical_direction=np.array([0, 0]), customized_max_freq=1
        )
        expected = np.array([1, 0])
        np.testing.assert_almost_equal(
            res,
            expected,
            decimal=2,
            err_msg="Unexpected for zero impression, smaller customized_max_freq",
        )

        res = optimizer.predict(
            hypothetical_direction=np.array([0, 0]), customized_max_freq=3
        )
        expected = np.array([1, 0, 0, 0])
        np.testing.assert_almost_equal(
            res,
            expected,
            decimal=2,
            err_msg="Unexpected for zero impression, larger customized_max_freq",
        )

        # Moderate number of impressions
        res = optimizer.predict(hypothetical_direction=np.array([0.5, 0.5]))
        expected = np.array([0.605, 0.245, 0.150])
        np.testing.assert_almost_equal(
            res, expected, decimal=2, err_msg="Unexpected for moderate impressions"
        )

        # Infinite impressions
        res = optimizer.predict(hypothetical_direction=np.array([10000, 10000]))
        expected = np.array([0.3, 0, 0.7])
        np.testing.assert_almost_equal(
            res, expected, decimal=2, err_msg="Unexpected for infinite impressions"
        )


class DiracMixtureMultiPublisherModelTest(absltest.TestCase):
    cls = DiracMixtureMultiPublisherModel

    def test_ensure_compatible_num_publishers(self):
        imps = [3, 4, 5]
        reaches = [1, 2, 3]
        reach_curves = [
            DiracMixtureSinglePublisherModel(
                data=[ReachPoint(impressions=[imp], kplus_reaches=[reach])]
            )
            for imp, reach in zip(imps, reaches)
        ]
        reach_points = [ReachPoint(impressions=imps, kplus_reaches=[4])]
        try:
            self.cls.ensure_compatible_num_publishers(reach_curves, reach_points)
        except:
            self.fail("Rejected compatible inputs")
        reach_points += [ReachPoint(impressions=[1, 2, 3, 4], kplus_reaches=[4])]
        with self.assertRaises(ValueError):
            self.cls.ensure_compatible_num_publishers(reach_curves, reach_points)

    def test_select_single_publisher_points(self):
        rp1 = ReachPoint(impressions=[3, 0, 0], kplus_reaches=[1])
        rp2 = ReachPoint(impressions=[5, 0, 0], kplus_reaches=[2])
        rp3 = ReachPoint(impressions=[2, 4, 0], kplus_reaches=[3])
        rp4 = ReachPoint(impressions=[0, 0, 7], kplus_reaches=[4])
        res = self.cls.select_single_publisher_points([rp1, rp2, rp3, rp4])
        expected = [0, 2]
        self.assertListEqual(list(res.keys()), expected)
        self.assertLen(res[0], 2)
        self.assertLen(res[2], 1)
        self.assertEqual(res[2][0].reach(1), 4)

    def test_obtain_marginal_frequency_histograms(self):
        rp1 = ReachPoint(impressions=[3, 0, 0], kplus_reaches=[1])
        rp2 = ReachPoint(impressions=[5, 0, 0], kplus_reaches=[2])
        rp3 = ReachPoint(impressions=[2, 4, 0], kplus_reaches=[3])
        rp4 = ReachPoint(impressions=[0, 0, 7], kplus_reaches=[4])
        with self.assertRaises(AssertionError):
            self.cls.obtain_marginal_frequency_histograms(
                [rp1, rp2, rp3, rp4], universe_size=15
            )
        rp5 = ReachPoint(impressions=[0, 8, 0], kplus_reaches=[5])
        res_hists, res_impressions = self.cls.obtain_marginal_frequency_histograms(
            [rp1, rp2, rp3, rp4, rp5], universe_size=15
        )
        expected_hists = np.array([[14, 1], [10, 5], [11, 4]])
        expected_impressions = np.array([3, 8, 7])
        np.testing.assert_equal(res_hists, expected_hists)
        np.testing.assert_equal(res_impressions, expected_impressions)

    def test_obtain_observable_directions(self):
        rp1 = ReachPoint(impressions=[3, 0, 0], kplus_reaches=[1])
        rp2 = ReachPoint(impressions=[5, 0, 0], kplus_reaches=[2])
        rp3 = ReachPoint(impressions=[2, 4, 0], kplus_reaches=[3])
        rp4 = ReachPoint(impressions=[0, 0, 7], kplus_reaches=[4])
        rp5 = ReachPoint(impressions=[0, 8, 0], kplus_reaches=[5])
        rp6 = ReachPoint(impressions=[4, 6, 10], kplus_reaches=[10])
        res = self.cls.obtain_observable_directions(
            reach_points=[rp1, rp2, rp3, rp4, rp5, rp6],
            impressions_at_prior=np.array([3, 8, 7]),
        )
        expected = np.array(
            [
                [1, 0, 0],
                [5 / 3, 0, 0],
                [2 / 3, 4 / 8, 0],
                [0, 0, 1],
                [0, 1, 0],
                [4 / 3, 6 / 8, 10 / 7],
            ]
        )
        np.testing.assert_almost_equal(res, expected, decimal=2)

    def test_obtain_frequency_histograms_on_observable_directions(self):
        rp1 = ReachPoint(impressions=[5, 6], kplus_reaches=[7, 2])
        rp2 = ReachPoint(impressions=[7, 3], kplus_reaches=[5, 4])
        res = self.cls.obtain_frequency_histograms_on_observable_directions(
            reach_points=[rp1, rp2], universe_size=15
        )
        expected = np.array([[8, 5, 2], [10, 1, 4]])
        np.testing.assert_equal(res, expected)

    def test_fit(self):
        rc1 = DiracMixtureSinglePublisherModel(
            data=[ReachPoint(impressions=[4], kplus_reaches=[2])]
        )
        rc2 = DiracMixtureSinglePublisherModel(
            data=[ReachPoint(impressions=[6], kplus_reaches=[4])]
        )
        rp1 = ReachPoint(impressions=[4, 0], kplus_reaches=[2])
        rp2 = ReachPoint(impressions=[0, 6], kplus_reaches=[4])
        rp3 = ReachPoint(impressions=[4, 6], kplus_reaches=[5])
        model = self.cls(
            reach_curves=[rc1, rc2],
            reach_points=[rp1, rp2, rp3],
            universe_size=10,
            ncomponents=1,
            # which means 1 component plus the zero component, a total of 2
            rng=FakeRandomGenerator(),
        )
        model._fit()
        expected_components = np.array([[0, 0], [1, 1]])
        expected_ws = np.array([0.478, 0.522])
        np.testing.assert_equal(model.optimizer.components, expected_components)
        np.testing.assert_almost_equal(model.optimizer.ws, expected_ws, decimal=2)

        # Zero reach
        rp1 = ReachPoint(impressions=[4, 0], kplus_reaches=[0])
        rp2 = ReachPoint(impressions=[0, 6], kplus_reaches=[0])
        rp3 = ReachPoint(impressions=[4, 6], kplus_reaches=[0])
        model = self.cls(
            reach_curves=[rc1, rc2],
            reach_points=[rp1, rp2, rp3],
            universe_size=10,
            ncomponents=1,
            # which means 1 component plus the zero component, a total of 2
            rng=FakeRandomGenerator(),
        )
        model._fit()
        expected_ws = np.array([1, 0])
        np.testing.assert_almost_equal(model.optimizer.ws, expected_ws, decimal=2)

    def test_by_impressions_no_single_pub_reach_agreement(self):
        rp1 = ReachPoint(impressions=[4, 0], kplus_reaches=[0])
        rp2 = ReachPoint(impressions=[0, 6], kplus_reaches=[0])
        rp3 = ReachPoint(impressions=[4, 6], kplus_reaches=[0])
        # The above parameters don't matter since we'll manually set its optimizer
        # parameters below.
        model = self.cls(
            reach_curves=None,
            reach_points=[rp1, rp2, rp3],
            universe_size=10,
            ncomponents=1,
            rng=FakeRandomGenerator(),
        )
        model._fit()
        model.optimizer.components = np.array([[0, 0], [0.5, 0.3]])
        model.optimizer.ws = np.array([1, 0])
        res = model.by_impressions_no_single_pub_reach_agreement(
            impressions=[3, 5], max_frequency=2
        )
        self.assertEqual(res.reach(1), 0)
        self.assertEqual(res.reach(2), 0)
        model.optimizer.ws = np.array([0.4, 0.6])
        res = model.by_impressions_no_single_pub_reach_agreement(
            impressions=[3, 5], max_frequency=2
        )
        self.assertEqual(res.reach(1), 3)
        self.assertEqual(res.reach(2), 1)
        # confirmed by manual computation
        model.optimizer.ws = np.array([0.4, 0.6])
        res = model.by_impressions_no_single_pub_reach_agreement(
            impressions=[1000, 1000], max_frequency=2
        )
        self.assertEqual(res.reach(1), 6)
        self.assertEqual(res.reach(2), 6)

    def test_backsolve_impression(self):
        curve = lambda x: x ** 2
        target_reach = 26
        expected = 5
        starting_impression = 9
        res = self.cls.backsolve_impression(curve, target_reach, starting_impression)
        self.assertEqual(res, expected)
        starting_impression = 2
        res = self.cls.backsolve_impression(curve, target_reach, starting_impression)
        self.assertEqual(res, expected)

    def test_by_impressions_with_single_pub_reach_agreement(self):
        rc1 = DiracMixtureSinglePublisherModel(
            data=[ReachPoint(impressions=[20], kplus_reaches=[11])]
        )
        rc2 = DiracMixtureSinglePublisherModel(
            data=[ReachPoint(impressions=[20], kplus_reaches=[7])]
        )
        rp1 = ReachPoint(impressions=[20, 0], kplus_reaches=[6])
        rp2 = ReachPoint(impressions=[0, 20], kplus_reaches=[12])
        rp3 = ReachPoint(impressions=[20, 20], kplus_reaches=[13])
        # The above parameters don't matter since we'll manually set its optimizer
        # parameters below.
        model = self.cls(
            reach_curves=[rc1, rc2],
            reach_points=[rp1, rp2, rp3],
            universe_size=30,
            ncomponents=5,
            rng=FakeRandomGenerator(),
        )
        model._fit()
        res = model.by_impressions_with_single_pub_reach_agreement(
            impressions=[20, 0], max_frequency=2
        )
        self.assertEqual(res.reach(1), 11)
        # As a comparison, look at the result when not forcing single
        # pub agreement
        res = model.by_impressions_no_single_pub_reach_agreement(
            impressions=[20, 0], max_frequency=2
        )
        self.assertEqual(res.reach(1), 9)
        res = model.by_impressions_with_single_pub_reach_agreement(
            impressions=[0, 20], max_frequency=2
        )
        self.assertEqual(res.reach(1), 7)


if __name__ == "__main__":
    absltest.main()
