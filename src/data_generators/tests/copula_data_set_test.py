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
"""Tests for copula_data_set.py."""

from collections import Counter
from re import M
from typing import Dict, Tuple
from copy import deepcopy
from itertools import product
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scipy import linalg as splinalg
from statsmodels.distributions.copula.elliptical import GaussianCopula, StudentTCopula
from statsmodels.distributions.copula.other_copulas import IndependenceCopula

from wfa_planning_evaluation_framework.data_generators.copula_data_set import (
    CopulaDataSet,
    CopulaCorrelationMatrixGenerator,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)


<<<<<<< HEAD
class AnyFrequencyDistributionTest(absltest.TestCase):
    def test_ppf(self):
        hist = np.array([5, 3, 2])
        dist = AnyFrequencyDistribution(hist)
        self.assertEqual(dist.ppf(0.9), 2)
        self.assertEqual(dist.ppf(0.6), 1)
        self.assertEqual(dist.ppf(0.5), 0)
        self.assertEqual(dist.ppf(0.1), 0)


class CopulaDataSetTest(parameterized.TestCase):
=======
class CopulaDataSetTest(absltest.TestCase):
>>>>>>> jiayu_copula_data_set
    def test_zero_included_pmf(self):
        pdf = PublisherData([(1, 0.01), (2, 0.02), (1, 0.04), (3, 0.05)])
        res = CopulaDataSet.zero_included_pmf(pdf, 10)
        expected = np.array([0.7, 0.2, 0.1])
        np.testing.assert_equal(res, expected)

    def test_to_impressions(self):
        frequnecy_vectors = [(1, 1), (1, 1), (2, 0), (2, 0), (3, 3)]
        frequnecy_vectors = [np.array(vec) for vec in frequnecy_vectors]
        res = CopulaDataSet.to_impressions(frequnecy_vectors)
        expected = [[0, 1, 2, 3, 2, 3, 4, 4, 4], [0, 1, 4, 4, 4]]
        np.testing.assert_equal(res, expected)

    @staticmethod
    def frequency_dictionary(pdf: PublisherData) -> Dict:
        freq_by_vid = pdf.user_counts_by_impressions(pdf.max_impressions)
        return dict(Counter(freq_by_vid.values()))

    def test_approximate_agreement_with_marginals(self):
        impressions1 = list(range(100)) * 1 + list(range(100, 200)) * 2
        pdf1 = PublisherData(FixedPriceGenerator(0.1)(impressions1))
        impressions2 = list(range(150)) * 3
        pdf2 = PublisherData(FixedPriceGenerator(0.1)(impressions2))
        dataset = CopulaDataSet(
            unlabeled_publisher_data_list=[pdf1, pdf2],
            copula_generator=IndependenceCopula(),
            universe_size=300,
            random_generator=np.random.default_rng(0),
        )

        res1, res2 = [self.frequency_dictionary(pdf) for pdf in dataset._data]
        self.assertTrue(1 in res1)
        self.assertTrue(2 in res1)
        self.assertTrue(3 in res2)
        self.assertAlmostEqual(res1[1] / 100, 1, delta=0.2)
        self.assertAlmostEqual(res1[2] / 100, 1, delta=0.2)
        self.assertAlmostEqual(res2[3] / 150, 1, delta=0.2)

    def test_independent_copula(self):
        # 200 ids with freq = 1 and 200 ids with freq = 2
        impressions = list(range(200)) * 1 + list(range(200, 400)) * 2
        pdf1 = PublisherData(FixedPriceGenerator(0.1)(impressions))
        pdf2 = deepcopy(pdf1)
        dataset = CopulaDataSet(
            unlabeled_publisher_data_list=[pdf1, pdf2],
            copula_generator=IndependenceCopula(),
            universe_size=400,
            random_generator=np.random.default_rng(0),
        )
        res = dataset.frequency_vectors_sampled_distribution
        # Because of the independence, the frequency vectors
        # (1, 1), (1, 2), (2, 1), (2, 2) should roughly appear
        # 50 times respectively.
        self.assertTrue((1, 1) in res)
        self.assertTrue((2, 1) in res)
        self.assertTrue((1, 2) in res)
        self.assertTrue((2, 2) in res)
        self.assertAlmostEqual(res[(1, 1)] / 100, 1, delta=0.2)
        self.assertAlmostEqual(res[(2, 1)] / 100, 1, delta=0.2)
        self.assertAlmostEqual(res[(1, 2)] / 100, 1, delta=0.2)
        self.assertAlmostEqual(res[(2, 2)] / 100, 1, delta=0.2)

    def test_fully_positively_correlated_copula(self):
        # 100 ids with freq = 1 and 100 ids with freq = 2
        impressions = list(range(100)) * 1 + list(range(100, 200)) * 2
        pdf1 = PublisherData(FixedPriceGenerator(0.1)(impressions))
        pdf2 = deepcopy(pdf1)
        dataset = CopulaDataSet(
            unlabeled_publisher_data_list=[pdf1, pdf2],
            # correlation = 1 is not allowed in GaussianCopula, so
            # choosing a correlation very close to 1 in the next line.
            copula_generator=GaussianCopula(1 - 1e-9),
            universe_size=200,
            random_generator=np.random.default_rng(0),
        )
        res = dataset.frequency_vectors_sampled_distribution
        # Because of the fully positive correlation, the frequency vectors
        # (1, 2) and (2, 1) are impossible.
        self.assertFalse((1, 2) in res)
        self.assertFalse((2, 1) in res)

    def test_fully_negatively_correlated_copula(self):
        impressions = list(range(100)) * 1 + list(range(100, 200)) * 2
        pdf1 = PublisherData(FixedPriceGenerator(0.1)(impressions))
        pdf2 = deepcopy(pdf1)
        dataset = CopulaDataSet(
            unlabeled_publisher_data_list=[pdf1, pdf2],
            copula_generator=GaussianCopula(-1 + 1e-9),
            universe_size=200,
            random_generator=np.random.default_rng(0),
        )
        res = dataset.frequency_vectors_sampled_distribution
        # Because of the fully negative correlation, the frequency vectors
        # (1, 1) and (2, 2) are impossible.
        self.assertFalse((1, 1) in res)
        self.assertFalse((2, 2) in res)

    # When using statsmodels.distributions.copula for more than 2 pubs, there is
    # "UserWarning: copulas for more than 2 dimension is untested."
    # So here, we added some tests with more than 2 pubs.
    # We did more tests in notebooks, and we believe the pacakge is correctly
    # generating at least the Gaussian and t- copulas even with >2 pubs.
    def test_approximate_agreement_with_marginals_with_more_than_two_pubs(self):
        impressions = list(range(100)) * 1 + list(range(100, 200)) * 2
        pdf = PublisherData(FixedPriceGenerator(0.1)(impressions))
        for num_pubs in [3, 5, 10]:
            # Correlation matrix with all correlations = 0.5
            cor_mat = splinalg.toeplitz([1] + [0.5] * (num_pubs - 1))
            for gen in [
                GaussianCopula(corr=cor_mat),
                StudentTCopula(corr=cor_mat, df=5),
            ]:
                dataset = CopulaDataSet(
                    unlabeled_publisher_data_list=[pdf] * num_pubs,
                    copula_generator=gen,
                    universe_size=300,
                    random_generator=np.random.default_rng(0),
                )
                for pdf in dataset._data:
                    res = self.frequency_dictionary(pdf)
                    self.assertTrue(1 in res)
                    self.assertTrue(2 in res)
                    # With more than 2 pubs, we conduct a large number of tests.
                    # Then, some test results significantly deviate from the expected
                    # values purely due to randomness.  As such, we set the tolerance
                    # delta to be larger (0.5 instead of the previous 0.2).
                    self.assertAlmostEqual(res[1] / 100, 1, delta=0.5)
                    self.assertAlmostEqual(res[2] / 100, 1, delta=0.5)

    def test_uncorrelated_copula_with_more_than_two_pubs(self):
        for num_pubs in [3, 4]:
            for gen in [
                GaussianCopula(np.identity(num_pubs)),
                StudentTCopula(np.identity(num_pubs), df=5),
            ]:
                # Suppose there are 100 * 2^p users in total, where p = #pubs.
                # At each pub, half users have frequency 1, and the other half
                # have frequency 2.
                half_size = int(100 * 2 ** num_pubs / 2)
                impressions = (
                    list(range(half_size)) * 1
                    + list(range(half_size, half_size * 2)) * 2
                )
                pdf = PublisherData(FixedPriceGenerator(0.1)(impressions))
                dataset = CopulaDataSet(
                    unlabeled_publisher_data_list=[pdf] * num_pubs,
                    copula_generator=gen,
                    universe_size=100 * 2 ** num_pubs,
                    random_generator=np.random.default_rng(0),
                )
                res = dataset.frequency_vectors_sampled_distribution
                for key in product(*tuple([(1, 2)] * num_pubs)):
                    self.assertTrue(key in res)
                    self.assertAlmostEqual(res[key] / 100, 1, delta=0.5)

    def test_fully_positively_correlated_copula_with_more_than_two_pubs(self):
        for num_pubs in [3, 4]:
            cor_mat = splinalg.toeplitz([1] + [1 - 1e-9] * (num_pubs - 1))
            for gen in [
                GaussianCopula(cor_mat),
                StudentTCopula(cor_mat, df=5),
            ]:
                impressions = list(range(100)) * 1 + list(range(100, 200)) * 2
                pdf = PublisherData(FixedPriceGenerator(0.1)(impressions))
                dataset = CopulaDataSet(
                    unlabeled_publisher_data_list=[pdf] * num_pubs,
                    copula_generator=gen,
                    universe_size=100 * 2 ** num_pubs,
                    random_generator=np.random.default_rng(0),
                )
                res = dataset.frequency_vectors_sampled_distribution
                key1 = tuple([1] * num_pubs)
                self.assertTrue(key1 in res)
                self.assertAlmostEqual(res[key1] / 100, 1, delta=0.2)
                key2 = tuple([2] * num_pubs)
                self.assertTrue(key2 in res)
                self.assertAlmostEqual(res[key2] / 100, 1, delta=0.2)

    def test_fully_negatively_correlated_copula_special_case(self):
        # Consider a specail case where pub 1 and pub 2 are fully positively
        # correlated and they are both fully negatively correlated with pub 3.
        cor_mat = np.array(
            [
                [1, 1 - 1e-9, -1 + 1e-9],
                [1 - 1e-9, 1, -1 + 1e-9],
                [-1 + 1e-9, -1 + 1e-9, 1],
            ]
        )
        for gen in [
            GaussianCopula(cor_mat),
            StudentTCopula(cor_mat, df=5),
        ]:
            impressions = list(range(100)) * 1 + list(range(100, 200)) * 2
            pdf = PublisherData(FixedPriceGenerator(0.1)(impressions))
            dataset = CopulaDataSet(
                unlabeled_publisher_data_list=[pdf] * 3,
                copula_generator=gen,
                universe_size=200,
                random_generator=np.random.default_rng(0),
            )
            res = dataset.frequency_vectors_sampled_distribution
            self.assertTrue((1, 1, 2) in res)
            self.assertAlmostEqual(res[(1, 1, 2)] / 100, 1, delta=0.2)
            self.assertTrue((2, 2, 1) in res)
            self.assertAlmostEqual(res[(2, 2, 1)] / 100, 1, delta=0.2)


class CopulaCorrelationMatrixGeneratorTest(absltest.TestCase):
    def test_homogenous(self):
        res = CopulaCorrelationMatrixGenerator.homogeneous(p=3, rho=0.1)
        expected = np.array([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])
        np.testing.assert_almost_equal(res, expected, decimal=3)

    def test_autoregressive(self):
        res = CopulaCorrelationMatrixGenerator.autoregressive(p=3, rho=0.5)
        expected = np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
        np.testing.assert_almost_equal(res, expected, decimal=3)

    def test_random_positive_definitiveness(self):
        for seed in [1, 2, 3, 4]:
            res = CopulaCorrelationMatrixGenerator.random(
                p=5, rng=np.random.default_rng(seed)
            )
            # For a randomly generated correlation matrix, we test if it's positive definite
            self.assertTrue(all(np.linalg.eig(res)[0] > 0))

    def test_random_uniformity_with_two_pubs(self):
        rng = np.random.default_rng(0)
        cor_mats = [
            CopulaCorrelationMatrixGenerator.random(p=2, rng=rng) for _ in range(400)
        ]
        cors = [cor_mat[0, 1] for cor_mat in cor_mats]
        correlation_frequencies = np.histogram(a=cors, bins=4, range=(-1, 1))[0] / 400
        np.testing.assert_almost_equal(correlation_frequencies, [1 / 4] * 4, decimal=1)

    def test_random_uniformity_with_three_pubs(self):
        rng = np.random.default_rng(0)
        cor_mats = [
            CopulaCorrelationMatrixGenerator.random(p=3, rng=rng) for _ in range(1000)
        ]

        def to_polar_coordinates(
            x: float, y: float, z: float
        ) -> Tuple[float, float, float]:
            """Find the polar coordinates of any 3D point.

            Find (r, theta, phi) such that
            - x = r cos(phi) sin(theta)
            - y = r sin(phi) sin(theta)
            - z = r cos(theta).
            """
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            phi = np.arctan2(y, x)
            theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
            return (r, phi, theta)

        correlation_tuples = [
            (cor_mat[0, 1], cor_mat[0, 2], cor_mat[1, 2]) for cor_mat in cor_mats
        ]
        polar_tuples = [
            to_polar_coordinates(*correlation_tuple)
            for correlation_tuple in correlation_tuples
        ]

        # 1. Test the uniformity of the radius.  Because of the positive definitiveness
        # constraint, not all values of the radius are possible.  It can be proved that
        # When (x, y, z) has a radius no greater than 0.5, the correlation matrix of the
        # correlations (x, y, z) is always positive definite.
        # We then expect that the radius_cube = radius ** 3 of the (x, y, z) of
        # the generated matrix is uniform at least in (0, 0.5).  This is mathematically
        # proved to be a necessary condition for the uniformity of the whole correlation
        # matrix.
        radius_cube_frequencies = np.array(
            np.histogram(
                a=[polar_tuple[0] ** 3 for polar_tuple in polar_tuples],
                bins=4,
                range=(0, 0.5),
            )[0],
            dtype=float,
        )
        radius_cube_frequencies /= sum(radius_cube_frequencies)
        np.testing.assert_almost_equal(radius_cube_frequencies, [1 / 4] * 4, decimal=1)

        # 2. Test the uniformity of the phi angle.  Another necessary condition for the
        # uniformity of the whole correlation matrix is that phi is uniform in [-pi, pi].
        phi_frequencies = (
            np.histogram(
                a=[polar_tuple[1] for polar_tuple in polar_tuples],
                bins=4,
                range=(-np.pi, np.pi),
            )[0]
            / 1000
        )
        np.testing.assert_almost_equal(phi_frequencies, [1 / 4] * 4, decimal=1)

        # 3. Test the uniformity of the theta angle.  Another necessary condition for the
        # uniformity of the whole correlation matrix is that cos(theta) is uniform in
        # [0, 1] in any unit ball contained in the region of which any (x, y, z) form
        # a valid, i.e., positive definite correlation matrix.  This condition is not
        # obvious but provable.
        theta_frequencies = np.array(
            np.histogram(
                a=[
                    np.cos(polar_tuple[2])
                    for polar_tuple in polar_tuples
                    if polar_tuple[0] ** 3 < 0.5
                ],
                bins=4,
                range=(-1, 1),
            )[0],
            dtype=float,
        )
        theta_frequencies /= sum(theta_frequencies)
        np.testing.assert_almost_equal(theta_frequencies, [1 / 4] * 4, decimal=1)


if __name__ == "__main__":
    absltest.main()
