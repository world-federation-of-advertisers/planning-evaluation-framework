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
from typing import Dict
from copy import deepcopy
from absl.testing import absltest
import numpy as np
from statsmodels.distributions.copula.elliptical import GaussianCopula
from statsmodels.distributions.copula.other_copulas import IndependenceCopula

from wfa_planning_evaluation_framework.data_generators.copula_data_set import (
    AnyFrequencyDistribution,
    CopulaDataSet,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)


class AnyFrequencyDistributionTest(absltest.TestCase):
    def test_ppf(self):
        hist = np.array([5, 3, 2])
        dist = AnyFrequencyDistribution(hist)
        self.assertEqual(dist.ppf(0.9), 2)
        self.assertEqual(dist.ppf(0.6), 1)
        self.assertEqual(dist.ppf(0.5), 0)
        self.assertEqual(dist.ppf(0.1), 0)


class CopulaDataSetTest(absltest.TestCase):
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
        np.testing.assert_equal(res[0], expected[0])
        np.testing.assert_equal(res[1], expected[1])

    def test_approximate_agreeement_with_marginals(self):
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

        def frequency_dictionary(pdf: PublisherData) -> Dict:
            freq_by_vid = pdf.user_counts_by_impressions(pdf.max_impressions)
            return dict(Counter(freq_by_vid.values()))

        res1, res2 = [frequency_dictionary(pdf) for pdf in dataset._data]
        try:
            self.assertAlmostEqual(res1[1] / 100, 1, delta=0.2)
            self.assertAlmostEqual(res1[2] / 100, 1, delta=0.2)
            self.assertAlmostEqual(res2[3] / 150, 1, delta=0.2)
        except:
            self.fail()

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
        try:
            self.assertAlmostEqual(res[(1, 1)] / 100, 1, delta=0.2)
            self.assertAlmostEqual(res[(2, 1)] / 100, 1, delta=0.2)
            self.assertAlmostEqual(res[(1, 2)] / 100, 1, delta=0.2)
            self.assertAlmostEqual(res[(2, 2)] / 100, 1, delta=0.2)
        except:
            self.fail()

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


if __name__ == "__main__":
    absltest.main()
