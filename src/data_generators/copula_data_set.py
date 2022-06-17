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
"""Generate multi-pub data by Copula from a list of single-pub data."""


import numpy as np
from typing import List, Iterable, Dict
from collections import Counter
from scipy import stats
from scipy import linalg as splinalg
from statsmodels.distributions.copula.copulas import (
    Copula,
    CopulaDistribution,
)
from statsmodels.distributions.copula.elliptical import GaussianCopula


from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.data_set import (
    DataSet,
)
from wfa_planning_evaluation_framework.data_generators.pricing_generator import (
    PricingGenerator,
)
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)


class CopulaDataSet(DataSet):
    """A DataSet constructed by gluing a list of PublisherData with Copula."""

    def __init__(
        self,
        unlabeled_publisher_data_list: Iterable[PublisherData],
        copula_generator: Copula = GaussianCopula(corr=0),
        universe_size: int = None,
        pricing_generator: PricingGenerator = FixedPriceGenerator(0.1),
        random_generator: np.random.Generator = np.random.default_rng(0),
        name: str = "copula",
    ):
        """Constructor for OverlapDataSet.

        Args:
            unlabeled_publisher_data_list:  A list of PublisherData. Each PublisherData
                indicates the frequency distribution at a publisher --- the
                ids labels are meaningless while their distribution matters.
            copula_generator:  An instance of a subclass of
                statsmodels.distributions.copula.copulas.Copula.  See here for the
                possible choices:
                https://github.com/statsmodels/statsmodels/tree/32dc52699f994acbf9dbdb9bd10d7eff04d860f5/statsmodels/distributions
            universe_size:  A cross-publisher universe size to construct the copula.
                If not given, then specify the universe size as the maximum single
                publisher reach times 2.
            pricing_generator:  A PricingGenerator object that annotates a list of
                id's with randomly generated price information.  At this moment,
                assume that all the publishers share the same PricingGenerator.
            random_generator:  A random generator to draw sample from the copula.
            name:  If specified, a human-readable name that will be associated to this
                DataSet.
        """
        if universe_size is None:
            universe_size = 2 * max(
                [1] + [data.max_reach for data in unlabeled_publisher_data_list]
            )
        self.marginal_distributions = []
        for pub in unlabeled_publisher_data_list:
            pmf = self.zero_included_pmf(pub, universe_size)
            self.marginal_distributions.append(
                stats.rv_discrete(values=(range(len(pmf)), pmf))
            )
        self.copula_generator = copula_generator
        self.distribution = CopulaDistribution(
            copula=self.copula_generator,
            marginals=self.marginal_distributions,
        )
        self.sample = self.distribution.rvs(
            nobs=universe_size,
            random_state=random_generator.integers(0, 1e9),
        ).astype("int32")
        imps = self.to_impressions(self.sample)
        for pub_imps in imps:
            random_generator.shuffle(pub_imps)
        super().__init__(
            publisher_data_list=[
                PublisherData(
                    impression_log_data=pricing_generator(pub_imps),
                    name=original_pub_data.name,
                )
                for pub_imps, original_pub_data in zip(
                    imps, unlabeled_publisher_data_list
                )
            ],
            name=name,
            universe_size=universe_size,
        )

    @property
    def frequency_vectors_sampled_distribution(self) -> Dict:
        """A dictionary of the distribution of the sampled frequency vectors.

        Returns:
            A dict d where for any frequency vector (f_1, ..., f_p),
            d[(f_1, ..., f_p)] is the count of this frequency vector in
            the DataSet.
        """
        return dict(Counter(tuple(obs) for obs in self.sample))

    @staticmethod
    def zero_included_pmf(
        publisher_data: PublisherData, universe_size: int
    ) -> np.ndarray:
        """Convert a PublisherData to a zero-included pmf vector."""
        hist = np.bincount(
            list(
                publisher_data.user_counts_by_impressions(
                    publisher_data.max_impressions
                ).values()
            )
        )
        hist[0] = universe_size - sum(hist)
        return hist / sum(hist)

    @staticmethod
    def to_impressions(frequency_vectors: List[np.ndarray]) -> List[List[int]]:
        """Convert a sample of frequency vectors to a list of impressions.

        An intermediate step to convert a sample of frequency vectors to a
        DataSet.

        Args:
            frequency_vectors:  A length <n> list of length <p> arrays, where
                n is the number of users in the sample, and p is the number of
                publishers.
                frequency_vectors[k] [i] indicates the frequency at pub i of
                the k-th user in the sample.

        Returns:
            A list of <p> lists of lengths n_1, ..., n_p,
            where p is the number of pubs and n_i is the number of impressions
            at pub i.
            Call this return as `cross_pub_impressions`.  Then,
            `cross_pub_impressions[i]` is a list of user ids with multiplicities
            that represents the sequence of purchasable impressions at publisher
            i as one increases the spend.
            The returned `cross_pub_impressions` is consistent with the input
            `frequency_vectors`.  Explicitly, any frequency vector
            [f_0, f_1, ..., f_{p-1}] appears k times in `frequency_vectors` if and
            only if there exist k distinct user ids that appear f_0 times in
            `cross_pub_impressions[0]`,  f_1 times in `cross_pub_impressions[1]`,
            ..., f_{p-1} times in `cross_pub_impressions[p - 1]`.
        """
        p = len(frequency_vectors[0])
        freq_vec_dist_dict = dict(
            Counter(tuple(obs.astype("int32")) for obs in frequency_vectors)
        )
        impressions = [[]] * p
        num_vids = 0
        for freq_vec, count in freq_vec_dist_dict.items():
            vids = list(range(num_vids, num_vids + count))
            for i in range(p):
                impressions[i] = impressions[i] + vids * freq_vec[i]
            num_vids += count
        return impressions


class CopulaCorrelationMatrixGenerator:
    @staticmethod
    def homogeneous(p: int, rho: float) -> np.ndarray:
        """Generate a homogeneous correlation matrix.

        Args:
            p:  Number of pubs.
            rho:  Homogenous correlation.

        Returns:
            A p * p matrix with diagonals being 1 and all off-diagonals being rho.
        """
        return splinalg.toeplitz([1] + [rho] * (p - 1))

    @staticmethod
    def autoregressive(p: int, rho: float) -> np.ndarray:
        """Generate a autoregressive correlation matrix.

        Args:
            p:  Number of pubs.
            rho:  Homogenous correlation.

        Returns:
            A <p * p> matrix C where C[i, j] = rho^|i - j| for any i, j.
        """
        return splinalg.toeplitz(np.power(np.array([rho] * p), np.arange(p)))

    @staticmethod
    def random(
        p: int,
        eta: float = 1,
        rng: np.random.Generator = np.random.default_rng(0),
    ) -> np.ndarray:
        """Randomly, uniformly draw a correlation matrix.

        A p * p correlation matrix is in the (p * p) dimensional Euclidean space.
        So, the geometric measure of (p * p) dimensional Euclidean space induces a
        measure on the manifold of p * p correlation matrices. This geoemetric
        measure defines a natural probability space.  This function uniformly draws
        a correlation matrix from this natural probability space.

        The uniform sampling is realized by the algorithm of:
            D. Lewandowski, D. Kurowickaa, H. Joe, "Generating random correlation
            matrices based on vines and extended onion method," Journal of
            Multivariate Analysis, Vol. 100, Iss. 9, October 2009, pp. 1989-2001.

        Args:
            p:  Number of pubs.
            eta:  A non-negative a tuning parameter.  Default eta = 1 gives a uniform
                distribution as described above.  (Other eta can give a non-uniform
                sampling of which the density is function of the determinant of the
                correlation matix, and eta.  In this way we can tune the weights of
                sampling stronger or weaker correlations.)
            rng:  A random number generator.

        Returns:
            A <p * p> random correlation matrix.
        """
        # The following is a line-to-line translation of the algorithm in
        # Section 3.2 of the paper.
        b = (p - 2) / 2 + eta
        u = rng.beta(b, b)
        r12 = 2 * u - 1
        r = np.array([[1, r12], [r12, 1]])
        for k in range(2, p):
            b -= 0.5
            y = rng.beta(a=k / 2, b=b)
            v = rng.normal(size=k)
            theta = v / np.linalg.norm(v)
            w = np.dot(np.sqrt(y), theta)
            q = np.dot(splinalg.sqrtm(r), w)
            next_r = np.zeros((k + 1, k + 1))
            next_r[:k, :k] = r
            next_r[k, k] = 1
            next_r[k, :k] = q
            next_r[:k, k] = q
            r = next_r
        return r
