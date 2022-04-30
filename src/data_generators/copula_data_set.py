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


class AnyFrequencyDistribution:

    """Instantiate any frequency distribution with the ppf method.

    We will use `CopulaDistribution` later to construct a copula.
    `CopulaDistribution` was designed for combining the distribution instances
    in scipy.stats.  After examing its source codes, we found that any list
    of instances with a `ppf` method can be a valid input of `CopulaDistribution`.
    In order to combine any single publisher distributions (i.e., any
    ImpressionGenerators), this AnyFrequencyDistribution class instantiate
    any frequency histogram with a ppf method.
    """

    def __init__(self, histogram: np.ndarray):
        """Constructs AnyFrequencyDistribution.

        Args:
            histogram: An array h of counts or probabilities. h[f] is the count
                or probability of frequency f, for f = 0, ..., a maximum
                frequency.
        """
        self.pmf = histogram / sum(histogram)
        self.cdf = np.cumsum(self.pmf)

    def ppf(self, p: float) -> int:
        """Calculates ppf of the count distribution defined by the histogram.

        ppf is the inverse of cdf as will be explained in "Returns".

        Args:
            p:  A probability.

        Returns:
            A count x for which cdf[x] <= p and cdf[x + 1] > p, where cdf is
            the cumulative sum of self.hist.
        """
        return np.searchsorted(a=self.cdf, v=p)


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
            pricing_generator:  A PricingGenerator object that annotates a list of
                id's with randomly generated price information.  At this moment,
                assume that all the publishers share the same PricingGenerator.
            random_generator:  A random generator to draw sample from the copula.
            name:  If specified, a human-readable name that will be associated to this
                DataSet.
        """
        self.marginal_distributions = [
            AnyFrequencyDistribution(self.zero_included_pmf(pub, universe_size))
            for pub in unlabeled_publisher_data_list
        ]
        self.copula_generator = copula_generator
        self.distribution = CopulaDistribution(
            copula=self.copula_generator,
            marginals=self.marginal_distributions,
        )
        self.sample = self.distribution.rvs(
            nobs=universe_size,
            random_state=random_generator.integers(0, 1e9),
        ).astype("int32")
        super().__init__(
            publisher_data_list=[
                PublisherData(
                    impression_log_data=pricing_generator(pub_imps),
                    name=original_pub_data.name,
                )
                for pub_imps, original_pub_data in zip(
                    self.to_impressions(self.sample), unlabeled_publisher_data_list
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
                frequency_vectors[k] [i] is the frequency at pub i of the k-th
                user in the sample.

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

    @staticmethod
    def random_correlation_matrix(p: int, eta: float = 0) -> np.ndarray:
        """Randomly draw a correlation matrix using the "Onion method".



        The onion method is referred to the "onion method" in [Ref.1] and the
        "extended onion method" in [Ref.2] in the following:
        [Ref.1] S. Ghosh, S. Henderson, "Behavior of the NORTA Method for
            Correlated Random Vector Generation as the Dimension Increases,"
            ACM Transactions on Modeling and Computer Simulation, Vol. 13,
            Iss. 3, July 2003, pp. 276–294
        [Ref.2] D. Lewandowski, D. Kurowickaa, H. Joe, "Generating random
            correlation matrices based on vines and extended onion method,"
            Journal of Multivariate Analysis, Vol. 100, Iss. 9, October 2009,
            pp. 1989-2001 (see Section 3.2)
        Compared to the algorithm in [Ref.1], the algorithm in [Ref.2] provides
        an additional input argument 'eta'. Since the algorithm in [Ref.2] has
        typos, the algorithm in [Ref.1] is implmented here by incorparing 'eta'.

        Args:
            p:  A positive integer, the number of the publishers
            eta:  A non-negative float, a tuning parameter

        Returns:
            Shape <p * p> 2d-array, a correlation matrix
        """
        # Initliaze the correlation matrix
        corr = np.matrix(np.ones(1))

        # Increment the size of correlation matrix by one each time
        for k in range(2, p + 1):
            # sample y = r^2 from a beta distribution
            # with alpha_1 = (k-1)/2 and alpha_2 = (d-k)/2
            y = np.random.beta((k - 1) / 2, (p + 1 - k) / 2 + eta)
            r = np.sqrt(y)

            # sample a unit vector theta uniformly
            # from the unit ball surface B^(k-1)
            v = np.random.randn(k - 1)
            theta = v / np.linalg.norm(v)

            # set w = r theta
            w = np.dot(r, theta)

            # set q = corr**(1/2) w
            q = np.dot(linalg.sqrtm(corr), w)

            # incrementally create the next_corr
            next_corr = np.zeros((k, k))
            next_corr[: (k - 1), : (k - 1)] = corr
            next_corr[k - 1, k - 1] = 1
            next_corr[k - 1, : (k - 1)] = q
            next_corr[: (k - 1), k - 1] = q

            corr = next_corr
