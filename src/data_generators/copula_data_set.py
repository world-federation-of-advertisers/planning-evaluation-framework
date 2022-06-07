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