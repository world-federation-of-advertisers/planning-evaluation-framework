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
"""Generates test points for the ShareShift use case."""

import numpy as np
from typing import Iterable, List
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.driver.test_point_generator import (
    TestPointGenerator,
)


class ShareShiftTestPointGenerator(TestPointGenerator):
    """Generates test points that proportionally shift budget to or from a publisher."""

    def __init__(
        self,
        dataset: DataSet,
        campaign_spend_fractions: np.ndarray,
        shift_fraction_choices: List[float] = [-1.0, -0.5, 0.5, 1.0],
    ):
        """Constructs a ShareShiftTestPointGenerator.

        Suppose that the original spend vector is [s_1, ..., s_p].
        This class generates p * m testing points v with:
            v[i] = s_i + s_i * a_k and
            v[j] = s_j - s_j / (sum(s) - s[i]) * (s_i * a_k) for j != i,
        iterating over 1 <= i <= p and k = 1, ..., m.  Here, a_1, ..., a_m are a list
        of shift fractions.
        (Note that a testing point will be dropped if any of its element is negative,
        so there could be fewer than p * m testing points.)

        For example, if
            * the original spend is [1, 2, 3]
            * we consider shift fraction = - 0.5, 0.5,
        then this class generates 2 * 3 testing points with the following spend
        vectors:
            * [1 - 0.5 * 1, 2 + (0.5 * 1) * 2 / (2 + 3), 3 + (0.5 * 1) * 3 / (2 + 3)]
            * [1 + 0.5 * 1, 2 - (0.5 * 1) * 2 / (2 + 3), 3 - (0.5 * 1) * 3 / (2 + 3)]
            * [1 + (0.5 * 2) * 1 / (1 + 3), 2 - 0.5 * 2, 3 + (0.5 * 2) * 3 / (1 + 3)]
            * [1 - (0.5 * 2) * 1 / (1 + 3), 2 + 0.5 * 2, 3 - (0.5 * 2) * 3 / (1 + 3)]
            * [1 + (0.5 * 3) * 1 / (1 + 2), 2 + (0.5 * 3) * 2 / (1 + 2), 3 - 3 * 0.5]
            * [1 - (0.5 * 3) * 1 / (1 + 2), 2 - (0.5 * 3) * 2 / (1 + 2), 3 + 3 * 0.5]

        Args:
            dataset:  The DataSet for which test points are to be generated.
            campaign_spend_fractions:  The campaign_spend / inventory_spend
                at each publisher, where each inventory_spend is given by
                dataset._max_spends.
            shift_fraction_choices:  A list of shift fractions to consider, i.e.,
                the a_1, ..., a_m in the above description.
        """
        super().__init__(dataset)
        self._campaign_spends = self._max_spends * campaign_spend_fractions
        self._shift_fraction_choices = shift_fraction_choices

    def one_testing_point(
        self,
        selected_pub: int,
        shift_fraction: float,
    ) -> np.ndarray:
        """Generates one ShareShift testing point.

        Args:
            selected_pub:  The index of publisher that we shift budget from
                or to.
            shift_fraction:  How much budget we shift from/to the selected
                publisher.  It can be either positive or negative.
                A positive shift_fraction means that we decrease the budget
                at one publisher and proportionally increase budgets at others.
                A negative shift_fraction means that we increase the budget
                at one publisher and proportionally decrease budgets at others.

        Returns:
            The spend vector of the testing point.  Explicitly, this vector
            v has:
                v[i] = s_i - s_i * a and
                v[j] = s_j + s_j / (sum(s) - s[i]) * (s_i * a) for j != i,
            where i = selected_pub and a = shift_fraction.
        """
        v = self._campaign_spends.copy()
        shift_amount = v[selected_pub] * shift_fraction
        if sum(v) - v[selected_pub] > 0:
            shift_proportions = v / (sum(v) - v[selected_pub])
        else:
            shift_proportions = np.array(
                [1 / (self._npublishers - 1)] * self._npublishers
            )
        v += shift_proportions * shift_amount
        v[selected_pub] = self._campaign_spends[selected_pub] - shift_amount
        return v

    def test_points(self) -> Iterable[List[float]]:
        """Returns a generator for generating a list of test points.

        Returns:
            An iterable of spend vectors representing locations where
            the true reach surface is to be compared to the modeled reach
            surface.
        """
        for selected_pub in range(self._npublishers):
            for shift_fraction in self._shift_fraction_choices:
                point = self.one_testing_point(selected_pub, shift_fraction)
                if not any(point < 0):
                    yield point
