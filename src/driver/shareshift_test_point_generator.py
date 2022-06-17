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
        rng: np.random.Generator = np.random.default_rng(0),
    ):
        """Constructs a ShareShiftTestPointGenerator.


        A ShareShift testing point means a spend vector is obtained by "share-shifting"
        from the original spend vector.  "Share-shifting" means increasing/descreasing
        the budget on one pub, and descreasing/increasing the budgets proportionally
        on other pubs to keep the same total budget.

        For example, consider
            * 3 pubs with original spend vector = [1, 2, 3],
            * 2 shift fractions of interest, -0.5 and 0.5,
        then this class generates 3 * 2 spend vectors for testing:
            a. Decreasing budget on pub 1, proportionally increasing 2 & 3
                [1 - 0.5 * 1, 2 + (0.5 * 1) * 2 / (2 + 3), 3 + (0.5 * 1) * 3 / (2 + 3)]
            b. Increasing budget on pub 1, proportionally increasing 2 & 3
                [1 + 0.5 * 1, 2 - (0.5 * 1) * 2 / (2 + 3), 3 - (0.5 * 1) * 3 / (2 + 3)]
            c. Decreasing budget on pub 2, proportionally increasing 1 & 3
                [1 + (0.5 * 2) * 1 / (1 + 3), 2 - 0.5 * 2, 3 + (0.5 * 2) * 3 / (1 + 3)]
            d. Increasing budget on pub 2, proportionally decreasing 1 & 3
                [1 - (0.5 * 2) * 1 / (1 + 3), 2 + 0.5 * 2, 3 - (0.5 * 2) * 3 / (1 + 3)]
            * [1 + (0.5 * 3) * 1 / (1 + 2), 2 + (0.5 * 3) * 2 / (1 + 2), 3 - 3 * 0.5]
            * [1 - (0.5 * 3) * 1 / (1 + 2), 2 - (0.5 * 3) * 2 / (1 + 2), 3 + 3 * 0.5]

        Generally, consider
            * p pubs with original spend vector = [s_1, ..., s_p],
            * m shift fractions of interest, a_1, ..., a_m,
        we generate p * m spend vectors for testing:
        For i in {1, ..., p}:
            For j in {1, ..., m}:
                The (i, j)-th spend vector v has
                v[i] = s_i + a_j * s_i, and
                for any k != i,
                v[k] = s_k - (a_j * s_i) * <proprotion shift to pub k>
                where <proprotion shift to pub k> = s_k / (sum(s) - s_i).
        (Note that a testing point will be dropped if any of its element is negative,
        so there could be fewer than p * m testing points.)

        Args:
            dataset:  The DataSet for which test points are to be generated.
            campaign_spend_fractions:  A length <p> array where p = #pubs.
                Indicates the <orginal spend vector> in the above description normalized
                by the inventory max spends.
            shift_fraction_choices:  A length <m> list where m = #shift fractions of interest.
                Indicates the shift fractions a_1, ..., a_m in the above description.
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
                v[i] = s_i + a * s_i, and
                for any k != i,
                v[k] = s_k - (a * s_i) * <proprotion shift to pub k>
                where <proprotion shift to pub k> = s_k / (sum(s) - s_i).
            where i = selected_pub and a = shift_fraction.
        """
        v = self._campaign_spends.copy()
        shift_amount = v[selected_pub] * shift_fraction
        if v[selected_pub] < sum(v):
            # The k-th entry of shift_proportions is the
            # <proprotion shift to pub k> in the description
            shift_proportions = v / (sum(v) - v[selected_pub])
        else:
            # In the extreme case that sum(v) - v[selected_pub] = 0, i.e.,
            # all other pubs (except the selected_pub) have original spends = 0,
            # we just evenly allocate the shift to all other pubs.
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
