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
"""PricingGenerator for fixed CPM models."""

from typing import List
from typing import Tuple
from wfa_planning_evaluation_framework.data_generators.pricing_generator import (
    PricingGenerator,
)


class FixedPriceGenerator(PricingGenerator):
    """PricingGenerator for fixed CPM models.

    This PricingGenerator generates pricing information for fixed CPM models.
    This is mainly included as an example of how to write a PricingGenerator.
    """

    def __init__(self, cost_per_impression: float):
        """Constructor for the FixedPriceGenerator.

        Args:
          cost_per_impression:  The cost per impression
        """
        self._cpi = cost_per_impression

    def __call__(self, impressions: List[int]) -> List[Tuple[int, float]]:
        """Generate a random sequence of prices.

        Args:
          impressions:  A list of user id's, with multiplicities, to which
            pricing data is to be associated.
        Returns:
          A list of pairs (user_id, total_spend).  The length of the list would
          be the same as the list of impressions, and user_id's would be in 1-1
          correspondences with those in the list of impressions.  Associated to
          each user_id is the total spend amount at which the impression would be
          included in those shown by the advertiser.
        """
        return [(id, (i + 1) * self._cpi) for i, id in enumerate(impressions)]
