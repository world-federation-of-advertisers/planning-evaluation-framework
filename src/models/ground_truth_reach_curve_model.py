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
"""Ground Truth Reach Curve Model.

Reports the true reach as a function of spend or impressions.
"""

from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve


class GroundTruthReachCurveModel(ReachCurve):
    """Ground Truth Reach Curve Model."""

    def __init__(self, data_set: DataSet, publisher_index: int):
        """Constructs an Goerg single point reach model.

        Args:
          data_set:  A DataSet object containing the underlying publisher
            data for this simulation run.
          publisher_index: The index of the publisher within the data_set
            for which a ground truth reach curve is to be provided.
        """
        self._data_set = data_set
        self._publisher_index = publisher_index
        self._max_reach = data_set._data[publisher_index].max_reach

    def by_impressions(self, impressions: [int], max_frequency: int = 1) -> ReachPoint:
        """Returns the estimated reach as a function of impressions.

        Args:
          impressions: list of ints of length 1, specifying the hypothetical number
            of impressions that are shown.
          max_frequency: int, specifies the number of frequencies for which reach
            will be reported.
        Returns:
          A ReachPoint specifying the estimated reach for this number of impressions.
        """
        if len(impressions) != 1:
            raise ValueError("Impressions vector must have a length of 1.")
        data_set_impressions = [0] * self._data_set.publisher_count
        data_set_impressions[self._publisher_index] = impressions[0]
        return self._data_set.reach_by_impressions(data_set_impressions, max_frequency)

    def by_spend(self, spends: [int], max_frequency: int = 1) -> ReachPoint:
        """Returns the estimated reach as a function of spend assuming constant CPM

        Args:
          spend: list of floats of length 1, specifying the hypothetical spend.
          max_frequency: int, specifies the number of frequencies for which reach
            will be reported.
        Returns:
          A ReachPoint specifying the estimated reach for this number of impressions.
        """
        if len(spends) != 1:
            raise ValueError("Spend vector must have a length of 1.")
        data_set_spends = [0] * self._data_set.publisher_count
        data_set_spends[self._publisher_index] = spends[0]
        return self._data_set.reach_by_spend(data_set_spends, max_frequency)

    def impressions_for_spend(self, spend: float) -> int:
        return self._data_set._data[self._publisher_index].impressions_by_spend(spend)
