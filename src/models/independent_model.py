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
"""Independent multi publisher model."""

import numpy as np
from typing import List
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface


class IndependentModel(ReachSurface):
    """Dirac mixture multi publisher k+ model."""

    def __init__(
        self,
        reach_points: List[ReachPoint] = None,
        reach_curves: List[ReachCurve] = None,
        universe_size: int = 0,
    ):
        self._reach_curves = reach_curves
        self.universe_size = universe_size

    def _fit(self):
        return

    def by_impressions(
        self, impressions: List[int], max_frequency: int = 1
    ) -> ReachPoint:
        marginal_rps = [
            curve.by_impressions([imp], max_frequency)
            for curve, imp in zip(self._reach_curves, impressions)
        ]
        marginal_freq_hists = [
            rp._frequencies + [rp._kplus_reaches[-1]] for rp in marginal_rps
        ]
        marginal_zero_included_relative_freq_hists = [
            np.array([self.universe_size - sum(hist)] + hist) / self.universe_size
            for hist in marginal_freq_hists
        ]
        union_zero_included_relative_freq_hist = [1]
        for hist in marginal_zero_included_relative_freq_hists:
            union_zero_included_relative_freq_hist = np.convolve(
                union_zero_included_relative_freq_hist, hist
            )
            truncated_reach = sum(
                union_zero_included_relative_freq_hist[max_frequency:]
            )
            union_zero_included_relative_freq_hist = (
                union_zero_included_relative_freq_hist[: (max_frequency + 1)]
            )
            union_zero_included_relative_freq_hist[-1] = truncated_reach
        relative_kplus_reaches_from_zero = np.cumsum(
            union_zero_included_relative_freq_hist[::-1]
        )[::-1]
        kplus_reaches = (
            (self.universe_size * relative_kplus_reaches_from_zero[1:])
            .round(0)
            .astype("int32")
        )
        return ReachPoint(
            impressions=impressions,
            kplus_reaches=kplus_reaches,
            universe_size=self.universe_size,
        )
