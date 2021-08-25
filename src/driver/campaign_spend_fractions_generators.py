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
"""Functions to generate campaign_spend_fractions."""

from itertools import cycle, islice

# A class of functions that maps the number of publishers p
# to a list of length p.  The i-th element of the output list is
# the campaign spend fraction at publisher i.


class CampaignSpendFractionsGenerators:
    def all_20(p):
        return [0.2] * p

    def cyc_10_20_30(p):
        return list(islice(cycle([0.1, 0.2, 0.3]), p))

    def all_60(p):
        return [0.6] * p

    def cyc_40_80(p):
        return list(islice(cycle([0.4, 0.8]), p))

    def cyc_30_60_90(p):
        return list(islice(cycle([0.3, 0.6, 0.9]), p))
