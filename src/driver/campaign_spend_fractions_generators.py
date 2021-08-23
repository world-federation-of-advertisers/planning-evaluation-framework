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

# A dictionary of functions that maps the number of publishers p
# to a list of length p.  The i-th element of the output list is
# the campaign spend fraction at publisher i.

CAMPAIGN_SPEND_FRACTIONS_GENERATORS = {
    "all_0.2": lambda p: [0.2] * p,
    "cyc_0.1_0.2_0.3": lambda p: list(islice(cycle([0.1, 0.2, 0.3]), p)),
    "all_0.6": lambda p: [0.6] * p,
    "cyc_0.4_0.8": lambda p: list(islice(cycle([0.4, 0.8]), p)),
    "cyc_0.3_0.6_0.9": lambda p: list(islice(cycle([0.3, 0.6, 0.9]), p)),
}
