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
"""A class of a primitive region in a Venn diagram reach."""

from typing import List
from typing import NamedTuple


class VennDiagramRegion(NamedTuple):
    """A single primitive region in a Venn digram reach.
    
    impressions:  The number of impressions that were served by
      each publisher.
    spends:  The amount that was spent at this region on each publisher.
    kplus_reaches:  An iterable of values representing the number of 
      people reached at various frequencies.  kplus_reaches[k] is the 
      number of people who were reached AT LEAST k+1 times.

    """
    impressions: List[int]
    spend: List[float]
    kplus_reaches: List[int]