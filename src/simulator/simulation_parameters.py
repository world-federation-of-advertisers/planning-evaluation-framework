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
"""Defines the execution environment of a simulation instance."""

from typing import List
from typing import NamedTuple


class SimulationParameters(NamedTuple):
    """Parameters defining a simulation run.

    Attributes:
        campaign_spend:  A list of spend values, one per campaign,
            representing the total actual spend for each campaign.
        liquid_legions_a:  Decay rate parameter of Liquid Legions
            sketch.
        liquid_legions_m:  Size of Liquid Legions sketch.
    """

    campaign_spend: List[float]
    liquid_legions_a: int
    liquid_legions_m: int
