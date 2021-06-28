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

import numpy as np
from typing import List
from typing import NamedTuple
from typing import Callable


class LiquidLegionsParameters(NamedTuple):
    """Simulation parameters specific to Liquid Legions.

    decay_rate:  Decay rate parameter of Liquid Legions
        sketch (often abbreviated as a).
    sketch_size:  Size of Liquid Legions sketch (often
        abbreviated as m).
    random_seed:  A common random seed that is used for
        creating all Liquid Legions sketches.  This determines the hash
        function.  Any two sketches that might need to be combined need
        to be created with the same seed.
    """

    decay_rate: int = 12
    sketch_size: int = int(1e5)
    random_seed: int = 1


class SystemParameters(NamedTuple):
    """Parameters defining a simulation run.

    Attributes:
        campaign_spend_fractions:  A list of values, each between 0 and 1,
            one per campaign, representing the amount spent on the campaign
            as a fractrion of total possible.
        liquid_legions:  Parameters specific to constructing Liquid Legions
            sketches.
        generator:  The single source of randomness that will be used
            for this modeling run.
        campaign_spend_fractions_generator:  A function that maps the number
            of publishers to campaign_spend_fractions of the corresponding
            length. If this is an empty function, then read
            campaign_spend_fractions from the first argument directly.
            Otherwise, the campaign_spend_fractions will always be updated using
            this function in the halo_simulator.
            Examples:
            (1) lambda npublishers: [0.2] * npublishers
            All publishers have campaign spend fraction=0.2.
            (2) lambda npublishers: list(islice(cycle([0.1, 0.2, 0.5]), npublishers))
            Roughly one third of publishers have camapign spend fraction=0.1,
            0.2, 0.5 respectively.
    """

    campaign_spend_fractions: List[float] = [0]
    liquid_legions: LiquidLegionsParameters = LiquidLegionsParameters()
    generator: np.random.Generator = np.random.default_rng(1)
    campaign_spend_fractions_generator: Callable[[int], List[float]] = lambda x: None

    def __str__(self) -> str:
        spend_str = ",".join([f"{s}" for s in self.campaign_spend_fractions])
        ll_str = "decay_rate={},sketch_size={}".format(
            self.liquid_legions.decay_rate, self.liquid_legions.sketch_size
        )
        return "spends=[{}],{}".format(spend_str, ll_str)
