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

import copy
import numpy as np
from typing import List
from typing import NamedTuple
from typing import Callable
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet


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
            as a fraction of total possible.  Note that
            len(campaign_spend_fractions) must equal the number of publishers.
        liquid_legions:  Parameters specific to constructing Liquid Legions
            sketches.
        generator:  The single source of randomness that will be used
            for this modeling run.
        campaign_spend_fractions_generator:  A function that returns the
            campaign_spend_fractions vector that should be used for a
            particular DataSet.
    """

    campaign_spend_fractions: List[float] = [0]
    liquid_legions: LiquidLegionsParameters = LiquidLegionsParameters()
    generator: np.random.Generator = np.random.default_rng(1)
    campaign_spend_fractions_generator: Callable[[DataSet], List[float]] = None

    def __str__(self) -> str:
        spend_str = ",".join([f"{s}" for s in self.campaign_spend_fractions])
        ll_str = "decay_rate={},sketch_size={}".format(
            self.liquid_legions.decay_rate, self.liquid_legions.sketch_size
        )
        return "spends=({}),{}".format(spend_str, ll_str)

    def update_from_dataset(self, dataset: DataSet) -> "SystemParameters":
        """Uses the dataset to fill in various context-specific items."""
        if self.campaign_spend_fractions_generator:
            campaign_spend_fractions = self.campaign_spend_fractions_generator(dataset)
        else:
            campaign_spend_fractions = copy.deepcopy(self.campaign_spend_fractions)
        return SystemParameters(
            campaign_spend_fractions, copy.deepcopy(self.liquid_legions), self.generator
        )
