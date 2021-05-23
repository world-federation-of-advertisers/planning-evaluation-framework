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
"""Defines the components and parameters of a specific modeling strategy."""

from numpy.random import Generator
from typing import Dict
from typing import NamedTuple
from typing import Type

from wfa_planning_evaluation_framework.models.goerg_model import (
    GoergModel,
)
from wfa_planning_evaluation_framework.models.pairwise_union_reach_surface import (
    PairwiseUnionReachSurface,
)
from wfa_planning_evaluation_framework.models.restricted_pairwise_union_reach_surface import (
    RestrictedPairwiseUnionReachSurface,
)
from wfa_planning_evaluation_framework.simulator.modeling_strategy import (
    ModelingStrategy,
)

# A dictionary mapping names of single publisher models to the
# corresponding classes that implement them.
SINGLE_PUB_MODELS = {
    "goerg": GoergModel,
}

# A dictionary mapping names of multipublisher models to the
# corresponding classes that implement them.
MULTI_PUB_MODELS = {
    "pairwise_union": PairwiseUnionReachSurface,
    "restricted_pairwise_union": RestrictedPairwiseUnionReachSurface,
}

# A dictionary mapping names of modeling strategies to the
# corresponding classes that implement them.
MODELING_STRATEGIES = {
    # TODO: Uncomment the following after the M3 Proposal is implemented.
    #    'm3proposal': M3Proposal,
}


class ModelingStrategyDescriptor(NamedTuple):
    """Parameters defining a modeling strategy.

    Attributes:
      strategy:  Name of the overall strategy.
      strategy_kwargs:  Dictionary of keyword args for the strategy.
      single_pub_model:  Name of the single publisher model that is
        used for this simulation.
      single_pub_model_kwargs:  Dictionary of keyword args that
        are used for single pub model in this simulation.
      multi_pub_model:  Name of the multipublisher model that is
        used for this simulation.
      multi_pub_model_kwargs:  Dictionary of keyword args that are
        used for the multipublisher model used for this simulation.

    TODO: Add support for pricing models when they are introduced.
    """

    strategy: str
    strategy_kwargs: Dict
    single_pub_model: str
    single_pub_model_kwargs: Dict
    multi_pub_model: str
    multi_pub_model_kwargs: Dict

    def instantiate_strategy(self):
        """Returns ModelingStrategy object defined by this descriptor."""
        return MODELING_STRATEGIES[self.strategy](
            SINGLE_PUB_MODELS[self.single_pub_model],
            self.single_pub_model_kwargs,
            MULTI_PUB_MODELS[self.multi_pub_model],
            self.multi_pub_model_kwargs,
            **self.strategy_kwargs
        )

    def _dict_to_string(self, s: str, d: Dict) -> str:
        if not d:
            return s
        return s + ":" + ",".join(["{}={}".format(k, v) for (k, v) in d.items()])

    def __str__(self) -> str:
        return (
            self._dict_to_string(self.strategy, self.strategy_kwargs)
            + ":"
            + self._dict_to_string(self.single_pub_model, self.single_pub_model_kwargs)
            + ":"
            + self._dict_to_string(self.multi_pub_model, self.multi_pub_model_kwargs)
        )
