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
from copy import deepcopy

from wfa_planning_evaluation_framework.models.gamma_poisson_model import (
    GammaPoissonModel,
)
from wfa_planning_evaluation_framework.models.goerg_model import (
    GoergModel,
)
from wfa_planning_evaluation_framework.models.kinflated_gamma_poisson_model import (
    KInflatedGammaPoissonModel,
)
from wfa_planning_evaluation_framework.models.dirac_mixture_single_publisher_model import (
    DiracMixtureSinglePublisherModel,
)
from wfa_planning_evaluation_framework.models.pairwise_union_reach_surface import (
    PairwiseUnionReachSurface,
)
from wfa_planning_evaluation_framework.models.restricted_pairwise_union_reach_surface import (
    RestrictedPairwiseUnionReachSurface,
)
from wfa_planning_evaluation_framework.models.dirac_mixture_multi_publisher_model import (
    DiracMixtureMultiPublisherModel,
)
from wfa_planning_evaluation_framework.models.independent_model import (
    IndependentModel,
)
from wfa_planning_evaluation_framework.simulator.modeling_strategy import (
    ModelingStrategy,
)
from wfa_planning_evaluation_framework.simulator.m3_strategy import (
    M3Strategy,
)
from wfa_planning_evaluation_framework.simulator.local_dp_strategy import (
    LocalDpLiquidlegionsStrategy,
)
from wfa_planning_evaluation_framework.simulator.global_dp_strategy import (
    GlobalDpLiquidlegionsStrategy,
)
from wfa_planning_evaluation_framework.simulator.single_publisher_strategy import (
    SinglePublisherStrategy,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet


# A dictionary mapping names of single publisher models to the
# corresponding classes that implement them.
SINGLE_PUB_MODELS = {
    "goerg": GoergModel,
    "gamma_poisson": GammaPoissonModel,
    "kinflated_gamma_poisson": KInflatedGammaPoissonModel,
    "dirac_mixture_single": DiracMixtureSinglePublisherModel,
}

# A dictionary mapping names of multipublisher models to the
# corresponding classes that implement them.
MULTI_PUB_MODELS = {
    "pairwise_union": PairwiseUnionReachSurface,
    "restricted_pairwise_union": RestrictedPairwiseUnionReachSurface,
    "dirac_mixture_multi": DiracMixtureMultiPublisherModel,
    "independent": IndependentModel,
    "none": None,
    # TODO: Uncomment the following after the Dirac Mixture model is implemented.
    # 'dirac_mixture': DiracMixtureReachSurface,
    # TODO: Uncomment the following after the Generalized Mixture model is implemented.
    # 'generalized_mixture': GeneralizedMixtureReachSurface,
}

# A dictionary mapping names of modeling strategies to the
# corresponding classes that implement them.
MODELING_STRATEGIES = {
    "m3strategy": M3Strategy,
    "single_publisher": SinglePublisherStrategy,
    "local_dp": LocalDpLiquidlegionsStrategy,
    "global_dp": GlobalDpLiquidlegionsStrategy,
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
    strategy_kwargs: Dict = {}
    single_pub_model: str = ""
    single_pub_model_kwargs: Dict = {}
    multi_pub_model: str = ""
    multi_pub_model_kwargs: Dict = {}

    def update_from_dataset(
        self, data_set: DataSet = None
    ) -> "ModelingStrategyDescriptor":
        if self.strategy == "local_dp":
            return self
        if self.strategy == "global_dp":
            return self

        largest_pub_size = max([pub.max_reach for pub in data_set._data])
        single_pub_model_kwargs = deepcopy(self.single_pub_model_kwargs)
        if "largest_pub_to_universe_ratio" in self.single_pub_model_kwargs:
            single_pub_model_kwargs["universe_size"] = int(
                largest_pub_size
                / single_pub_model_kwargs["largest_pub_to_universe_ratio"]
            )
            del single_pub_model_kwargs["largest_pub_to_universe_ratio"]
        multi_pub_model_kwargs = deepcopy(self.multi_pub_model_kwargs)
        if "largest_pub_to_universe_ratio" in self.multi_pub_model_kwargs:
            multi_pub_model_kwargs["universe_size"] = int(
                largest_pub_size
                / multi_pub_model_kwargs["largest_pub_to_universe_ratio"]
            )
            del multi_pub_model_kwargs["largest_pub_to_universe_ratio"]
        return ModelingStrategyDescriptor(
            strategy=deepcopy(self.strategy),
            strategy_kwargs=deepcopy(self.strategy_kwargs),
            single_pub_model=deepcopy(self.single_pub_model),
            single_pub_model_kwargs=single_pub_model_kwargs,
            multi_pub_model=deepcopy(self.multi_pub_model),
            multi_pub_model_kwargs=multi_pub_model_kwargs,
        )

    def instantiate_strategy(self):
        """Returns ModelingStrategy object defined by this descriptor."""
        if self.strategy == "local_dp":
            return LocalDpLiquidlegionsStrategy()
        if self.strategy == "global_dp":
            return GlobalDpLiquidlegionsStrategy()
        return MODELING_STRATEGIES[self.strategy](
            SINGLE_PUB_MODELS[self.single_pub_model],
            self.single_pub_model_kwargs,
            MULTI_PUB_MODELS[self.multi_pub_model],
            self.multi_pub_model_kwargs,
            **self.strategy_kwargs,
        )

    def _dict_to_string(self, s: str, d: Dict) -> str:
        """Example output:  s(a=b,c=d,e=f)"""
        if not d:
            return s
        return s + "(" + ",".join([f"{k}={v}" for (k, v) in d.items()]) + ")"

    def __str__(self) -> str:
        """Returns string representing this modeling strategy.

        Example outputs:
          m3_strategy,goerg,pairwise_union
          m3_strategy(split=0.2),goerg,pairwise_union(penalty=0.1)
        """
        return (
            self._dict_to_string(self.strategy, self.strategy_kwargs)
            + ","
            + self._dict_to_string(self.single_pub_model, self.single_pub_model_kwargs)
            + ","
            + self._dict_to_string(self.multi_pub_model, self.multi_pub_model_kwargs)
        )
