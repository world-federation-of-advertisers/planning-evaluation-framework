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
"""Tests for modeling_strategy_descriptor."""

from absl.testing import absltest
from typing import Dict
from typing import Type
import numpy as np
from wfa_planning_evaluation_framework.models.goerg_model import (
    GoergModel,
)
from wfa_planning_evaluation_framework.models.reach_curve import (
    ReachCurve,
)
from wfa_planning_evaluation_framework.models.reach_surface import (
    ReachSurface,
)
from wfa_planning_evaluation_framework.models.pairwise_union_reach_surface import (
    PairwiseUnionReachSurface,
)
from wfa_planning_evaluation_framework.simulator.modeling_strategy import (
    ModelingStrategy,
)
from wfa_planning_evaluation_framework.driver.modeling_strategy_descriptor import (
    MODELING_STRATEGIES,
    ModelingStrategyDescriptor,
)


class FakeModelingStrategy(ModelingStrategy):
    def __init__(self,
                 single_pub_model: Type[ReachCurve],
                 single_pub_model_kwargs: Dict,
                 multi_pub_model: Type[ReachSurface],
                 multi_pub_model_kwargs: Dict,
                 x: int):
        self.name = 'fake'
        self.x = 1
        super().__init__(single_pub_model, single_pub_model_kwargs,
                         multi_pub_model, multi_pub_model_kwargs)
        
    
class ModelingStrategyDescriptorTest(absltest.TestCase):
    def test_modeling_strategy_descriptor(self):
        MODELING_STRATEGIES['fake'] = FakeModelingStrategy
        desc = ModelingStrategyDescriptor('fake', {'x':1}, 'goerg', {}, 'pairwise_union', {})
        strategy = desc.instantiate_strategy()
        self.assertEqual(strategy.name, 'fake')
        self.assertEqual(strategy.x, 1)
        self.assertEqual(strategy._single_pub_model, GoergModel)
        self.assertEqual(strategy._single_pub_model_kwargs, {})
        self.assertEqual(strategy._multi_pub_model, PairwiseUnionReachSurface)
        self.assertEqual(strategy._multi_pub_model_kwargs, {})


if __name__ == "__main__":
    absltest.main()
