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
"""Generate multi-pub data from single-pub data."""

from typing import Iterable
from enum import Enum
from numpy.random import RandomState
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator import (
    SequentiallyCorrelatedSetGenerator,
    ORDER_ORIGINAL, ORDER_REVERSED, ORDER_RANDOM,
    CORRELATED_SETS_ALL, CORRELATED_SETS_ONE,
)
from wfa_planning_evaluation_framework.data_generators.data_set import (
    DataSet,
)
from wfa_planning_evaluation_framework.data_generators.overlap_data_set import (
    OverlapDataSet,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)


class OrderOptions(str, Enum):
  original = ORDER_ORIGINAL
  reversed = ORDER_REVERSED
  random = ORDER_RANDOM


class CorrelatedSetsOptions(str, Enum):
  all = CORRELATED_SETS_ALL
  one = CORRELATED_SETS_ONE


class SequentiallyCorrelatedOverlapDataSet(OverlapDataSet):

  def __init__(self,
               unlabeled_publisher_data_list: Iterable[PublisherData],
               order: OrderOptions,
               correlated_sets: CorrelatedSetsOptions,
               shared_prop: float,
               random_state: RandomState = None,
               name: str = 'sequentially_correlated') -> DataSet:
    super().__init__(
        unlabeled_publisher_data_list=unlabeled_publisher_data_list,
        overlap_generator=SequentiallyCorrelatedSetGenerator,
        overlap_generator_kwargs={
            'order': order, 'correlated_sets': correlated_sets,
            'shared_prop': shared_prop, 'random_state': random_state},
        name=name)
