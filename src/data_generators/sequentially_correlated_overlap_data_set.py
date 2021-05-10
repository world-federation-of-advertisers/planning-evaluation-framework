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
"""Generate sequentially correlated multi-pub data from single-pub data."""

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
  """Construct a multi-pub DataSet with sequentially correlated overlap."""

  def __init__(self,
               unlabeled_publisher_data_list: Iterable[PublisherData],
               order: OrderOptions,
               correlated_sets: CorrelatedSetsOptions,
               shared_prop: float,
               random_state: RandomState = None,
               name: str = 'sequentially_correlated') -> DataSet:
    """Constructor for IndependentOverlapDataSet.

    Args:
      unlabeled_publisher_data_list:  a list of PublisherDataSet indicating the
        reach curve of a publisher.
      order: The order of the sets to be returned. It should be one of
        'original', 'reversed' and 'random'.
        Here a 'set' means the reached ids of a publisher.
      correlated_sets: One of 'all' and 'one', indicating how the current set
        is correlated with the previously generated sets when the order is
        'original'.
      shared_prop: A number between 0 and 1 that specifies the proportion of ids
        in the current set that are overlapped with the previous set(s).
        See wfa_cardinality_estimation_evaluation_framework.simulations.set_generator
        for more explanations on the args order, correlated_sets, shared_prop.
      random_state: a random state for generating the sequentially correlated
        reached ids.
      name:  If specified, a human-readable name that will be associated to this
        DataSet.
    """
    super().__init__(
        unlabeled_publisher_data_list=unlabeled_publisher_data_list,
        overlap_generator=SequentiallyCorrelatedSetGenerator,
        overlap_generator_kwargs={
            'order': order, 'correlated_sets': correlated_sets,
            'shared_prop': shared_prop, 'random_state': random_state},
        name=name)
