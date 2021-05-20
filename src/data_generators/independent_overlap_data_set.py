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
"""Generate independent multi-pub data from single-pub data."""

import math
from typing import Iterable
from numpy.random import Generator
from numpy.random import RandomState
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator import (
    IndependentSetGenerator,)
from wfa_planning_evaluation_framework.data_generators.data_set import (
    DataSet,)
from wfa_planning_evaluation_framework.data_generators.overlap_data_set import (
    OverlapDataSet,)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,)


class IndependentOverlapDataSet(OverlapDataSet):
  """Construct a multi-pub DataSet with independent overlap."""

  def __init__(
      self,
      unlabeled_publisher_data_list: Iterable[PublisherData],
      universe_size: int,
      random_generator: Generator,
      name: str = 'independent',
  ) -> DataSet:
    """Constructor for IndependentOverlapDataSet.
      Args:
        unlabeled_publisher_data_list:  a list of PublisherDataSet indicating
          the reach curve of a publisher.
        universe_size:  the universe size for applying the independent model of
          overlap. Explicitly, for any two pubs 1 and 2, the overlap reach
          between these two pubs 1 equals <pub 1 reach> * <pub 2 reach> /
          universe_size.
        random_state: a random state for generating the independent reached ids.
        name:  If specified, a human-readable name that will be associated to
          this DataSet.
    """

  super().__init__(
      unlabeled_publisher_data_list=unlabeled_publisher_data_list,
      overlap_generator=IndependentSetGenerator,
      overlap_generator_kwargs={
          'universe_size':
              universe_size,
          'random_state':
              RandomState(seed=random_generator.integers(100000, size=1)[0])
      },
      name=name)
