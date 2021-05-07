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

from numpy.random import RandomState
from typing import Iterable
from typing import List
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.data_generators.publisher_overlap_generator import (
    PublisherOverlapGenerator,
)
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator import (
    IndependentSetGenerator,
)


class IndependentPublisherOverlapGenerator(PublisherOverlapGenerator):

  def __init__(self):
    super().__init__(IndependentSetGenerator)

  def __call__(self, publisher_data_iter: Iterable[PublisherData],
               universe_size: int,
               random_state: RandomState = None) -> List[PublisherData]:
    return super().__call__(
        publisher_data_iter,
        {'universe_size': universe_size, 'random_state': random_state})
