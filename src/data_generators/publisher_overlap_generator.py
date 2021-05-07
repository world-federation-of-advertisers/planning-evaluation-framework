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

import numpy as np
from typing import Iterable
from typing import List
from typing import Dict
from typing import Any
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator_base import (
    SetGeneratorBase,
)


class PublisherOverlapGenerator:
  """Add overlap to a list of single-pub PublisherData.

  Once we have a list of PublisherData for each publsiher, this class
  generates the cross-publisher reach overlap, relabel the reached ids to
  reflect the overlap, and finally generates a new list of PublisherData.
  """

  def __init__(self, overlap_generator: SetGeneratorBase):
    """Constructor for the PublisherOverlapGenerator.

    This would typically be overridden with a method whose signature would
    specify the various parameters of the publisher overlap to be generated.
    """
    self.overlap_generator = overlap_generator

  def __call__(self, publisher_data_iter: Iterable[PublisherData],
               overlap_generator_kwargs: Dict[str, Any]) -> List[PublisherData]:
    """Generate a list of PublisherData with overlap reach."""
    set_sizes = [pub_data.max_reach for pub_data in publisher_data_iter]
    set_ids_gen = self.overlap_generator(
        set_sizes=set_sizes, **overlap_generator_kwargs)
    a = PublisherOverlapGenerator._map_ids(set_ids_gen, publisher_data_iter)
    return a

  @classmethod
  def _map_ids(cls,
               set_ids_iter: Iterable[np.array],
               publisher_data_iter: Iterable[PublisherData]):
    new_publisher_data_list = []
    for set_ids, pub_data in zip(set_ids_iter, publisher_data_iter):
      assert len(set_ids) == pub_data.max_reach, 'single-pub reach does not match.'
      original_ids = set([oid for oid, _ in pub_data._data])
      id_map = dict(zip(original_ids, set_ids))
      new_impression_log_data = [(id_map[oid], x) for oid, x in pub_data._data]
      new_publisher_data_list.append(
          PublisherData(new_impression_log_data, pub_data.name)
      )
    return new_publisher_data_list
