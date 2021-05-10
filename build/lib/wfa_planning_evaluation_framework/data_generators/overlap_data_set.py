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
from wfa_planning_evaluation_framework.data_generators.data_set import (
    DataSet,
)
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator_base import (
    SetGeneratorBase,
)


class OverlapDataSet(DataSet):
  """Add overlap to a list of single-pub PublisherData.

  Once we have a list of PublisherData for each publsiher, this class
  generates the cross-publisher reach overlap, relabel the reached ids to
  reflect the overlap, and finally generates a new list of PublisherData.
  """

  def __init__(self,
               unlabeled_publisher_data_list: Iterable[PublisherData],
               overlap_generator: SetGeneratorBase,
               overlap_generator_kwargs: Dict[str, Any] = {},
               name: str = None) -> DataSet:
    """Constructor for the PublisherOverlapGenerator.
    """
    set_sizes = [pub_data.max_reach
                 for pub_data in unlabeled_publisher_data_list]
    set_ids_gen = overlap_generator(
        set_sizes=set_sizes, **overlap_generator_kwargs)
    super().__init__(
        publisher_data_list=OverlapDataSet._map_ids(
            set_ids_gen, unlabeled_publisher_data_list),
        name=name)

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
