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
"""Generate multi-pub data (with overlap) from a list of single-pub data."""

from typing import Any
from typing import Dict
from typing import Iterable

import numpy as np
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator_base import (
    SetGeneratorBase,
)
from wfa_planning_evaluation_framework.data_generators.data_set import (
    DataSet,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)


class OverlapDataSet(DataSet):
  """Construct a multi-pub DataSet from a list of single-pub PublisherData.

  Once we have a list of PublisherData for each publisher, this class
  generates the cross-publisher reach overlap, relabels the reached ids to
  reflect the overlap, and finally includes these labeled ids to a DataSet.
  """

  def __init__(self,
               unlabeled_publisher_data_list: Iterable[PublisherData],
               overlap_generator: SetGeneratorBase,
               overlap_generator_kwargs: Dict[str, Any] = {},
               name: str = None) -> DataSet:
    """Constructor for OverlapDataSet.

    Args:
      unlabeled_publisher_data_list:  a list of PublisherDataSet. Each
        PublisherDataSet indicates the reach curve of a publisher. The ids in
        each PublisherDataSet are unlabeled and thus meaningless. The spend and
        frequency of these ids are meaningful. In OverlapDataSet we label these
        ids to reflect the cross-publisher overlap; and preserve the frequency
        and spend of these ids so that each single-pub reach curve is anchored.
      overlap_generator:  a class which generates cross-pub reach overlap from
        per-pub reach. Can be any class in
        wfa_cardinality_estimation_evaluation_framework.simulations.set_generator.
      overlap_generator_kwargs:  the inputs of an overlap_generator include
        set_sizes (indicating per-pub reach) and some other args. These other
        args are specified in overlap_generator_kwargs. For example, if
        overlap_generator=IndependentSetGenerator, then overlap_generator_kwargs
        can be {'universe_size': 1e6, 'random_state': np.random.RandomState(1)}.
      name:  If specified, a human-readable name that will be associated to this
        DataSet.
    """
    set_sizes = [pub_data.max_reach
                 for pub_data in unlabeled_publisher_data_list]
    set_ids_gen = overlap_generator(
        set_sizes=set_sizes, **overlap_generator_kwargs)
    super().__init__(
        publisher_data_list=OverlapDataSet._label_ids(
            set_ids_gen, unlabeled_publisher_data_list),
        name=name)

  @classmethod
  def _label_ids(cls,
                 labeled_set_ids_iter: Iterable[np.array],
                 unlabeled_publisher_data_iter: Iterable[PublisherData]):
    """Label the reached ids to reflect cross-pub overlap.

    Args:
      labeled_set_ids_iter:  a list or generator of per-publisher reached ids.
        These ids are labeled, i.e., meaningful of cross-pub overlap.
      unlabeled_publisher_data_iter:  a list or generator of PublisherData. The
        ids here are unlabeled, i.e., meaningless. For each PublisherData here,
        its i-th id will be labeled as the i-th id in the corresponding
        labeled_set_ids.

    Returns:
      A labeled list of PublisherData.
    """
    new_publisher_data_list = []
    for set_ids, pub_data in zip(labeled_set_ids_iter,
                                 unlabeled_publisher_data_iter):
      assert len(set_ids) == pub_data.max_reach, 'single-pub reach does not match.'
      original_ids = set([oid for oid, _ in pub_data._data])
      id_map = dict(zip(original_ids, set_ids))
      new_impression_log_data = [(id_map[oid], x) for oid, x in pub_data._data]
      new_publisher_data_list.append(
          PublisherData(new_impression_log_data, pub_data.name)
      )
    return new_publisher_data_list
