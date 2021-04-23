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
"""Base class for data designs.

A data design represents the collection of data sets against which
an experimental design is to be evaluated.
"""

from os import listdir
from os.path import isdir, join
from pathlib import Path
from typing import Iterable
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet


class DataDesign:
    """A collection of DataSets used for evaluating an ExperimentalDesign."""

    def __init__(self, data_sets: Iterable[DataSet]):
        """Constructor

        Args:
          data_sets:  An iterable list of DataSets, one for each experiment
            that is to be conducted.
        """
        self._data_sets = data_sets

    @property
    def data_set_count(self):
        """Number of DataSets represented in this design."""
        return len(self._data_sets)

    def data_set(self, index: int) -> DataSet:
        """Returns the DataSet with the given index."""
        return self._data_sets[index]

    @classmethod
    def read_data_design(cls, dirpath: str) -> "DataDesign":
        """Reads a DataDesign from disk."""
        data_sets = []
        for dir in sorted(listdir(dirpath)):
            if isdir(join(dirpath, dir)):
                data_sets.append(DataSet.read_data_set(join(dirpath, dir)))
        return cls(data_sets)

    def write_data_design(self, dirpath: str) -> None:
        """Writes this DataDesign object to disk."""
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        for data_set in self._data_sets:
            data_set.write_data_set(dirpath)
