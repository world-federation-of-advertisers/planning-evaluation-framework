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
"""Base class for DataDesign

A DataDesign represents the collection of DataSets against which
an experimental design is to be evaluated.  Because the size of all
of the DataSets in a DataDesign can be large, the individual
DataSets within a DataDesign are loaded lazily.
"""

from os import listdir
from os.path import exists, isdir, join
from typing import List

from wfa_planning_evaluation_framework.filesystem_wrappers import (
    filesystem_wrapper_base,
)
from wfa_planning_evaluation_framework.filesystem_wrappers import (
    filesystem_pathlib_wrapper,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
import numpy as np


FsWrapperBase = filesystem_wrapper_base.FilesystemWrapperBase
FsPathlibWrapper = filesystem_pathlib_wrapper.FilesystemPathlibWrapper


class DataDesign:
    """A collection of DataSets used for evaluating an ExperimentalDesign."""

    def __init__(self, dirpath: str, filesystem: FsWrapperBase = FsPathlibWrapper()):
        """Constructor

        Args:
          dirpath:  The directory on disk where the DataSets comprising this
            DataDesign will be stored.
          filesystem:  The filesystem object that manages all file operations.
        """
        self._dirpath = dirpath
        self._data_set_names = set()
        self._filesystem = filesystem

        self._filesystem.mkdir(dirpath, parents=True, exist_ok=True)
        for p in sorted(self._filesystem.glob(dirpath, "*")):
            if self._filesystem.is_dir(p):
                self._data_set_names.add(self._filesystem.name(p))

    @property
    def count(self) -> int:
        """Number of DataSets represented in this design."""
        return len(self._data_set_names)

    @property
    def names(self) -> List[str]:
        """Returns a list of the DataSet names in this DataDesign."""
        return sorted(self._data_set_names)

    def by_name(self, name: str) -> DataSet:
        """Returns the DataSet having the given name."""
        return DataSet.read_data_set(join(self._dirpath, name), self._filesystem)

    def add(self, data_set: DataSet) -> None:
        """Adds a DataSet to this DataDesign."""
        ## chenwei ##
        if len(data_set.name) > 255:
            new_data_set_name = data_set.name[:245] + str(np.random.randint(1e10))
        else:
            new_data_set_name = data_set.name
        data_set_path = self._filesystem.joinpath(self._dirpath, new_data_set_name)
        #############
        if self._filesystem.exists(data_set_path):
            raise ValueError(
                "This DataDesign already contains a DataSet with name {}".format(
                    data_set.name
                )
            )
        data_set.write_data_set(self._dirpath, filesystem=self._filesystem)
        self._data_set_names.add(data_set.name)
