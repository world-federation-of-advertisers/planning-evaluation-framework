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
"""Base class for data sets.

Represents real or simulated impression log data for multiple publishers.
"""

from collections import defaultdict
from copy import deepcopy
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from functools import lru_cache
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint


class DataSet:
    """Real or simulated impression log data for multiple publishers.

    A DataSet represents a real or simulated configuration of impression log
    data for a collection of related campaigns across multiple publishers.
    It represents the basic unit across which modeling strategies are compared.

    It is expected that this class will be sub-classed for each of the different
    types of publisher overlap models that will be investigate.  Thus, we might
    have an IndependentDataSet, a SequentiallyCorrelatedDataSet, etc.
    """

    def __init__(self, publisher_data_list: Iterable[PublisherData], name: str = None):
        """Constructor

        Args:
          publisher_data_list:  An iterable list of PublisherDatas,
            one for each of the publishers that comprise this DataSet.
          name: If specified, a human-readable name that will be associated
            to this DataSet.  For example, it could be an encoding
            of the parameters that were used to create this DataSet,
            such as "homog_p=10_rep=3".  If no name is given, then a random
            digit string is assigned as the name.
        """
        # TODO: Resolve issue of using deepcopy in Dataflow mode
        # self._data = deepcopy(publisher_data_list)
        self._data = publisher_data_list

        total_audience = set()
        for pub in self._data:
            total_audience.update([id for id, _ in pub._data])
        self._maximum_reach = len(total_audience)

        if name:
            self._name = name
        else:
            self._name = "{:012d}".format(randint(0, 1e12))

    @property
    def publisher_count(self):
        """Number of publishers represented in this DataSet."""
        return len(self._data)

    @property
    def maximum_reach(self):
        """Total number of reachable people across all publishers."""
        return self._maximum_reach

    @property
    def name(self):
        """Name of this DataSet."""
        return self._name

    def spend_by_impressions(self, impressions: Iterable[int]) -> List[float]:
        """Returns spend vector corresponding to a given impression vector.

        Args:
          impressions:  Iterable of hypothetical impression buys, having
            one value per publisher.
        Returns:
          List of corresponding spends.  If I is the vector of impressions
          and S is the returned vector of spends, then S[k] is the amount
          that would need to be spent with the k-th publisher to obtain
          I[k] impressions.
        """
        return [
            self._data[i].spend_by_impressions(impressions[i])
            for i in range(len(self._data))
        ]

    def impressions_by_spend(self, spends: Iterable[float]) -> List[int]:
        """Returns impression vector corresponding to a given spend vector.

        Args:
          spends:  Iterable of hypothetical spend amounts, having
            one value per publisher.
        Returns:
          List of corresponding impression counts.  If S is the vector of
          spends and I is the returned vector of impression counts, then
          I[k] is the number of impressions that would be obtained for
          a spend of S[k] with publisher k.
        """
        return [
            self._data[i].impressions_by_spend(spends[i])
            for i in range(len(self._data))
        ]

    def reach_by_impressions(
        self, impressions: Iterable[int], max_frequency: int = 10
    ) -> ReachPoint:
        """Number of people reached for a given impression count.

        Args:
          impressions:  A list of impression counts.  The length of the list must
            equal the value of publisher_count.  Specifies the number of impressions
            that each publisher will deliver.
          max_frequency: int, The maximum frequency that should be counted.  All
            frequencies about this amount will be grouped into a single bucket.
        Returns:
          A ReachPoint object representing the k+ reach for each frequency
          in the range 1..max_frequency.
        """
        if len(impressions) != self.publisher_count:
            raise ValueError(
                "Invalid impression vector length.  Got {}, expected {}".format(
                    len(impressions), self.publisher_count
                )
            )
        counts = defaultdict(int)
        spends = []
        for i, imp in enumerate(impressions):
            spends.append(self._data[i].spend_by_impressions(imp))
            for id, freq in self._data[i].user_counts_by_impressions(imp).items():
                counts[id] += freq
        kplus_reaches = self._counts_to_histogram(counts, max_frequency)
        return ReachPoint(impressions, kplus_reaches, spends)

    def _counts_to_histogram(
        self, counts: Dict[int, int], max_frequency: int
    ) -> List[int]:
        """Constructs k+ reach list from a dictionary of per-id reach counts."""
        frequency_counts = [0] * max_frequency
        for c in counts.values():
            frequency_counts[min(c, max_frequency) - 1] += 1
        # At this point, frequency_counts[k] represents the number of people who are
        # reach exactly k+1 times, except that frequency_counts[max_frequency-1] contains
        # the number of people reached at least max_frequency times.  Now, we convert this
        # to a list of k+ reach values.
        for i in range(max_frequency - 2, -1, -1):
            frequency_counts[i] += frequency_counts[i + 1]
        return frequency_counts

    def reach_by_spend(
        self, spends: Iterable[float], max_frequency: int = 10
    ) -> ReachPoint:
        """Number of people reached for a given spend.

        Args:
          spends:  A list of spend amounts.  The length of the list must
            equal the value of publisher_count.  Specifies the amount spent with
            each publisher.
          max_frequency: int, The maximum frequency that should be counted.  All
            frequencies about this amount will be grouped into a single bucket.
        Returns:
          A ReachPoint object representing the k+ reach for each frequency
          in the range 1..max_frequency.
        """
        if len(spends) != self.publisher_count:
            raise ValueError(
                "Invalid spends vector length.  Got {}, expected {}".format(
                    len(spends), self.publisher_count
                )
            )
        counts = defaultdict(int)
        impressions = []
        for i, publisher_spend in enumerate(spends):
            user_counts = self._data[i].user_counts_by_spend(publisher_spend)
            impressions.append(sum(user_counts.values()))
            for id, freq in user_counts.items():
                counts[id] += freq
        kplus_reaches = self._counts_to_histogram(counts, max_frequency)
        return ReachPoint(impressions, kplus_reaches, spends)

    def write_data_set(self, parent_dir: str, dataset_dir: str = None) -> None:
        """Writes this DataSet object to disk.

        Args:
          parent_dir:  The directory where the DataSet is to be written.
          dataset:dir: The directory name of the DataSet itself.  If not
            specified, then the name given in the object constructor is
            used.  If no name was given in the object constructor, then a
            random name is used.
        """
        if not dataset_dir:
            dataset_dir = self._name
        fulldir = join(parent_dir, dataset_dir)
        Path(fulldir).mkdir(parents=True, exist_ok=True)
        for pdf in self._data:
            with open(join(fulldir, pdf.name), "w") as file:
                pdf.write_publisher_data(file)
                file.close()

    @classmethod
    @lru_cache(maxsize=128)
    def read_data_set(cls, dirpath: str) -> "DataSet":
        """Reads a DataSet from disk.

        A DataSet is given by a directory containing a collection of files,
        each of which represents a PublisherDataSet.  The name associated to
        the DataSet object is the last component of the dirpath.

        Args:
          dirpath:  Directory containing the PublisherDataSets that comprise
            this DataSet.
        Returns:
          The DataSet object representing the contents of this directory.
        """
        if dirpath.startswith("gs://"):
            from cloudpathlib import CloudPath as Path
        else:
            from pathlib import Path

        pdf_list = []
        dirpath = Path(dirpath)
        for filepath in sorted(dirpath.glob("*")):
            if filepath.is_file():
                with filepath.open() as file:
                    try:
                        pdf = PublisherData.read_publisher_data(file)
                        pdf.name = filepath
                        pdf_list.append(pdf)
                    except (ValueError, RuntimeError) as e:
                        raise RuntimeError(
                            "In publisher file {}".format(filepath)
                        ) from e
        return cls(pdf_list, dirpath.name)
