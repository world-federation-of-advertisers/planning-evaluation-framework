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
"""Base class for publisher data

Represents real or simulated impression log data for a single publisher.
"""

from collections import Counter
from copy import deepcopy
from numpy.random import randint
from os.path import join
from typing import Dict
from typing import Iterable
from typing import Tuple
from wfa_planning_evaluation_framework.data_generators.impression_generator import (
    ImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.pricing_generator import (
    PricingGenerator,
)


class PublisherDataFile:
    """Real or simulated impression log data for a single publisher.

    A PublisherDataFile represents a viewing log for a single publisher.
    It can be thought of as a sequence of events of the form (id, spend),
    where id is the (virtual) id of a user, and spend is the total spend
    amount that would cause this impression to be shown.

    A PublisherDataFile object can be constructed in one of several ways:
      1. It can be constructed directly from a list of (id, spend) tuples.
      2. It can be read from a flat file.
      3. It can be generated randomly.

    In the last case, e.g. random generation, the generation of the publisher
    data file is divided into two steps, which are orthogonal to each other.
    In the first step, the sequence of id's is generated, with multiplicities.
    In the second step, spend information is associated to the id's.  Thus,
    to generate a random PublisherDataFile, two objects must be given: an
    ImpressionGenerator object and a PriceGenerator object.
    """

    def __init__(
        self, impression_log_data: Iterable[Tuple[int, float]], name: str = None
    ):
        """Constructs a PublisherDataFile object from raw event data.

        Internally, the data is sorted in increasing order by total spend.
        To fetch the users reached for a given spend, the prefix of events
        up to that spend are taken.  To fetch the users reached for an
        impression buy of N impressions, the first N events in the sorted
        list are taken.

        Args:
          impression_log_data: An iterable consisting of pairs (id, spend),
            where id is a (virtual) user id, and spend is the total amount
            that would need to be spent in order for this impression to be
            shown.
          name: If specified, a human-readable name that will be associated
            to this publisher data file.  For example, it could be an encoding
            of the parameters that were used to create this publisher data file,
            such as "homog_n10000_lambda5".
        """
        self._data = deepcopy(impression_log_data)
        self._data.sort(key=lambda x: x[1])  # sort by spend
        self._name = name
        self._spend = max([spend for (_, spend) in impression_log_data])
        self._reach = len(set([id for (id, _) in impression_log_data]))

    @property
    def impressions(self):
        """Total number of impressions represented by this PublisherDataFile."""
        return len(self._data)

    @property
    def spend(self):
        """Total spend represented by this PublisherDataFile."""
        return self._spend

    @property
    def reach(self):
        """Total number of unique users represented by this PublisherDataFile."""
        return self._reach

    @property
    def name(self):
        """Returns the name associated with this PublisherDataFile."""
        return self._name

    def spend_by_impressions(self, impressions):
        """Returns the amount spent to obtain a given number of impressions."""
        if not impressions:
            return 0.0
        return self._data[impressions - 1][1]

    def user_counts_by_impressions(self, impressions: int) -> Dict[int, int]:
        """Number of times each user is reached for a given impression buy.

        Args:
          impressions:  The number of impressions that are shown (purchased).
        Returns:
          A dictionary D mapping user id's to frequencies.  D[u] is the number
          of times that user u is reached for the given impression purchase.
        """
        return dict(Counter([id for id, _ in self._data[:impressions]]))

    def user_counts_by_spend(self, spend: float) -> Dict[int, int]:
        """Number of times each user is reached for a given spend.

        Args:
          spend:  The total spend.
        Returns:
          A dictionary D mapping user id's to frequencies.  D[u] is the number
          of times that user u is reached for the given spend.
        """
        return dict(Counter([id for id, s in self._data if s <= spend]))

    def write_publisher_data_file(self, parent_dir: str, filename: str = None) -> None:
        """Writes this PublisherDataFile object to disk.

        Args:
          parent_dir:  The directory where the file is to be written.
          filename:  The name of the file.  If not specified, then the name
            given in the object constructor is used.  If no name was given
            in the object constructor, then a random string is chosen.
        """
        if not filename and not self._name:
            filename = "{:012d}".format(randint(0, 1e12))
        elif not filename:
            filename = self._name
        fullpath = join(parent_dir, filename)
        with open(fullpath, "w") as file:
            for user_id, spend in self._data:
                file.write("{},{}\n".format(user_id, spend))

    @classmethod
    def read_publisher_data_file(cls, filepath: str) -> "PublisherDataFile":
        """Reads a publisher data file from disk and returns the object.

        The file is assumed to consist of a sequence of lines of the form
           id,total_spend
        where id is the id of a (virtual) user and total_spend is the total
        amount that would have to be spent in order for this impression to
        be shown.

        The name associated to the PublisherDataFile object is the last
        component of the filepath.

        Args:
          filepath:  Directory path from which the file should be read.
        Returns:
          The PublisherDataFile object representing the contents of this form.
        """
        impression_log = []
        line_no = 0
        with open(filepath) as file:
            for line in file:
                line_no += 1
                try:
                    id_string, spend_string = line.split(",")
                    impression_log.append((int(id_string), float(spend_string)))
                except (ValueError, RuntimeError) as e:
                    raise RuntimeError("At line {}: {}".format(line_no, line)) from e
        name = filepath.split("/")[-1]
        return cls(impression_log, name)

    @classmethod
    def generate_publisher_data_file(
        cls,
        impression_generator: ImpressionGenerator,
        pricing_generator: PricingGenerator,
        name: str = None,
    ) -> "PublisherDataFile":
        """Generates a random publisher data file.

        Args:
          impression_generator:  An ImpressionGenerator object that returns a
            randomly generated list of id's with multiplicities.
          pricing_generator:  A PricingGenerator object that annotates a list of
            id's with randomly generated price information.
          name:  If specfied, the name that will be associated with this
            PublisherDataFile.
        Returns:
          A randomly generated PublisherDataFile.
        """
        return cls(pricing_generator(impression_generator()), name)
