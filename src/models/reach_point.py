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
"""Base class for modeling a reach point.

Represents a single point on either a reach curve or a reach surface.
"""

from typing import Dict
from typing import Iterable
from typing import List
import numpy as np


class ReachPoint:
    """A single point on a reach surface."""

    def __init__(
        self,
        impressions: Iterable[int],
        kplus_reaches: Iterable[int],
        spends: Iterable[float] = None,
        universe_size: int = None,
    ):
        """Represents a single point on a reach surface.

        Args:
          impressions:  The number of impressions that were served by
            each publisher.  This should be an iterable.
          kplus_reaches:  An iterable of values representing the number
            of people reached at various frequencies.  kplus_reaches[k]
            is the number of people who were reached AT LEAST k+1 times.
          spends:  If given, the amount that was spent at this point on each
            publisher.  An iterable.
          universe_size:  If given, the universe size associated with this
            reach point.
        """
        if spends and len(impressions) != len(spends):
            raise ValueError("impressions and spends must have same length")
        self._impressions = tuple(impressions)
        self._kplus_reaches = tuple(kplus_reaches)
        # TODO: include the last frequency level into the following
        # list, and modify other places (such as dirac mixture models)
        # accordingly.
        self._frequencies = [
            kplus_reaches[i] - kplus_reaches[i + 1]
            for i in range(len(kplus_reaches) - 1)
        ]
        if spends:
            self._spends = tuple(spends)
        else:
            self._spends = None
        self._universe_size = universe_size

    @property
    def impressions(self) -> int:
        """Returns the number of impressions associated with this point."""
        return self._impressions

    @property
    def universe_size(self) -> int:
        """Returns the universe size associated with this point."""
        if self._universe_size is None:
            raise ValueError("Universe size is not given in this ReachPoint")
        return self._universe_size

    @property
    def max_frequency(self) -> int:
        """Maximum frequency reported by this point."""
        return len(self._kplus_reaches)

    def reach(self, k=1) -> int:
        """Returns the k+ reach associated with this point."""
        if not 0 < k <= len(self._kplus_reaches):
            raise ValueError(
                "k {} is out of range. max allowed is {}".format(
                    k, len(self._kplus_reaches)
                )
            )
        return self._kplus_reaches[k - 1]

    def frequency(self, k=1) -> int:
        """Returns the number of people reached with frequency exactly k."""
        if not 0 < k < len(self._kplus_reaches):
            raise ValueError(
                "k {} is out of range. max allowed is {}".format(
                    k, len(self._kplus_reaches)
                )
            )
        return self._frequencies[k - 1]

    @property
    def frequencies(self) -> List[int]:
        return self._frequencies

    @property
    def frequencies_with_kplus_bucket(self) -> List[int]:
        return self._frequencies + [self._kplus_reaches[-1]]

    @property
    def zero_included_histogram(self) -> List[int]:
        """The zero-included frequency histogram of a ReachPoint.

        Translate the ReachPoint to a vector v where v[f] is the reach at
        frequency f, for 0 <= f <= F - 1, and v[F] = the reach with frequency
        >= F, where F is the maximum frequency of the given ReachPoint.
        """
        if self._universe_size is None:
            raise ValueError(
                "It requires the universe size to be known to obtain a zero-included histogram. "
                "Please specify the universe size of this ReachPoint."
            )
        return [self.universe_size - self.reach(1)] + self.frequencies_with_kplus_bucket

    @property
    def spends(self) -> Iterable[float]:
        """Returns the spends associated with this point."""
        return self._spends

    @staticmethod
    def frequencies_to_kplus_reaches(frequencies: Iterable[int]):
        """Converts a list of frequencies to corresponding k-plus reaches.

        Args:
          frequencies:  List of frequencies, where frequencies[i] is the
            number of people reached exactly i+1 times.
        Returns:
          kplus_reaches, where kplus_reaches[k] is the number of people
            reached k+1 or more times.
        """
        kplus_reaches = frequencies.copy()
        for k in range(len(frequencies) - 2, -1, -1):
            kplus_reaches[k] += kplus_reaches[k + 1]
        return kplus_reaches

    @staticmethod
    def user_counts_to_frequencies(
        counts: Dict[int, int], max_frequency: int
    ) -> List[int]:
        """Constructs frequency counts from a dictionary of per-id reach counts.

        Args:
          counts: A dictionary mapping user id to the number of times that
            the id is reached.
          max_frequency: The maximum frequency to include in the output
            list of k+ reaches.
        Returns:
          frequency_counts: List of frequencies with frequency cap, where
            frequencies[i] is the number of people reached exactly i+1 times.
        """
        frequency_counts = [0] * max_frequency
        for c in counts.values():
            frequency_counts[min(c, max_frequency) - 1] += 1
        return frequency_counts

    @staticmethod
    def user_counts_to_kplus_reaches(
        counts: Dict[int, int], max_frequency: int
    ) -> List[int]:
        """Constructs k+ reaches from a dictionary of per-id reach counts.

        Args:
          counts: A dictionary mapping user id to the number of times that
            the id is reached.
          max_frequency: The maximum frequency to include in the output
            list of k+ reaches.
        Returns:
          kplus_reaches, where kplus_reaches[k] is the number of people
            reached k+1 or more times.
        """
        kplus_reaches = ReachPoint.user_counts_to_frequencies(counts, max_frequency)
        for i in range(max_frequency - 2, -1, -1):
            kplus_reaches[i] += kplus_reaches[i + 1]
        return kplus_reaches
