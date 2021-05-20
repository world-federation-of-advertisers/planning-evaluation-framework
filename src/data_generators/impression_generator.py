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
"""Generate random viewing impressions."""

from typing import List


class ImpressionGenerator:
  """Generate a random sequence of viewer id's of ad impressions.
    This class, along with PricingGenerator, assists in the generation of
    random PublisherDataFiles.  The ImpressionGenerator will generate a
    sequence of random impressions according to specified criteria.
    """

  def __init__(self, n: int):
    """Constructor for the ImpressionGenerator.
        This would typically be overridden with a method whose signature
        would specify the various parameters of the impression distribution
        to be generated.
        Args:
          n:  The number of users.
    """
    pass

  def __call__(self) -> List[int]:
    """Generate a random sequence of impressions.
        Returns:
          A list of randomly generated user id's.  An id may occur multiple
          times in the output list, representing the fact that the user may
          see multiple ads from the publisher over the course of the campaign.
        """
    pass
