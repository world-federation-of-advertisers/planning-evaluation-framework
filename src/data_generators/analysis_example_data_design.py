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
"""A simple example of a data design."""

from typing import Iterable
import itertools
import numpy as np

from wfa_planning_evaluation_framework.data_generators.data_set_parameters import (
    DataSetParameters,
)

# The following are the parameter sets that are varied in this data design.
# The data design constructs the cartesian product of these parameter settings.
NUM_PUBLISHERS = [1, 2, 5]
LARGEST_PUBLISHER = [30, 60]
PUBLISHER_RATIOS = [0.8, 0.2]


def generate_data_design_config(
    random_generator: np.random.Generator,
) -> Iterable[DataSetParameters]:
    """Generates a data design configuration.

    This examples illustrates a simple cartesian product of parameter settings.
    """
    data_design_config = []
    for params in itertools.product(
        NUM_PUBLISHERS, LARGEST_PUBLISHER, PUBLISHER_RATIOS
    ):
        num_publishers, largest_publisher, publisher_ratio = params
        data_design_config.append(
            DataSetParameters(
                num_publishers=num_publishers,
                largest_publisher_size=largest_publisher,
                largest_to_smallest_publisher_ratio=publisher_ratio,
            )
        )
    return data_design_config
