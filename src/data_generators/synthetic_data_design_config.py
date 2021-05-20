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
"""Encapculates the config for a DataDesign."""

from typing import List
from numpy.random import Generator
from wfa_planning_evaluation_framework.data_generators.data_set_parameters import (
    DataSetParameters,
)

NAMING_FIELDS = [
    "num_publishers",
    "largest_publisher_size",
    "largest_to_smallest_publisher_ratio",
]


class SyntheticDataDesignConfig:
    """Encapsulates with synthetic data config.

    This class geneerates DataSetParameters objects to be passed to
    SyntheticDataGenerator.
    """

    @classmethod
    def get_data_set_name(
        cls, data_set_parameters: DataSetParameters, random_generator: Generator
    ):
        # This signature is same for all runs with the same seed. It is
        # deterministic, the same value will be generated for the same seed and the
        # same DataSetParameters that construct the same underliying objects.
        random_signature = str(random_generator.integers(100000, size=1)[0])
        parameter_signature = "_".join(
            [
                x + "=" + str(getattr(data_set_parameters, x))
                for x in data_set_parameters._fields
                if x in NAMING_FIELDS
            ]
        )
        return parameter_signature + "_rs=" + random_signature

    @classmethod
    def get_data_set_params_list(
        random_generator: Generator,
    ) -> List[DataSetParameters]:
        """Generates list of data set parameters to create a data design from.

        Returns:
           List of DataSetParameters objects. These object can be hard coded or be
           constructed through some business logic.
        Args:
          seed:  Random seed to be used for RandomState object.
        """
        pass
