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
"""Encapculates the config for a DataSet."""

from typing import List
from wfa_planning_evaluation_framework.data_generators.data_set_parameters import DataSetParameters

class SyntheticDataDesignConfig():
  """Generates a DataDesign with synthetic data derived from parameters.

    This class translates a DataDesignParameters object to a DataDesign by
    constructing the underlying objects for a DataSet and duplicating the
    DataSet with different random seeds.
    """


  @classmethod
  def get_data_set_params_list() -> List[DataSetParameters]:
    """Generates list of data set parameters to create a data design from.

    Returns:
       List of DataSetParameters objects. These object can be hard coded or be 
       consturucted through some business logic.
    """
    pass
