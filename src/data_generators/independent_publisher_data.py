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

from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from io import IOBase
from numpy.random import randint
from typing import Dict
from typing import Iterable
from typing import Tuple
from wfa_planning_evaluation_framework.data_generators.impression_generator import (
    ImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.pricing_generator import (
    PricingGenerator,
)


from cardinality_estimation_evaluation_framework.estimators.same_key_aggregator import (
    StandardizedHistogramEstimator,
)

