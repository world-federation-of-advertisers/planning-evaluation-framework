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
"""Test Point Aggregator.

Given a collection of true reach values and a corresponding collection
of reach estimates, computes a single row DataFrame with summary statistics
on the differences between the true values and the estimated values.
"""

import numpy as np
import pandas as pd
from typing import List

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint


# A list of functions for computing aggregation metrics.  Each function
# takes as input the true reach and modeled reach points and produces
# as output a scalar value (could be float, int or string).
AGGREGATORS = {
    # Number of test points
    'npoints': lambda x, y: len(x),

    # Mean error (bias)
    'mean_error': lambda x, y: np.mean(_reach(x) - _reach(y)),
    
    # Mean absolute error in predicted reach
    'mean_abs_error': lambda x, y: np.mean(np.abs(_reach(x) - _reach(y))),

    # Mean squared error in predicted reach
    'mean_squared_error': lambda x, y: np.mean((_reach(x) - _reach(y))**2),
    
    # Mean absolute relative error in predicted reach
    'mean_abs_relative_error': lambda x, y: np.mean(np.abs(_reach(x) - _reach(y))/_reach(x)),
                                           
    # Mean squared relative error in predicted reach
    'mean_squared_relative_error': lambda x, y: np.mean((_reach(x) - _reach(y))**2/_reach(x)),

    # Error variance in predicted reach
    'var_error': lambda x, y: np.var(_reach(x) - _reach(y)),

    # Relative error variance in predicted reach
    'var_relative_error': lambda x, y: np.var((_reach(x) - _reach(y))/_reach(x)),

    # Quantiles of relative error in predicted reach
    'relative_error_q10': lambda x, y: np.quantile(np.abs(_reach(x) - _reach(y))/_reach(x), 0.10),
    'relative_error_q20': lambda x, y: np.quantile(np.abs(_reach(x) - _reach(y))/_reach(x), 0.20),
    'relative_error_q30': lambda x, y: np.quantile(np.abs(_reach(x) - _reach(y))/_reach(x), 0.30),
    'relative_error_q40': lambda x, y: np.quantile(np.abs(_reach(x) - _reach(y))/_reach(x), 0.40),
    'relative_error_q50': lambda x, y: np.quantile(np.abs(_reach(x) - _reach(y))/_reach(x), 0.50),
    'relative_error_q60': lambda x, y: np.quantile(np.abs(_reach(x) - _reach(y))/_reach(x), 0.60),
    'relative_error_q70': lambda x, y: np.quantile(np.abs(_reach(x) - _reach(y))/_reach(x), 0.70),
    'relative_error_q80': lambda x, y: np.quantile(np.abs(_reach(x) - _reach(y))/_reach(x), 0.80),
    'relative_error_q90': lambda x, y: np.quantile(np.abs(_reach(x) - _reach(y))/_reach(x), 0.90),
    
    # Mean shuffle distance
    'mean_shuffle_distance': lambda x, y: np.mean([_shuffle_distance(x[i], y[i]) for i in range(len(x))]),
                                                  
    # Mean squared shuffle distance
    'mean_squared_shuffle_distance': lambda x, y: np.mean([_shuffle_distance(x[i], y[i])**2 for i in range(len(x))]),
    
    # Variance of shuffle distance
    'var_shuffle_distance': lambda x, y: np.var([_shuffle_distance(x[i], y[i]) for i in range(len(x))]),
}

def _reach(point_list: List[ReachPoint], k=1) -> np.array:
    """Returns list of k+ frequencies from list of ReachPoints."""
    return np.array([point.reach(k) for point in point_list])

def _shuffle_distance(xpoint: ReachPoint, ypoint: ReachPoint, k=5) -> float:
    """Computes shuffle distance of first k frequency buckets."""
    if xpoint.max_frequency <= k or ypoint.max_frequency <= k:
        return 1.0
    xfreq = np.array([xpoint.frequency(i+1) for i in range(k)])
    yfreq = np.array([ypoint.frequency(i+1) for i in range(k)])
    if sum(xfreq) == 0 or sum(yfreq) == 0:
        return 0.5
    return 0.5 * np.sum(np.abs(xfreq/sum(xfreq) - yfreq/sum(yfreq)))

def aggregate(true_reach: List[ReachPoint], simulated_reach: List[ReachPoint]) -> pd.DataFrame:
    """Returns a DataFrame of the statistics listed in keys.

    Args:
      keys:  A list of strings.  Each string should specify the name of an aggregation
        statistic, as given in AGGREGATORS.
      true_reach:  A list of points representing true reach values.
      simulated_reach:  A list of points representing modeled reach values.  This list must
        be of the same length as true_reach.  The value of simulated_reach[i] should be the
        output of the modeling function for the spend vector that was used to compute
        true_reach[i].
    Returns:
      A single row DataFrame representing the values of the statistics listed in keys.
    """
    stats = {}
    for key in AGGREGATORS:
        stats[key] = [AGGREGATORS[key](true_reach, simulated_reach)]
    return pd.DataFrame(data = stats)
    
  
