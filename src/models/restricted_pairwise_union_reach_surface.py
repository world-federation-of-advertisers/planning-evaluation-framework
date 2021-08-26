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
"""Class for modeling Restricted Pairwise union reach surface."""

import copy
import warnings
import numpy as np
from typing import List
from scipy.optimize import minimize
import cvxpy as cp
from cvxopt import solvers
from typing import Iterable
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.pairwise_union_reach_surface import (
    PairwiseUnionReachSurface,
)


class RestrictedPairwiseUnionReachSurface(PairwiseUnionReachSurface):
    """Models reach with the pairwise union overlap model."""

    def _fit(self) -> None:
        self._reach_vectors = np.array(
            [
                self.get_reach_vector(reach_point.impressions)
                for reach_point in self._data
            ]
        )
        cons = self._get_constraints()
        with warnings.catch_warnings(record=True) as w:
            fit_result = self._fit_with_constraints(cons)
            if len(w) > 1 or (
                len(w) == 1 and not str(w[0].message).startswith("delta_grad == 0.0")
            ):
                raise RuntimeError(
                    "Unexpected warning in RestrictedPairwiseUnionReachSurface: {}".format(
                        ",".join([str(m) for m in w])
                    )
                )
            if not fit_result.success:
                raise RuntimeError(
                    "Optimizer failure in RestrictedPairwiseUnionReachSurface"
                )
        self._construct_a_from_lambda(fit_result["x"])

    def _fit_with_constraints(self, cons):
        return minimize(
            fun=lambda x: self._loss(x),
            x0=np.array([2 / self._p] * self._p) / 2,
            constraints=cons,
            method="trust-constr",
            options={"disp": False},
        )

    def _construct_a_from_lambda(self, lbd: List[float]):
        """Get value of flattened a matrix from lamdas.

        Args:
          lbd: a length p vector indicating lambda_i for each pub.

        Returns:
          the value of flattened a matrix.
        """

        p = len(lbd)
        self._lbd = lbd
        self._a = lbd.reshape(p, 1) * lbd.reshape(1, p) - (np.eye(p) * lbd) ** 2
        self._a = self._a.reshape(p * p, 1)

    def _get_constraints(self):
        """Get constraints to be used in optimization.

        Returns:
          the list of constraint functions
        """

        cons = []
        for i in range(self._p):
            # All lambdas are non negative : lbd[i] >= 0
            cons.append({"type": "ineq", "fun": lambda x: x[i]})
            # Lambda j sum times Lambda i is less than 1 : 1 - lbd[i] * sum(lbd) >= 0
            cons.append({"type": "ineq", "fun": lambda x: 1 - x[i] * sum(x)})
        return cons

    def _evaluate_point(self, lbd: List[float], reach_vector: List[float]):
        """Evaluate reach at a point.

        Args:
          lbd: a length p vector indicating lambda_i for each pub.
          reach_vector: a length p vector indicating the single-pub reach of each
            pub at a single data point.

        Returns:
          the value of union reach at the given point.
        """
        reach_sum = sum(reach_vector)
        overlap = sum(
            [
                (lbd[i] * lbd[j] * reach_vector[i] * reach_vector[j])
                / (
                    max(
                        self._reach_curves[i].max_reach, self._reach_curves[j].max_reach
                    )
                    * 2
                )
                for i in range(self._p - 1)
                for j in range(i, self._p)
            ]
        )
        return reach_sum - overlap

    def _loss(self, lbd: List[float]):
        """Get value of loss function.

        Args:
          lbd: a length p vector indicating lambda_i for each pub.

        Returns:
          the value of fitted union reach.
        """
        val = sum(
            [
                (
                    self._data[k].reach()
                    - self._evaluate_point(lbd, self._reach_vectors[k])
                )
                ** 2
                for k in range(self._n)
            ]
        )
        return val
