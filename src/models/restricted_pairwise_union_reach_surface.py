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
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.pairwise_union_reach_surface import (
    PairwiseUnionReachSurface,
)


NUM_INITIAL_LBDS = 30

# description doc: https://docs.google.com/document/d/1FINz2Vi4u5CjpnEroAYqFTh8dnbj3IIcD42Xd1cLqEE/edit
#TODO(jiayu): clean up description doc and upload to WFA


class RestrictedPairwiseUnionReachSurface(PairwiseUnionReachSurface):
    """Models reach with the pairwise union overlap model."""

    def _fit(self) -> None:
        self._define_criteria()
        self._setup_predictor_response()
        self._uniform_initial_lbds()
        self._fitted_loss = self._loss(self._init_lbds[0])
        for lbd in self._init_lbds:
            fitted, loss, converge = self._fit_one_init_lbd(lbd)
            if loss < self._fitted_loss:
                self._fitted_lbd = fitted
                self._fitted_loss = loss
                self._model_success = converge
        self._construct_a_from_lambda(self._fitted_lbd)

    def _construct_a_from_lambda(self, lbd: List[float]):
        """Get value of flattened a matrix from lamdas.
        Args:
          lbd: a length p vector indicating lambda_i for each pub.
        Returns:
          the value of flattened a matrix.
        """
        a = np.outer(lbd, lbd)
        a -= np.diag(np.diag(a))
        self._a = a.flatten()

    def _uniform_initial_lbds(self, num_init=NUM_INITIAL_LBDS) -> None:
        self._init_lbds = []
        while len(self._init_lbds) < num_init:
            lbd = np.random.rand(self._p)
            if self._check_lbd_feasiblity(lbd):
                self._init_lbds.append(lbd)

    @classmethod
    def _check_lbd_feasiblity(cls, lbd, tol=1e-6):
        for l in lbd:
            if l < 0 or l * (sum(lbd) - l) > 1 + tol:
                return False
        return True

    def _define_criteria(self) -> None:
        self._min_improvement = 1e-3  # threshold on the loss function for termination
        self._max_num_rounds = 1000

    def _setup_predictor_response(self) -> None:
        rs = []
        # rs[l] = single pub reach vector in the l-th training point, i.e.,
        # rs[l] [i] = reach at pub i in the l-th training point.
        self._ys = []
        # self._y[l] = sum(single pubs reaches) - union reach in the l-th training point
        for rp in self._data:
            if (
                rp.impressions.count(0) < self._p - 1
            ):  # Exclude single-pub training point
                r = self.get_reach_vector(rp.impressions)
                self._ys.append(max(0.0001, sum(r) - rp._kplus_reaches[0]))
                rs.append(r)
        mat_m = np.array(
            [
                [
                    max(
                        self._reach_curves[i].max_reach, self._reach_curves[j].max_reach
                    )
                    for i in range(self._p)
                ]
                for j in range(self._p)
            ]
        )
        # mat_m[i, j] = max(pub i max reach, pub j max reach)
        self._ds = []  # ds[l] [i, j] = d_ij^(l) in the description doc
        for r in rs:
            self._ds.append(np.outer(r, r) / 2 / mat_m)

    @classmethod
    def _get_feasible_bound(cls, lbd, i, tol=1e-6):
        total = sum(lbd)
        res = 1 / max(total - lbd[i], 1e-6)
        for j in range(len(lbd)):
            if j != i:
                res = min(res, 1 / max(lbd[j], tol) - total + lbd[i] + lbd[j])
        return res

    def _step(self, lbd, i):
        us = np.array([self._compute_u(lbd, i, D) for D in self._ds])
        vs = np.array([self._compute_v(lbd, i, D) for D in self._ds])
        lbd_i_hat = np.inner(np.array(self._ys) - us, vs) / np.inner(vs, vs)
        if lbd_i_hat < 0:
            return 0
        bound = self._get_feasible_bound(lbd, i)
        if lbd_i_hat > bound:
            return bound
        return lbd_i_hat

    def _round(self, lbd):
        for i in np.random.permutation(self._p):
            lbd[i] = self._step(lbd, i)
        return lbd

    def _loss(self, lbd):
        """Compute L2 loss literally following formula ? in the description doc."""
        loss = 0
        for y, D in zip(self._ys, self._ds):
            fitted = 0
            for i in range(self._p):
                for j in range(i + 1, self._p):
                    fitted += lbd[i] * lbd[j] * D[i, j]
            loss += (y - fitted) ** 2
        return loss

    def _fit_one_init_lbd(self, lbd):
        prev_loss = self._loss(lbd)
        cur_loss = prev_loss - 1
        num_rounds = 0
        while (
            num_rounds < self._max_num_rounds
            and cur_loss < prev_loss - self._min_improvement
        ):
            lbd = self._round(lbd)
            prev_loss = cur_loss
            cur_loss = self._loss(lbd)
            num_rounds += 1
        return lbd, cur_loss, (cur_loss >= prev_loss - self._min_improvement)

    def _compute_u(self, lbd, i, D):
        """Compute u literally following formula X in the description doc."""
        u = 0
        for j in range(self._p):
            for k in range(j + 1, self._p):
                if j != i and k != i:
                    u += lbd[j] * lbd[k] * D[j, k]
        return u

    def _compute_v(self, lbd, i, D):
        """Compute v literally following formula Y in the description doc."""
        v = 0
        for j in range(self._p):
            if j != i:
                v += lbd[j] * D[i, j]
        return v
