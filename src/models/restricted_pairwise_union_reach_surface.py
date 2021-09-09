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
from typing import List, Tuple
from numpy.typing import ArrayLike
from scipy.optimize import minimize
import cvxpy as cp
from cvxopt import solvers
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.pairwise_union_reach_surface import (
    PairwiseUnionReachSurface,
)


NUM_INITIAL_LBDS = 30
MIN_IMPROVEMENT_PER_ROUND = 1e-3
MAX_NUM_ROUNDS = 200


class RestrictedPairwiseUnionReachSurface(PairwiseUnionReachSurface):
    """Predicts union reach using the restricted pairwise union overlap model.

    The restricted pairwise union overlap model is simplified from the pairwise
    union overlap model, with the parameters degenerated from a matrix `a` to a
    vector `lbd` (denoting the greek letter lambda).

    Recall that the pairwise union overlap model assumes
    E(r) = \sum_i r_i - \sum_{i \neq j} a_ij r_i r_j / max(m_i, m_j),
    where E(r) means the expected union reach, each r_i indicates the single
    publisher reach at publisher i, each m_i indicates the maximum reach at
    publisher i, and the sum is taken over all publishers.  The coefficients to
    estimate, a_ij, indicates the interaction between publishers i and j.  These
    coefficients are subject to constraints:
    (i) a_ij >= 0 for each i, j (ii) a_ii = 0 for each i
    (iii) sum_j a_ij <= 1 for each i (iv) sum_i a_ij <= 1 for each j
    to guarantee consistency criteria of the fitted model.

    The restricted pairwise union overlap model inherits the pairwise model form
    and the constraints, and just further assumes that there exist vector lbd
    such that a_ij = lbd_i * lbd_j for each i, j.  In this way, the model
    degrees of freedom is reduced from p^2 - p to p, where p is the number of
    publishers.  As such, the model can be used when the number of training
    points is small, like barely above p.

    While the model degrees of freedom is reduced, the restricted pairwise union
    overlap model becomes non-linear on the coefficients lbd.  It is no longer
    fittable using quadratic programming as we did for the pairwise union
    overlap model.  Nevertheless, the restricted pairwise union overlap model
    can be efficiently fitted using a coordinate descent algorithm.  At each
    step, we can iteratively optimize each coordinate of lbd while fixing other
    coordinates.  Each step can be simply implemented via fitting a simple
    linear regression.

    See the WFA shared doc
    https://docs.google.com/document/d/1zeiCLoKRWO7Cs2cA1fzOkmWd02EJ8C9vgkB2PPCuLvA/edit?usp=sharing
    for the detailed fitting algorithm.  The notations and formulas in the codes
    well correspond to those in the doc.
    """

    def _fit(self) -> None:
        """Fitting the restricted pairwise union overlap model.

        We will choose a number of initial points, and search local optima from
        each initial point using the coordinate descent method.  The final
        estimated is chosen as the local optimum that achieves the minimum loss
        function.
        """
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
        self._construct_a_from_lambda()

    def _define_criteria(
        self,
        min_improvement_per_round: float = MIN_IMPROVEMENT_PER_ROUND,
        max_num_rounds: int = MAX_NUM_ROUNDS,
    ) -> None:
        """Define criteria for terminating the iterations.

        Args:
          min_improvement_per_round:  A threshold of the loss function.  We
            terminate the iterations if we fail to reduce the loss function by
            this much at a round.
          max_num_rounds:  Terminate if the number of rounds exceeds this
            maxmimum number.
        """
        self._min_improvement = min_improvement_per_round
        self._max_num_rounds = max_num_rounds

    def _setup_predictor_response(self) -> None:
        """Re-formulate the predictors and response of the model.

        The model has input being the per-pub reach (r_1, ..., r_p) and output
        being the union reach r.  Following the algorithm description doc, it is
        conveninent to treat matrix D with
        d_ij = r_i r_j / [2 max(m_i, m_j)]
        as the predictors, and
        y = r - \sum_i r_i
        as the response.  This method computes D and y for each training point
        and put them in the lists `self._Ds` and `self._ys`.
        """
        self._ys = []
        # Will have self._y[l] = sum(single pubs reaches) - union reach in the
        # l-th training point
        self._Ds = []
        # Will have self._Ds[l] [i, j] = d_ij^(l), i.e., d_ij at the l-th
        # training point
        m = [rc.max_reach for rc in self._reach_curves]
        # m[i] is the maximum reach at publisher i
        for rp in self._data:
            if (
                rp.impressions.count(0) < self._p - 1
            ):  # Exclude single-pub training point
                r = self.get_reach_vector(rp.impressions)
                self._ys.append(max(0.0001, sum(r) - rp._kplus_reaches[0]))
                self._Ds.append(self._construct_D_from_r_and_m(r, m))

    @classmethod
    def _construct_D_from_r_and_m(cls, r: ArrayLike, m: ArrayLike):
        """An intermediate method in self._setup_predictor_response.

        Compute matrix D following equation (1) in the algorithm description
        doc.

        Args:
          r:  A vector where r[i] is the single publisher reach at publisher i.
          m:  A vector where m[i] is the maximum reach at publisher i.

        Returns:
          The matrix D, i.e., what we reformulate as the predictor of the model.
        """
        mat_m = np.array(
            [[max(m[i], m[j]) for i in range(len(m))] for j in range(len(m))]
        )
        # mat_m[i, j] = max(max reach of pub i, max reach of pub j)
        return np.outer(r, r) / 2 / mat_m

    def _uniform_initial_lbds(self, num_init: int = NUM_INITIAL_LBDS) -> None:
        """Sample initial values of lbd almost uniformly from the feasible region.

        To search for global optimum, we conduct coordinate descent starting
        from a number of randomly distributed initial values, instead of just
        one initial value.  To choose the initial values of lbd, it is
        natural to uniformly sample from its feasible region.

        The exact feasible region of lbd is defined by the inequalities
        lbd[i] >= 0 and
        lbd[i] * sum(lbd) <= 1
        for each i.  This region has an irregular shape with long tails, which
        is not easy to sample from.  To facilitate sampling, we further force
        each lbd[i] be to <= 1.  This does truncate too much volume of the
        exact feasible region.  Then we uniformy sample from the truncated
        region.  Explicitly, uniformly sample from the cube
        {lbd: 0 <= lbd[i] <= 1 for each i} and accept the samples that satisfies
        lbd[i] * sum(lbd) <= 1 for each i.

        Args:
          num_init:  The number of initial values to be chosen.
        """
        self._init_lbds = []
        while len(self._init_lbds) < num_init:
            lbd = np.random.rand(self._p)
            if self._check_lbd_feasiblity(lbd):
                self._init_lbds.append(lbd)

    @classmethod
    def _check_lbd_feasiblity(cls, lbd: List[float], tol=1e-6):
        """Check if a choice of lbd falls in the feasible region.

        This is an intermediate method used in self._uniform_initial_lbds.
        """
        for l in lbd:
            if l < 0 or l * (sum(lbd) - l) > 1 + tol:
                return False
        return True

    # The above methods set up the variables to be used in the iterative algorithm.
    # The implementation of iterations starts from here.
    def _step(self, lbd: List[float], i: int):
        """Each step to update one coordinate of lbd.

        Args:
          lbd:  The current best guess of lbd.
          i:  The index of the coordinate to be updated at this step.

        Returns:
          The updated best guess of lbd[i].
        """
        us = np.array([self._compute_u(lbd, i, D) for D in self._Ds])
        vs = np.array([self._compute_v(lbd, i, D) for D in self._Ds])
        lbd_i_hat = np.inner(np.array(self._ys) - us, vs) / np.inner(vs, vs)
        # The above line follows equation (4) in the algorithm description doc.
        if lbd_i_hat < 0:
            return 0
        bound = self._get_feasible_bound(lbd, i)
        if lbd_i_hat > bound:
            return bound
        return lbd_i_hat

    @classmethod
    def _compute_u(cls, lbd: List[float], i: int, D: np.array):
        """Compute u following equation (2) in the algorithm description doc.

        This is an intermediate method in self._step.
        """
        u = 0
        for j in range(len(lbd)):
            for k in range(j + 1, len(lbd)):
                if j != i and k != i:
                    u += lbd[j] * lbd[k] * D[j, k]
        return u

    @classmethod
    def _compute_v(cls, lbd: List[float], i: int, D: np.array):
        """Compute v following equation (3) in the algorithm description doc.

        This is an intermediate method in self._step.
        """
        v = 0
        for j in range(len(lbd)):
            if j != i:
                v += lbd[j] * D[i, j]
        return v

    @classmethod
    def _get_feasible_bound(cls, lbd: List[float], i: int, tol: float = 1e-6):
        """Compute B(lbd_{-i}) of equation (5) in the algorithm description doc.

        B(lbd_{-i}) is the upper bound of lbd[i] so that lbd falls in the
        feasibele region.  This is an intermediate method in self._step.

        Args:
          lbd:  The current best guess of lbd.
          i:  The index of the coordinate to update
          tol:  An artifical threshold to avoid divide-by-zero error.

        Returns:
          A bound B satisfying the following property.  Suppose we change lbd[i]
          to a number 'l_new' while keeping the other coordinates unchanged.
          Then lbd is feasible (i.e., cls._check_lbd_feasiblity == True) if and
          only if 0 <= l_new <= B.
        """
        total = sum(lbd)
        res = 1 / max(total - lbd[i], 1e-6)
        for j in range(len(lbd)):
            if j != i:
                res = min(res, 1 / max(lbd[j], tol) - total + lbd[i] + lbd[j])
        return res

    def _round(self, lbd: List[float]):
        """Each round to update the whole vector lbd.

        A round consists of self._p steps.

        Args:
          lbd:  The current best guess of lbd.

        Returns:
          The updated best guess of lbd after a round.
        """
        for i in np.random.permutation(self._p):
            # We shuffle the order of coordinates to have symmetry across
            # different EDPs.
            lbd[i] = self._step(lbd, i)
        return lbd

    def _loss(self, lbd: List[float]):
        """Compute the L2 loss of any lbd."""
        loss = 0
        for y, D in zip(self._ys, self._Ds):
            fitted = 0
            for i in range(self._p):
                for j in range(i + 1, self._p):
                    fitted += lbd[i] * lbd[j] * D[i, j]
            loss += (y - fitted) ** 2
        return loss

    def _fit_one_init_lbd(self, lbd: List[float]) -> Tuple[List[float], float, bool]:
        """The complete updating procedure from one initial point of lbd.

        It conducts a number of rounds until we fail to reduce the loss function
        by self._min_improvement at a round.  That is, a local optimum is
        achieved at this round.

        Args:
          lbd:  Initial value of lbd.

        Returns:
          A tuple of (lbd_opt, loss, converge).
          - lbd_opt means the obtained lbd at the last round.
          - loss means the loss function of lbd_opt.
          - converge is a boolean indicating if a local optimum is indeed
            achieved.  Empirically, we find that local optima can beachieved in
            only a few rounds, and this is intuitively true in view of the
            geometric interpretation of coordinate descent.  But, a theoretical
            support is absent.  As such, we record if the reduction of loss is
            indeed smaller than the threshold (self._min_improvement) at the
            last round.  If not, it means that the local optimum is still not
            found after many rounds, i.e., the algorithm fails to converge when
            starting from the given initial point.
        """
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

    def _construct_a_from_lambda(self):
        """Obtain matrix `a` which will be used for model prediction.

        The matrix `a` in the parent class `PairwiseUnionReachSurface` is
        degenerated to a function of the vector of `lbd` in the child class
        `RestrictedPairwiseUnionReachSurface`.  This method converts the
        `self._fitted_lbd` in the child class to the `self._a` in the parent
        class, so as to inherit methods like `by_impressions` in the parent
        class for model prediction.
        """
        a = np.outer(self._fitted_lbd, self._fitted_lbd)
        a -= np.diag(np.diag(a))
        self._a = a.flatten()
