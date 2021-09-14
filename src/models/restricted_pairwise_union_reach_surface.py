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
from typing import List, Tuple, Callable
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from heapq import heappushpop
import cvxpy as cp
from cvxopt import solvers
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.pairwise_union_reach_surface import (
    PairwiseUnionReachSurface,
)


# Default values of the criteria to control the number of iterations in the
# algorithm, as will be used in RestrictedPairwiseUnionReachSurface._define_criteria
MIN_IMPROVEMENT_PER_ROUND = 1e-3
MAX_NUM_ROUNDS = 1000
DISTANCE_TO_THE_WALL = 0.01
NUM_NEAR_THE_WALL = 10
MIN_NUM_INIT = 30
MAX_NUM_INIT = 1000


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
    can be efficiently fitted using a coordinate descent algorithm.  We can
    iteratively optimize each coordinate of lbd while fixing other coordinates.
    Each iteration can be simply implemented via fitting a simple linear
    regression.

    See the WFA shared doc
    https://docs.google.com/document/d/1zeiCLoKRWO7Cs2cA1fzOkmWd02EJ8C9vgkB2PPCuLvA/edit?usp=sharing
    for the detailed fitting algorithm.  The notations and formulas in the codes
    well correspond to those in the doc.
    """

    def _fit(self, sampler_name: str = "truncated_uniform") -> None:
        """Fitting the restricted pairwise union overlap model."""
        self._define_criteria()
        self._setup_predictor_response()
        init_lbd_sampler = (
            self._truncated_uniform_initial_lbd
            if sampler_name == "truncated_uniform"
            else self._scaled_from_simplex_initial_lbd
        )
        self._fit_multi_init_lbds(init_lbd_sampler)
        self._construct_a_from_lambda()

    def _define_criteria(
        self,
        min_improvement_per_round: float = MIN_IMPROVEMENT_PER_ROUND,
        max_num_rounds: int = MAX_NUM_ROUNDS,
        distance_to_the_wall: float = DISTANCE_TO_THE_WALL,
        num_near_the_wall: int = NUM_NEAR_THE_WALL,
        min_num_init: int = MIN_NUM_INIT,
        max_num_init: int = MAX_NUM_INIT,
    ) -> None:
        """Define criteria that control the number of iterations in the algorithm.

        There are two types of iterations in the algorithm:
        (i)  Multiple rounds of coordinate descent until a local optimum is
             reached.
        (ii) Multiple choices of initial values until we have confidence that
             the best-so-far local optimum is close to the global optimum.
        In our algorithm, type-(i) iterations are terminated if one round of
        iteration fails to improve the fit, i.e., reduce the loss function.

        Type-(ii) iterations are terminated if we "hit a wall at bottom".
        Explicitly, we track the local-opt-losses, i.e., the loss function
        of all local optima.  We keep an eye on the bottom, say, the minimum 10
        among all the local-opt-losses.  If these bottom local-opt-losses are
        far away from each other, then there're still space to search for better
        local optimum.  If, however, the bottom losses become close to each
        other, we make a reasonable guess that these bottom local-opt-losses
        converge to a bound, and that the wall is exactly the global optimum.
        We then approximate the global optimum as the minimum local-opt-loss
        found so far.

        In addition to the termination criteria, we also specify some maximal
        or minimum number of iterations in this method.

        Args:
          min_improvement_per_round:  A threshold for terminating type-(i)
            iterations.  We terminate the iterations if we fail to reduce the
            loss function by this much at a round.
          max_num_rounds:  Force type-(i) to terminate if the number of rounds
            exceeds this number.
          distance_to_the_wall:  Relative distance to the minimum loss so far.
            Starting from each initial value, we find a local optimum.
            For each local optimum, we obtain its value of loss function.  Put
            these values into a list called local_opt_losses.  Any element
            local_opt_losses[i] is considered close enough to the bottom wall if
            local_opt_losses[i] < min(local_opt_losses) *
            (1 + distance_to_the_wall).
          num_near_the_wall:  Number of points close enough to the bottom wall
            when we terminate the search for more local optima.
            At each iteration, count how many points are close enough to the
            bottom wall, i.e., count of i that local_opt_losses[i] <
            min(local_opt_losses) * (1 + distance_to_the_wall).  Terminate if
            the count becomes greater than num_near_the_wall.
          min_num_init:  Force searching for at least these many initial points.
          max_num_init.:  Force type-(ii) iterations to terminate if the number
            of initial points exceeds max_num_init.
        """
        self._min_improvement = min_improvement_per_round
        self._max_num_rounds = max_num_rounds
        self._distance_to_the_wall = distance_to_the_wall
        self._num_near_the_wall = num_near_the_wall
        self._min_num_init = min_num_init
        self._max_num_init = max_num_init

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
                self._ys.append(max(0.0001, sum(r) - rp.reach()))
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

    @classmethod
    def _check_lbd_feasiblity(cls, lbd: List[float], tol=1e-6) -> bool:
        """Check if a choice of lbd falls in the feasible region.

        This is an intermediate method used in self._uniform_initial_lbds.
        """
        for l in lbd:
            if l < 0 or l * (sum(lbd) - l) > 1 + tol:
                return False
        return True

    # The above methods set up the variables to be used in the iterative algorithm.
    # The implementation of iterations starts from here.
    def _step(self, lbd: List[float], i: int) -> float:
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
    def _compute_u(cls, lbd: List[float], i: int, D: np.array) -> float:
        """Compute u following equation (2) in the algorithm description doc.

        This is an intermediate method in self._step.
        """
        res = np.sum(np.outer(lbd, lbd) * D)
        res -= 2 * lbd[i] * np.inner(lbd, D[:, i])
        res -= np.inner(np.square(lbd), np.diag(D))
        return (res + 2 * lbd[i] ** 2 * D[i, i]) / 2

    @classmethod
    def _compute_v(cls, lbd: List[float], i: int, D: np.array) -> float:
        """Compute v following equation (3) in the algorithm description doc.

        This is an intermediate method in self._step.
        """
        return np.inner(lbd, D[:, i]) - lbd[i] * D[i, i]

    @classmethod
    def _get_feasible_bound(cls, lbd: List[float], i: int, tol: float = 1e-6) -> float:
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
        res = 1 / max(total - lbd[i], tol)
        for j in range(len(lbd)):
            if j != i:
                res = min(res, 1 / max(lbd[j], tol) - total + lbd[i] + lbd[j])
        return res

    def _round(self, lbd: List[float]) -> List[float]:
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

    def _loss(self, lbd: List[float]) -> float:
        """Compute the L2 loss of any lbd."""
        loss = 0
        for y, D in zip(self._ys, self._Ds):
            fitted = (
                np.sum(np.outer(lbd, lbd) * D) - np.inner(np.square(lbd), np.diag(D))
            ) / 2
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

    @classmethod
    def _truncated_uniform_initial_lbd(cls, p: int) -> np.array:
        """Sample initial lbd uniformly from the truncated feasible region.

        This is one approach to sample lbd from the feasible region defined by
        lbd[i] >= 0 and lbd[i] * sum(lbd) <= 1 for each i.
        This region has an irregular shape with long tails, which is not easy to
        sample from.  To facilitate sampling, here we force each lbd[i] to be
        <= 1.  This does not truncate too much volume of the exact feasible
        region.  Then we uniformy sample from the truncated region.  Explicitly,
        uniformly sample from the cube {lbd: 0 <= lbd[i] <= 1 for each i} and
        accept the sample that satisfies lbd[i] * sum(lbd) <= 1 for each i.

        Args:
          p:  The length of lbd.

        Return:
          One sample of lbd of the given length.
        """
        while True:
            lbd = np.random.rand(p)
            if cls._check_lbd_feasiblity(lbd):
                return lbd

    @classmethod
    def _scaled_from_simplex_initial_lbd(cls, p: int, tol: float = 1e-6) -> np.array:
        """Sample initial lbd uniformly from the truncated feasible region.

        This is another approach to sample lbd from the feasible region
        lbd[i] >= 0 and lbd[i] * sum(lbd) <= 1 for each i.
        First, uniformly sample a point on the simplex
        {nu: nu[i] >=0 for each i and sum(nu) == 1}.
        The lbd to return is then determined by randomly scaling this point
        within the feasible region.  Explicitly, for any nu on the simplex,
        consider choosing lbd as s * nu.  To have lbd in the feasible region,
        we need to have s >= 0 and s * nu[i] * sum(s * nu) <= 1 for each i,
        i.e., s^2 * max(nu) * sum(nu) <= 1.  Thus, the feasible range of s is
        [0, 1 / sqrt(max(nu) * sum(nu))].  We uniformly choose s from this range
        and return s * nu.

        A uniform sample on the simplex is obtained by uniformly segmenting the
        unit interval.  Explicitly, let u_1, ..., u_{p - 1} be independently
        from Uniform(0, 1).  Let v_1, ..., v_{p - 1} be the ordered statistics
        of u_i.  Then, v_1 - 0, v_2 - v_1, ..., v_{p - 1} - v_{p - 2},
        1 - v_{p - 1} are proved to be a uniform sample on the simplex.

        Args:
          p:  The length of lbd.
          tol:  An artifical threshold to avoid divide-by-zero error.

        Return:
          One sample of lbd of the given length.
        """
        u = sorted(np.random.rand(p - 1))
        nu = np.append(u, 1) - np.append(0, u)
        s_max = 1 / max(np.sqrt(max(nu) * sum(nu)), tol)
        return np.random.rand() * s_max * nu

    def _fit_multi_init_lbds(
        self, init_lbd_sampler: Callable[[int], np.array], random_seed: int = 0
    ) -> None:
        """Fit the model from multiple initial lbds.

        We search local optima from a number of initial lbds, until a cluster of
        locally optimal losses are believed to converge to a lower bound.  In
        this method, the final estimate is chosen as the local optimum that
        achieves the minimum loss function, and saved as self._fitted_lbd.  Its
        value of loss function is saved as self._fitted_loss.  Here we also
        determine if the model has succeeded using a very strict criteria: we
        let self._model_success = 1 if the iterations of both types (i) and (ii)
        successfuly terminate within the maximum number of rounds, and = 0
        otherwise. See the description of self._define_criteria for details.

        Args:
          init_lbd_sampler:  A method to choose initial point.  Can be chosen
            from self._truncated_uniform_initial_lbd or
            self._scaled_from_simplex_initial_lbd.
          random_seed:  The random seed to generate random initial values.
        """
        np.random.seed(random_seed)
        max_heap = [-np.Inf] * self._num_near_the_wall
        # A List following the max heap order to save the negative losses to
        # save the smallest k locally optimum losses.
        # We take negative loss simply because we need a min heap to save the
        # smallest k values but python only (conveniently) supports a max heap.
        num_init = 0
        self._fitted_loss = np.Inf

        def _close_enough(heap: List[float]) -> bool:
            smallest_loss, kth_smallest_loss = -max(heap), -min(heap)
            return kth_smallest_loss < (1 + self._distance_to_the_wall) * smallest_loss

        while (
            num_init < self._min_num_init or _close_enough(max_heap)
        ) and num_init < self._max_num_init:
            init_lbd = init_lbd_sampler(self._p)
            local_fit, local_loss, local_converge = self._fit_one_init_lbd(init_lbd)
            heappushpop(max_heap, -local_loss)
            # update the smallest k locally optimum losses
            if local_loss < self._fitted_loss:
                self._fitted_loss = local_loss
                self._fitted_lbd = local_fit
                self._model_success = local_converge
            num_init += 1
        self._model_success &= _close_enough(max_heap)

    def _construct_a_from_lambda(self) -> None:
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
