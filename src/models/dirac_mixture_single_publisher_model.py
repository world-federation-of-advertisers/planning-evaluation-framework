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
"""Dirac mixture single publisher model.

The model and fitting algorithm are described in this companion doc:
https://drive.google.com/file/d/1P6GYeYXjtB-j8M_WQokJrIEWFlEjYrmy/view?resourcekey=0-iUFvPbzIU7AUy-myx3NxVw
"""

from absl import logging
import numpy as np
import cvxpy as cp
from typing import List, Tuple
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve


class MixedPoissonOptimizer:
    """Fit an univaraite mixed Poisson distribution on a frequency histogram."""

    def __init__(self, frequency_histogram: np.ndarray):
        """Construct an optimizer for univarate mixed Poisson distribution.

        Please read the companion doc
        https://drive.google.com/file/d/1P6GYeYXjtB-j8M_WQokJrIEWFlEjYrmy/view?resourcekey=0-iUFvPbzIU7AUy-myx3NxVw
        for notations and formulas.

        Args:
            frequency_histogram:  An array v where v[f] = number of observations with frequency
                f, for 0 <= f <= F - 1, and v[F] = number of observations with frequency >= F,
                for a max frequency F. Note that a multiplier on v does not affect the result,
                which means that the input can also be the relative frequency histogram.
        """
        self.vec_A = frequency_histogram
        self.max_freq = len(frequency_histogram) - 1
        self._get_log_factorials()

    def fit(self, method: str = "grid"):
        """Fixed a mixed Poisson distribution.

        Args:
            method:  Method for fitting the mixed Poisson distribution.  Current options are
                'grid' and 'adaptive'.
        """
        if method == "grid":
            try:
                self.grid_fit()
            except:
                logging.vlog(
                    2,
                    "Grid algorithm fails due to numerical errors. Using adaptive algorithm instead.",
                )
                self.adaptive_fit()
            # cvxpy might (with very small chance) run into numerical errors.  For example, it
            # may find a problem non-convex when the problem is in fact convex without numerical
            # errors.  The grid algorithm, as a one-step method, would fail in this case.
            # The adapative algorithm, on the other hand, stops when running into numerical errors
            # and makes the best use of the current results.  In addition, it iterates over multiple
            # initial values, so still finds a reasonable solution even if numerical errors occur at
            # some runs.  We switch to the adapative algorithm in case the grid algorithm fails.
        elif method == "adaptive":
            self.adaptive_fit()
        else:
            raise ValueError(f"method {method} is not yet supported.")

    def _get_log_factorials(self):
        """Save sum_{i=0}^f log(i!) for f from 0 to self.max_freq, as these quantities will be repeatly used."""
        self._log_factorials = [0]
        for f in range(1, self.max_freq):
            self._log_factorials.append(self._log_factorials[f - 1] + np.log(f))

    def _get_vec_pi(self, lbd: float, customized_max_freq: int = None) -> np.ndarray:
        """Get the probability mass function (pmf) vector for each Poisson component.

        Args:
            lbd:  A float >= 0 indicating the Poisson mean.
            customized_max_freq:  If specified, it will be used as the upper bound of the pmf vector.

        Returns:
            pmf vector from 0 to F for Poisson(lbd).  Equations (2) and (3) in the companion doc.
        """
        max_freq = self.max_freq if customized_max_freq is None else customized_max_freq
        if lbd < 1e-4:
            return np.array([1] + [0] * max_freq)
        vec_pi = np.zeros(max_freq + 1)
        for f in range(max_freq):
            vec_pi[f] = np.exp(f * np.log(lbd) - lbd - self._log_factorials[f])
        vec_pi[max_freq] = 1 - sum(vec_pi)
        return vec_pi

    def _mat_pi(
        self, lbds: List, customized_max_freq: int = None
    ) -> cp.atoms.affine.hstack.Hstack:
        """A matrix vstacking the pmf vectors for different Poisson components.  Equatipon (4) in the companion doc."""
        return cp.bmat([self._get_vec_pi(lbd, customized_max_freq) for lbd in lbds]).T

    def _log_likelihood(
        self, vec_w: np.ndarray, mat_pi: cp.atoms.affine.hstack.Hstack
    ) -> cp.atoms.affine.binary_operators.MulExpression:
        """log-likelihood of a mixed Poisson distribution.  Equatipon (5) in the companion doc."""
        return self.vec_A @ cp.log(mat_pi @ vec_w)
        # @ means matrix-vector or matrix-matrix multiplication in cvxpy

    def _gradient_towards_new_component(
        self,
        new_pmf_vec: np.ndarray,
        existing_weights: np.ndarray,
        existing_mat_pi: cp.atoms.affine.hstack.Hstack,
    ):
        """Gradient of log-likelihood towards a new component.  Equatipon (6) in the companion doc.

        Args:
            new_pmf_vec:  pmf vector of any new component.
            existing_mat_pi:  matrix of log-pmf vectors of all the existing components.
            existing weights:  weight vector of all the existing compoents.

        Returns:
            Gradient when starting to proportionally shift weight from existing components to new component.
        """
        return self.vec_A @ (new_pmf_vec / (existing_mat_pi @ existing_weights) - 1)

    def _gradient_towards_new_poisson(
        self,
        new_lbd: float,
        existing_weights: np.ndarray,
        existing_mat_pi: cp.atoms.affine.hstack.Hstack,
    ) -> cp.atoms.affine.binary_operators.MulExpression:
        """Gradient of log-likelihood towards a new, single Poisson component with mean new_lbd."""
        return self._gradient_towards_new_component(
            self._get_vec_pi(new_lbd), existing_weights, existing_mat_pi
        )

    def _solve_optimal_new_poisson(
        self,
        existing_weights: np.ndarray,
        existing_mat_pi: cp.atoms.affine.hstack.Hstack,
        grid_size_under_max_freq: int = 30,
        max_scalar_on_max_freq: float = 10,
        grid_size_beyond_max_freq: int = 10,
    ) -> Tuple[float, float]:
        """Grid search of the new, single Poisson component that maximizes the gradient of log-likelihood.

        Args:
            new_pmf_vec:  pmf vector of any new component.
            existing_mat_pi:  matrix of log-pmf vectors of all the existing components.
            existing weights:  weight vector of all the existing compoents.
            grid_size_under_max_freq:  we search these many Poisson lbds equally spaced in in [0, max_freq].
            max_scalar_on_max_freq:  A scalar beta > 1 for which we also search lbd in
                [max_freq, beta * max_freq].
            grid_size_beyond_max_freq:  We search these many Poisson lbds in [max_freq, beta * max_freq].
                Typically, we do a fine-grid search in [0, max_freq] and a coarse search beyond max_freq.

        Returns:
            Gradient when starting to proportionally shift weight from existing components to new component.
        """
        grid = [
            self.max_freq * x / grid_size_under_max_freq
            for x in range(grid_size_under_max_freq)
        ] + [
            self.max_freq
            + self.max_freq
            * (max_scalar_on_max_freq - 1)
            * x
            / grid_size_beyond_max_freq
            for x in range(grid_size_beyond_max_freq)
        ]
        vals = [
            self._gradient_towards_new_poisson(
                new_lbd, existing_weights, existing_mat_pi
            ).value
            for new_lbd in grid
        ]
        return grid[np.argmax(vals)], max(vals)

    def _solve_optimal_weights(
        self, mat_pi: cp.atoms.affine.hstack.Hstack
    ) -> Tuple[float, float]:
        """Convex optimization of the weight vector given a number of components."""
        J = mat_pi.shape[1]
        w = cp.Variable(J)
        objective = cp.Maximize(self._log_likelihood(w, mat_pi))
        constraints = [sum(w) == 1]
        for j in range(J):
            constraints.append(w[j] >= 0)
        optimal_objective = cp.Problem(objective, constraints).solve()
        return (w.value, optimal_objective)

    def _adaptive_fit_one_initial_lbd(
        self, initial_lbd: float
    ) -> Tuple[np.ndarray, List, float]:
        """Fit the model starting from an initial component.

        Args:
            initial_lbd: an arbitrary (location of) component in [0, self.max_freq].

        Returns:
            A tuple (vec_w, vec_lbd, objective), where
            - vec_w and vec_lbd specify a fitted mixed Poisson model sum_j w_j Poisson(lbd_j)
            - objective indicates the log-likelihood of the final fit
            Note that some intermediate results like the trajectory of objective are also
            obtained, but not returned. One can print them when doing detailed tests.
        """
        lbds = [initial_lbd]
        mat_pi = self._mat_pi(lbds)
        weights_trace = [np.ones(1)]
        objective_trace = [-self._log_likelihood(np.ones(1), mat_pi).value]
        gradient_trace = []

        for step in range(self.max_freq):
            # Find up to max_freq components. This is a heuristic choice. Based on initial experiments, any number
            # of components > max_freq / 3 works well.
            try:
                # Occasionally, due to numerical errors, cvxpy will run into bugs like
                # `RuntimeWarning: divide by zero encountered in log return np.log(values[0])`.
                # So, adding a try-except here.
                # This is not a significant issues since the exception occurs very occasionally.
                # And even if the exception occurs so that we exit the loop, the result obtained
                # in the previous step is still useful.
                res = self._solve_optimal_new_poisson(weights_trace[-1], mat_pi)
                lbds.append(res[0])
                gradient_trace.append(res[1])
                mat_pi = cp.hstack((mat_pi, self._mat_pi([res[0]])))

                res = self._solve_optimal_weights(mat_pi)
                weights_trace.append(res[0])
                objective_trace.append(res[1])
            except:
                logging.vlog(2, f"numerical error in cvxpy at step {step}")
                break
        return (weights_trace[-1], lbds, objective_trace[-1])

    def adaptive_fit(self, num_initial_lbd: int = 30):
        """Fit the model from a grid of initial components.

        Args:
            num_initial_lbd:  we fit the model from these many initial lbds and pick the best result.
        """
        results = [
            self._adaptive_fit_one_initial_lbd(lbd)
            for lbd in np.linspace(0.1, self.max_freq, num_initial_lbd)
            # For simplicity, initial lbd is chosen from a grid in (0, max_freq].
            # The grid starts from 0.1 (instead of 0) to avoid divide-by-zero errors.
        ]
        objectives = [res[2] for res in results]
        self.weights, self.lbds = results[np.argmax(objectives)][:2]

    def grid_fit(
        self,
        grid_size_under_max_freq: int = 30,
        max_scalar_on_max_freq: float = 10,
        grid_size_beyond_max_freq: int = 10,
    ):
        """Fit the model simply by solving the optimal weights on Poissons with a fixed grid of lbds.

        Args:
            grid_size_under_max_freq:  we have these many Poisson lbds equally spaced in [0, max_freq].
            max_scalar_on_max_freq:  A scalar beta > 1 for which we also have lbds in
                [max_freq, beta * max_freq].
            grid_size_beyond_max_freq:  We have these many Poisson lbds in [max_freq, beta * max_freq].
                Typically, we have a fine-grid in [0, max_freq] and a coarse grid beyond max_freq.

        Returns:
            Gradient when starting to proportionally shift weight from existing components to new component.
        """
        self.lbds = [
            self.max_freq * x / grid_size_under_max_freq
            for x in range(grid_size_under_max_freq)
        ] + [
            self.max_freq
            + self.max_freq
            * (max_scalar_on_max_freq - 1)
            * x
            / grid_size_beyond_max_freq
            for x in range(grid_size_beyond_max_freq)
        ]
        self.weights = self._solve_optimal_weights(self._mat_pi(self.lbds))[0]

    def predict(
        self, scaling_factor: float, customized_max_freq: int = None
    ) -> np.ndarray:
        """Predict the frequency histogram of mixed Poisson when scaling the location of each component.

        We fitted sum_j w_j Poisson(lambda_j) on the observed frequency histogram.  This method computes the
        frequency histogram, i.e., pmf of sum_j w_j Poisson(alpha * lambda_j) for any alpha >= 0.

        Args:
            scaling_factor: alpha in the above description.
            customized_max_freq:  If specified, it will be used as the upper bound of the pmf vector.

        Returns:
            A vector v where v[f] is the pmf at f of the distribution sum_j w_j Poisson(alpha * lambda_j).
        """
        scaled_lbds = [lbd * scaling_factor for lbd in self.lbds]
        return (self._mat_pi(scaled_lbds, customized_max_freq) @ self.weights).value


class DiracMixtureSinglePublisherModel(ReachCurve):
    def __init__(
        self,
        data: List[ReachPoint],
        method: str = "grid",
        universe_size: int = None,
        universe_reach_ratio: float = 3,
    ):
        """Constructs a Dirac mixture sinle publisher model.

        Please read the companion doc
        https://drive.google.com/corp/drive/folders/1AzrHFAgMn6f_GMj9kpw_5yoOeXy3GjdH
        for notations and formulas.

        Args:
            data:  A list consisting of the single ReachPoint to which the model is to be fit.
            method:  Method for fitting the mixed Poisson distribution.  Current options are
                'grid' and 'adaptive'.
            universe_size:  The universe size N so that a zero-included frequency histogram is
                obtained using equation (1) in the companion doc.
            universe_reach_ratio:  If the previous argument is not specified, then obtain the
                universe size as universe_reach_ratio * <1+ reach at the training point>.
        """
        if len(data) != 1:
            raise ValueError("Exactly one ReachPoint must be specified")
        if data[0].impressions[0] < 0.001:
            raise ValueError("Attempt to create model with 0 impressions")
        if data[0].impressions[0] < data[0].reach(1):
            raise ValueError(
                "Cannot have a model with fewer impressions than reached people"
            )
        self._reach_point = data[0]
        if data[0].spends:
            self._cpi = data[0].spends[0] / data[0].impressions[0]
        else:
            self._cpi = None
        if universe_size is None:
            self.N = data[0].reach(1) * universe_reach_ratio
        else:
            self.N = universe_size
        self._fit_computed = False
        self.method = method

    def _fit(self):
        if self._fit_computed:
            return
        hist = np.array(
            [self.N - self._reach_point.reach(1)]
            + self._reach_point._frequencies
            + [self._reach_point._kplus_reaches[-1]]
        )
        hist = np.maximum(0, hist)
        self.mpo = MixedPoissonOptimizer(hist / sum(hist))
        self.mpo.fit(self.method)
        self._fit_computed = True

    def by_impressions(
        self, impressions: List[int], max_frequency: int = 1
    ) -> ReachPoint:
        """Returns the estimated reach as a function of impressions.

        Args:
          impressions: list of ints of length 1, specifying the hypothetical number
            of impressions that are shown.
          max_frequency: int, specifies the number of frequencies for which reach
            will be reported.

        Returns:
          A ReachPoint specifying the estimated reach for this number of impressions.
        """
        if len(impressions) != 1:
            raise ValueError("Impressions vector must have a length of 1.")

        self._fit()
        predicted_relative_freq_hist = self.mpo.predict(
            impressions[0] / self._reach_point.impressions[0], max_frequency
        )
        # a relative histogram including zero, capped at the maximum frequency of training point
        relative_kplus_reaches_from_zero = np.cumsum(
            predicted_relative_freq_hist[::-1]
        )[::-1]
        kplus_reaches = self.N * relative_kplus_reaches_from_zero[1:]

        if self._cpi:
            return ReachPoint(impressions, kplus_reaches, [impressions[0] * self._cpi])
        else:
            return ReachPoint(impressions, kplus_reaches)

    def by_spend(self, spends: List[int], max_frequency: int = 1) -> ReachPoint:
        """Returns the estimated reach as a function of spend assuming constant CPM.

        Args:
          spend: list of floats of length 1, specifying the hypothetical spend.
          max_frequency: int, specifies the number of frequencies for which reach
            will be reported.
        Returns:
          A ReachPoint specifying the estimated reach for this number of impressions.
        """
        if len(spends) != 1:
            raise ValueError("Spend vector must have a length of 1.")
        return self.by_impressions(
            [self.impressions_for_spend(spends[0])], max_frequency
        )

    def impressions_for_spend(self, spend: float) -> int:
        if not self._cpi:
            raise ValueError("Impression cost is not known for this ReachPoint.")
        return spend / self._cpi
