# Copyright 2022 The Private Cardinality Estimation Framework Authors
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
"""Dirac mixture single publisher model."""

import numpy as np
import cvxpy as cp
from scipy import stats
from typing import List, Callable
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve


class UnivariateMixedPoissonOptimizer:
    """Fit a univariate mixed Poisson distribution on a frequency histogram.

    A mixed Poisson distribution has probablity mass function (pmf)
    f(x) = sum_{j=1}^J w_j pi(x | lambda_j), x = 0, 1, 2, ...
    where any
    pi(x | lambda) = lambda^x * exp(-lambda) / x!
    is the pmf of the Poisson distribution, and w_j are subject to the
    constraints that all w_j >= 0 and sum_{j=1}^J w_j = 1.
    Each lambda_j is called a component. w_j is called the weight of component
    lambda_j.

    At this moment, the fitting algorithm consists of a grid search of components
    and a convex optimization on the weights of different components.
    """

    def __init__(
        self,
        frequency_histogram: np.ndarray,
        ncomponents: int = 200,
    ):
        """Construct an optimizer for univariate mixed Poisson distribution.

        Args:
            frequency_histogram:  An array v where v[f] = number of observations
                with frequency f, for 0 <= f <= F - 1, and v[F] = number of
                observations with frequency >= F, for a max frequency F.
                Note that the histogram starts from frequency 0.
            ncomponents: Number of components in the Poisson mixture.  Based on
                our experience, 100 components are enough, and more components
                are also fine (they will not introduce overfitting).
        """
        self.validate_frequency_histogram(frequency_histogram)
        self.observed_pmf = frequency_histogram / sum(frequency_histogram)
        # Work with standardized histogram, i.e., pmf to avoid potential overflow.
        self.max_freq = len(frequency_histogram) - 1
        self.components = self.in_bound_purely_weighted_grid(
            ncomponents, self.observed_pmf
        )
        self.fitted = False

    @staticmethod
    def validate_frequency_histogram(frequency_histogram: np.ndarray):
        """Check if a frequency histogram is valid.

        A valid frequency histogram has all elements being non-negative, sums up
        to be positive, and has length at least 2.
        """
        if len(frequency_histogram) < 2:
            raise ValueError(
                "Please provide a histogram including at least frequencies 0 and 1."
            )
        if not all(frequency_histogram >= 0):
            raise ValueError("Histogram must be non-negative.")
        if sum(frequency_histogram) <= 0:
            raise ValueError("Histogram cannot be zeros.")

    @staticmethod
    def in_bound_grid(
        ncomponents: int, pmf: np.ndarray, dilusion: float = 0
    ) -> np.ndarray:
        """Samples components from [0, max_freq] weighted by the observed pmf to an extent.

        Main idea: A Poisson distribution has high pmf around its mean parameter.
        So, if we observe a high pmf at a frequency level, we tend to draw more
        Poisson components around it.

        For example, if we observe a pmf of [0.5, 0.3, 0, 0.2] and want to draw
        10 components, we may draw 5 components in [0, 1), 3 components in [1, 2),
        0 components in [2, 3), and 2 components in [3, 4).  The components are
        equally spaced within each interval of length 1.  We call it "purely
        weighted" method of choosing components.  See the
        `in_bound_purely_weighted_grid` method below.

        The purely weighted method efficiently captures the areas that are most
        likely to have components, but ignores the areas where the components
        are unfortunately unobserved due to randomness/noise.  In the above toy
        example, we do not sampling components between [2, 3), while there may
        actually be one.  As such, our sampling weights can be chosen between
        the observed pmf and the uniform pmf, in other words, we "dilute" the
        sampling weights to cover unobserved areas.  See the explanation of the
        `dilusion` argument below.

        Note that this method is "in-bound": we do not draw any components beyond
        max_freq.  So, it does not have a great fit in outlier cases where the
        true average frequency is close to or even higher than the observable
        max_freq.  We are thinking about methods to cover the beyond-bound
        components and will submit another PR.

        Args:
            ncomponents: Number of components to sample.
            pmf: A vector of probabilities that sum up to be 1.
            dilusion:  The sampling weights are chosen as (1 - dilusion) *
                observed_pmf + dilusion * uniform_pmf.  No significant impact of
                this parameter has been seen in initial simulations, but it is
                worth further investigation.

        Returns:
            All components in the model.
        """
        if dilusion > 0:
            water = np.array([dilusion / len(pmf)] * len(pmf))
            pmf = pmf * (1 - dilusion) + water
        n_left = ncomponents
        components = np.array([])
        f = 0
        while n_left > 0 and f < len(pmf):
            n = int(round(ncomponents * pmf[f]))
            components = np.concatenate(
                (components, np.linspace(f, f + 1, n, endpoint=False))
            )
            n_left -= n
            f += 1
        if n_left > 0:
            components = np.concatenate(
                (
                    components,
                    np.linspace(f, f + 1, n_left, endpoint=False),
                )
            )
        return components

    @classmethod
    def in_bound_purely_weighted_grid(
        cls, ncomponents: int, pmf: np.ndarray
    ) -> np.ndarray:
        """Samples components from [0, max_freq] purely weighted by the observed pmf.

        See the docstrings of the `in_bound_grid` method for more details.
        """
        return cls.in_bound_grid(ncomponents, pmf, dilusion=0)

    @classmethod
    def in_bound_uniform_grid(cls, ncomponents: int, pmf: np.ndarray) -> np.ndarray:
        """Samples components uniformly from [0, max_freq].

        See the docstrings of the `in_bound_grid` method for more details.
        """
        return cls.in_bound_grid(ncomponents, pmf, dilusion=1)

    @staticmethod
    def truncated_poisson_pmf_vec(poisson_mean: float, max_freq: int) -> np.ndarray:
        """pmf vector of a Poisson distribution truncated at a max frequency.

        Args:
            poisson_mean: mean of Poisson distribution.
            max_freq: Any observation beyond this maximum frequency will be
                rounded to this maximum frequency.

        Returns:
            A vector v where v[f] = Poisson_pmf at f for f < max_freq, and
            v[max_freq] = (probability >= max_freq)
            = 1 - Poisson_cdf(max_freq - 1).
        """
        v = stats.poisson.pmf(range(max_freq + 1), poisson_mean)
        v[-1] = stats.poisson.sf(max_freq - 1, poisson_mean)
        return v

    @staticmethod
    def cross_entropy(observed_arr: np.ndarray, fitted_arr: np.ndarray) -> float:
        """The cross entropy distance of fitted_arr relative to observed_arr.

        Minimizing cross entropy is equivalent to maximizing the likelihood.

        Args:
            observed_arr: A 1d array of the observed pmf. Or 2d array of the
                observed pmfs on different directions.
            fitted_arr: Any other array of pmf(s) that has the same dimension
                as observed_arr.
        Returns:
             The cross entropy of the fitted pmf(s) relative to the observed
             pmf(s), that is,
            - sum_i observed_arr[i] * log(fitted_arr[i]) for 1d arrays,
            - sum_{i, j} observed_arr[i, j] * log(fitted_arr[i, j]) for 2d arrays.
            If the input fitted_arr is a cvxpy expression, then the return is
            also a cvxpy expression -- use .value to extract its value.
        """
        # Small shift to avoid log-zero numerical error.
        fitted_arr = fitted_arr + 1e-200
        return -cp.sum(cp.multiply(observed_arr, cp.log(fitted_arr)))

    @staticmethod
    def relative_entropy(observed_arr: np.ndarray, fitted_arr: np.ndarray) -> float:
        """The relative entropy distance from fitted_arr relative to observed_arr.

        Args:
            observed_arr: A 1d array of the observed pmf. Or 2d array of the
                observed pmfs on different directions.
            fitted_arr: Any other array of pmf(s) that has the same dimension
                as observed_arr.
        Returns:
            - sum_i observed_arr[i] * log(fitted_arr[i] / observed_arr[i])
                for 1d arrays,
            - sum_{i, j} observed_arr[i, j] * log(fitted_arr[i, j] / observed_arr[i, j])
                for 2d arrays.
            If the input fitted_arr is a cvxpy expression, then the return is
            also a cvxpy expression -- use .value to extract its value.
        """
        return cp.sum(cp.rel_entr(observed_arr, fitted_arr))

    @staticmethod
    def solve_optimal_weights(
        observed_arr: np.ndarray, component_arrs: List, distance: Callable
    ) -> np.ndarray:
        """Convex optimization of weights on different components.

        Args:
            observed_arr: Any 1d or 2d arry.
            component_arrs: A list of arrays that are of the same dimension
                of observed_arr.
            distance: A function on two arrays of the same dimension. Calculates
                a certain distance between the two arrays.

        Returns:
            A weight vector such that the weighted sum of the arrays in
            component_arrs is closest to observed_arr with respect to the given
            distance.
        """
        ws = cp.Variable(len(component_arrs))
        pred = cp.sum([cp.multiply(w, arr) for w, arr in zip(ws, component_arrs)])
        problem = cp.Problem(
            objective=cp.Minimize(distance(observed_arr, pred)),
            constraints=[ws >= 0, cp.sum(ws) == 1],
        )
        # cvxpy is an interface of multiple convex optimization solver packages.
        # Among them, the open source packages are ECOS
        # (https://github.com/embotech/ecos/wiki), SCS
        # (https://www.cvxgrp.org/scs/api/index.html) and OSQP, where ECOS and
        # SCS are able to handle our problem. So, we try ECOS first and then SCS.
        # Note: ECOS is available in C.  SCS available in C/C++.  But they are
        # both unavailable in Java.
        try:
            problem.solve(solver=cp.ECOS)
        except cp.SolverError:
            problem.solve(solver=cp.SCS)
        return ws.value

    def fit(self):
        """Fits a univariate mixed Poisson distribution."""
        if self.fitted:
            return
        self.component_pmfs = [
            self.truncated_poisson_pmf_vec(component, self.max_freq)
            for component in self.components
        ]
        self.ws = self.solve_optimal_weights(
            self.observed_pmf,
            self.component_pmfs,
            self.cross_entropy,
        )
        self.fitted = True

    def predict(
        self, scaling_factor: float, customized_max_freq: int = None
    ) -> np.ndarray:
        """Predict the frequency histogram when scaling the number of impressions.

        The Dirac mixture model assumes that when scaling the number of
        impressions, different Poisson components are scaled accordingly,
        and that the pmf is shifted to the weighted sum of these scaled
        Poisson distributions with the same weights.

        Explicitly, if for the actual campaign its frequency distribution
        is fitted as
        sum_j w_j Poisson(lambda_j),
        Then for a hypothetical campaign that has alpha times the number
        of impressions of the actual campaign, its frequency distribution
        is predicted as
        sum_j w_j Poisson(alpha * lambda_j).

        Args:
            scaling_factor:  We predict the frequency histogram when the number
                of impressions is scaling_factor times that of the observed
                histogram.
            customized_max_freq:  If specified, the predicted frequency
                histogram will be capped by this maximum frequency.

        Returns:
            A vector v where v[f] is the pmf at f of the frequency.
        """
        if not self.fitted:
            self.fit()
        scaled_component_pmfs = [
            self.truncated_poisson_pmf_vec(
                poisson_mean=c * scaling_factor,
                max_freq=(
                    self.max_freq
                    if customized_max_freq is None
                    else customized_max_freq
                ),
            )
            for c in self.components
        ]
        return sum([w * pmf for w, pmf in zip(self.ws, scaled_component_pmfs)])