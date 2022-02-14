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
"""Dirac mixture single publisher model."""

import numpy as np
import cvxpy as cp
from scipy import stats
from typing import List, Callable
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve


class UnivariateMixedPoissonOptimizer:
    """Fit an univariate mixed Poisson distribution on a frequency histogram.

    At this moment, the fitting algorithm is grid search of components and
    convex optimization on the weights.
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
        self.components = self.weighted_grid_sampling(ncomponents, self.observed_pmf)
        self.fitted = False

    @classmethod
    def validate_frequency_histogram(cls, frequency_histogram: np.ndarray):
        """Check if a frequency histogram is valid.

        A valid frequency histogram has all elements being non-negative, and
        sums up to be positive.
        """
        if len(frequency_histogram) < 2:
            raise ValueError(
                "Please provide a histogram including at least frequencies 0 and 1."
            )
        if not all(frequency_histogram >= 0):
            raise ValueError("Histogram must be non-negative.")
        if sum(frequency_histogram) <= 0:
            raise ValueError("Histogram cannot be zeros.")

    @classmethod
    def weighted_grid_sampling(cls, ncomponents: int, pmf: np.ndarray):
        """Sampling grid according to a pmf.

        Roughly speaking, for each frequency level, sample n_f points equally
        spaced in [f, f + 1], where n_f is proportional to pmf[f].

        Args:
            ncomponents: Number of components to sample.
            pmf: A vector of probabilities that sum up to be 1.
        """
        n_left = ncomponents
        components = np.array([])
        for f in range(len(pmf)):
            n = int(ncomponents * pmf[f])
            components = np.concatenate(
                (components, np.linspace(f, f + 1, n, endpoint=False))
            )
            n_left -= n
        if n_left > 0:
            components = np.concatenate(
                (
                    components,
                    np.linspace(len(pmf), len(pmf) + 1, n_left, endpoint=False),
                )
            )
        return components

    @classmethod
    def truncated_poisson_pmf_vec(
        cls, poisson_mean: float, max_freq: int
    ) -> np.ndarray:
        """pmf vector of a Poisson distribution truncated at a max frequency.

        Args:
            poisson_mean: mean of Poisson distribution.
            max_freq: Any observation beyond this maximum frequency will be
                rounded to this maximum frequency.

        Returns:
            A vector v where v[f] = poisson pmf at f for f < max_freq, and
            v[max_freq] = (probability >= max_freq) = 1 - cdf(max_freq - 1).
        """
        v = stats.poisson.pmf(range(max_freq + 1), poisson_mean)
        v[-1] = 1 - stats.poisson.cdf(max_freq - 1, poisson_mean)
        return v

    @classmethod
    def cross_entropy(cls, observed_arr: np.ndarray, fitted_arr: np.ndarray) -> float:
        """The cross entropy distance between observed and fitted arrays.

        Minimizing cross entropy is equivalent to maximizing likelihood.

        Args:
            observed_arr: A 1d array of the observed pmf. Or 2d array of the
                observed pmfs on different directions.
            fitted_arr: An array of the same dimension as observed_arr.
        Returns:
            - sum_i observed_arr[i] * log(fitted_arr[i]) for 1d arrays,
            - sum_{i, j} observed_arr[i, j] * log(fitted_arr[i, j]) for 2d arrays.
        """
        fitted_arr = fitted_arr + 1e-200
        # Small shift to avoid log-zero numerical error.
        return -cp.sum(cp.multiply(observed_arr, cp.log(fitted_arr)))

    @classmethod
    def solve_optimal_weights(
        cls, observed_arr: np.ndarray, component_arrs: List, distance: Callable
    ) -> np.ndarray:
        """Convex optimization of weights on different components.

        Args:
            observed_arr: Any 1d or 2d arry.
            component_arrs: A list of arrays that are of the same dimension
                of observed_arr.
            distance: A function on two arrays of the same dimension. Calculates
                a certain distance between the two arrays.

        Returns:
            A weight vector so that the weighted sum of arrays in component_arrs
            is closest to observed_arr with respect to the given distance.
        """
        ws = cp.Variable(len(component_arrs))
        pred = cp.sum([cp.multiply(w, arr) for w, arr in zip(ws, component_arrs)])
        problem = cp.Problem(
            objective=cp.Minimize(distance(observed_arr, pred)),
            constraints=[ws >= 0, cp.sum(ws) == 1],
        )
        problem.solve()
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
        """Predict the frequency histogram when scaling number of impressions.

        We fitted sum_j w_j Poisson(lambda_j) on the observed frequency
        histogram.  When scaling the number of impressions by a factor alpha,
        the frequency histogram is predicted as
        sum_j w_j Poisson(alpha * lambda_j).

        Args:
            scaling_factor:  We predict the frequency histogram when the number
                of impressions is scaling_factor times that of the observed
                histogram.
            customized_max_freq:  If specified, the predicted frequency
                histogram will be capped by this maximum frequency.

        Returns:
            A vector v where v[f] is the pmf at f of the total frequency.
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


class DiracMixtureSinglePublisherModel(ReachCurve):
    def __init__(
        self,
        data: List[ReachPoint],
        universe_size: int = None,
        universe_reach_ratio: float = 3,
        ncomponents: int = 200,
    ):
        """Constructs a Dirac mixture single publisher model.

        At this moment, only support the grid fitting method.
        TODO(goal: by EoQ1 2022): Modify and include
            - Evgeny's implementation of adaptive sampling method
            - Gil's implementation using tfp.Gradient and softmax of parameters

        Args:
            data:  A list consisting of the single ReachPoint to which the model
                is to be fit.
            universe_size:  The universe size from which we can compute the
                non-reach from the given ReachPoint and thus obtain a
                zero-included frequency histogram.
            universe_reach_ratio:  Ratio between the universe size and the
                reach of the given ReachPoint.  If we don't know absolute
                universe size but just want the universe to be large enough
                compared to the reach, then specify this argument instead of
                the previous argument.
            ncomponents:  Number of components in the Poisson mixture.  Based on
                our experience, 100 components are enough, and more components
                are also fine (they will not introduce overfitting).
        """
        if len(data) != 1:
            raise ValueError("Exactly one ReachPoint must be specified")
        if data[0].impressions[0] < 0.001:
            raise ValueError("Attempt to create model with 0 impressions")
        if data[0].impressions[0] < data[0].reach(1):
            raise ValueError(
                "Cannot have a model with fewer impressions than reached people"
            )
        self._data = data
        self._reach_point = data[0]
        if data[0].spends:
            self._cpi = data[0].spends[0] / data[0].impressions[0]
        else:
            self._cpi = None
        if universe_size is None:
            self.N = data[0].reach(1) * universe_reach_ratio
        else:
            self.N = universe_size
        self.ncomponents = ncomponents
        self._fit_computed = False

    @classmethod
    def debiased_clip(cls, noised_histogram: np.ndarray) -> np.ndarray:
        """Clip a histogram to be non-negative without introducing much bias.

        The observed, noised histogram may have negative counts.  We can
        round these negative values to zero, but it introduces positive
        biases.  In particular, the 1+ reach is inflated.

        This function mitigates such positive biases. The algorithm is,
        iterating from the maximum frequency to one, whenever we round
        a negative count to zero, we record the positive bias introduced,
        and tries to balance this bias by substracting from a positive count
        in the next frequency levels.  In this way, we can (almost) guarantee
        that zero bias is introduced at least in the 1+ reach.

        Args:
            noised_histogram:  A noised frequency histogram of which some
                elements can be negative.

        Returns:
            A non-negative histogram of which the cumsums are as close to
            those of the given histogram as possiblle.
        """
        cum_bias = 0
        for i in range(len(noised_histogram) - 1, -1, -1):
            if noised_histogram[i] < 0:
                cum_bias -= noised_histogram[i]
                noised_histogram[i] = 0
            elif cum_bias > 0:
                pay_back = min(cum_bias, noised_histogram[i])
                noised_histogram[i] -= pay_back
                cum_bias -= pay_back
        return noised_histogram

    @classmethod
    def obtain_zero_included_histogram(
        cls, universe_size: int, rp: ReachPoint
    ) -> np.ndarray:
        reach = rp.reach(1)
        if reach <= 0:
            return np.array([1.0, 0.0])  # 100% non-reach, 0 reach
        return np.array(
            [universe_size - reach] + rp._frequencies + [rp._kplus_reaches[-1]]
        )

    def _fit(self):
        if self._fit_computed:
            return
        while True:
            hist = self.obtain_zero_included_histogram(self.N, self._reach_point)
            hist = self.debiased_clip(hist)
            while self.ncomponents > 0:
                self.optimizer = UnivariateMixedPoissonOptimizer(
                    frequency_histogram=hist, ncomponents=self.ncomponents
                )
                try:
                    self.optimizer.fit()
                    break
                except:
                    # There is a tiny chance of exception when cvxpy mistakenly
                    # thinks the problem is non-convex due to numerical errors.
                    # If this occurs, it is likely that we have a large number
                    # of components.  In this case, try reducing the number of
                    # components.
                    self.ncomponents = int(self.ncomponents / 2)
                    continue
            if self.optimizer.ws[0] > 0.1:
                # The first weight is that of the zero component.
                # We want the zero component to have significantly positive
                # weight so there's always room for non-reach.
                break
            else:
                self.N *= 2
                continue
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
          A ReachPoint specifying the estimated reach for this number of e.
        """
        if len(impressions) != 1:
            raise ValueError("Impressions vector must have a length of 1.")

        self._fit()
        predicted_relative_freq_hist = self.optimizer.predict(
            scaling_factor=impressions[0] / self._reach_point.impressions[0],
            customized_max_freq=max_frequency,
        )
        # a relative histogram including zero, capped at the maximum frequency of trainineg point
        relative_kplus_reaches_from_zero = np.cumsum(
            predicted_relative_freq_hist[::-1]
        )[::-1]
        kplus_reaches = (
            (self.N * relative_kplus_reaches_from_zero[1:]).round(0).astype("int32")
        )

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
