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
"""Gamma-Poisson reach curve model for underreported counts."""

import numpy as np
import scipy.stats

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve

# The following constant defines the highest frequency that will be
# used for parameter estimation.
MAXIMUM_COMPUTATIONAL_FREQUENCY = 1000

# The maximum number of basin-hopping iterations that will be performed
# during optimization.
MAXIMUM_BASIN_HOPS = 20


class GammaPoissonModel(ReachCurve):
    """Gamma-Poisson reach curve model for underreported counts.

    The Gamma-Poisson distribution is a discrete distribution defined on the
    non-negative integers.  The probably of observing a count of k is given
    by a Poisson distribution with parameter lambda, where lambda is drawn
    from a Gamma distribution with shape alpha and rate beta.  E.g.,
       Pr(X = k) = lambda^k e^{-lambda) / k!, where lambda ~ Gamma(alpha, beta).

    The Gamma-Poisson distribution is often used for modeling overdispersed
    data, which is common in the advertising industry.  In this setting, the
    random variable X represents the total number of impressions that could
    potentially be shown to a randomly chosen user.  This quantity is called
    the available inventory for X.

    In the underreported Gamma-Poisson model, we are given a set of
    observed impression counts, one value for each viewer that is
    reached.  Let these values be k_1,k_2,...,k_R.  Thus, R is the
    number of people that were reached, and the number of impressions
    that shown is I = k_1+k_2+...+k_R.  The unknown parameters that
    need to be estimated are:

      * the parameters alpha and beta of the associated Gamma distribution,
      * the total size I_max of the impression inventory, and
      * the total number N of viewers that could potentially be reached.

    This class implements code for estimating these parameters and for
    performing forecasts with those estimates.

    Let f(n | alpha, beta) be the probability that the size of the available
    inventory of a user is n.  Then, the probability that a randomly chosen
    user will have inventory n and be reached k times is given by

       g(k, n | alpha, beta, I, I_max) =
            C(n, k) (I / I_max)^k (1 - I/I_max)^{n-k} f(n | alpha, beta),

    where C(n, k) is the binomial coefficient n!/k!(n-k)!.

    Therefore, the probability that a randomly chosen user will be reached
    exactly k times is

       g(k | alpha, beta, I, I_max) =
            sum_{n=k}^{infinity} g(k, n | alpha, beta, I, I_max).

    Thus, if a total of N viewers can potentially be reached, then the expected
    number of people who will be reached k times is

       N * g(k | alpha, beta, I, I_max)

    In the Gamma-Poisson distribution, the expected size of the total
    impression inventory is

       E[I_max] = N * alpha / beta.

    To eliminate one parameter from consideration, we therefore assume
    that N = I_max * beta / alpha.  This leaves us with three parameters
    to estimate: alpha, beta and I_max.  To estimate these parameters
    we use a histogram matching approach.  That is to say, let h[i] be
    the number of people that were reached i times, and let hbar[i] be
    the estimated number of people that would be reached i times, given
    alpha, beta and I_max.  The objective function that we compute is
    as follows:

      chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i] +
           lambda * (alpha + beta + I_max / I)

    The first part of this objective function is the chi-squared statistic
    computed between the histograms h and hbar, while the second part of
    this objective function is a regularization term.

    There are a couple advantages to using the chi-squared objective
    function.  First, in practice, the parameter estimates that are found
    often give a nearly exact match to the histograms that are observed.
    Second, a low value of the objective function gives some confidence
    that a statistically good fit has been found.

    In a few informal experiments, the estimates of alpha and N were generally
    found to be fairly accurate.  However, a high amount of variance was
    observed in the estimates of beta and I_max.  The objective function is
    nearly constant along the line beta * I_max = constant.  This is why the
    regularization term was added.
    """

    def __init__(
        self,
        data: [ReachPoint],
        max_reach=None,
        regularization_parameter=1.0,
    ):
        """Constructs a Gamma-Poisson model of underreported count data.

        Args:
          data:  A list of ReachPoints to which the model is to be fit.
          max_reach:  Optional.  If specified, the maximum possible reach that can
            be achieved.
          regularization_parameter:  Optional.  A float specifying a regularization
            penalty that will be applied to parameter estimates.
        """
        if len(data) != 1:
            raise ValueError("Exactly one ReachPoint must be specified")
        self._reach_point = data[0]
        self._max_reach = max_reach
        self._regularization_parameter = regularization_parameter
        if data[0].spends:
            self._cpi = data[0].spends[0] / data[0].impressions[0]
        else:
            self._cpi = None

    def _logpmf(self, n, alpha, beta):
        """Log of the PMF of the Gamma-Poisson distribution.

        This implementation makes use of the equivalence between the
        Gamma-Poisson distribution with parameters (alpha, beta) and
        the negative binomial distribution with parameters (p, r) =
        (beta / (1 + beta), alpha).

        Args:
          n:  Value(s) at which the distribution is to be evaluated.
            This can be a scalar or a numpy array.
          alpha: Alpha parameter of the Gamma-Poisson distribution.
          beta: Beta parameter of the Gamma-Poisson distribution.
        Returns:
          f(n | alpha, beta), where f is the Gamma-Poisson distribution.
        """
        return scipy.stats.nbinom.logpmf(n, alpha, beta / (1.0 + beta))

    def _knreach(self, k, n, I, Imax, alpha, beta):
        """Probability that a random user has n impressions of which k are shown.

        Computes the function:

            C(n, k) (I / I_max)^k (1 - I/I_max)^{n-k} f(n | alpha, beta).

        where C(n, k) is the binomial coefficient n!/k!(n-k)!.

        Args:
          k:  Scalar number of impressions that were seen by the user.
          n:  Available inventory for the user.
            This can be either a scalar or a numpy array.
          I:  Total number of impressions shown to all users.
          Imax: Total size of impression inventory.
          alpha: Parameter of Gamma-Poisson distribution.
          beta: Parameter of Gamma-Poisson distribution.
        Returns:
          Probability that a randomly chosen user will have an inventory
          of n impressions, of which k are shown.
        """
        kprob = scipy.stats.binom.logpmf(k, n, I / Imax)
        return np.exp(kprob + self._logpmf(n, alpha, beta))

    def _kreach(self, k, I, Imax, alpha, beta):
        """Probability that a random user receives k impressions.

        Computes the function:
            sum_{n=k}^{infinity} g(k, n | alpha, beta, I, I_max).
        where g(k, n | alpha, beta, I, I_max) is the probability that
        a randomly chosen user will have an available inventory of n
        impressions, of which k are shown.

        Args:
          k:  np.array specifying number of impressions that were seen by the user.
          I:  Total number of impressions shown to all users.
          Imax: Total size of impression inventory.
          alpha: Parameter of Gamma-Poisson distribution.
          beta: Parameter of Gamma-Poisson distribution.
        Returns:
          For each k, probability that a randomly chosen user will have an inventory
          of n impressions, of which k are shown.
        """
        return np.array(
            [
                np.sum(
                    self._knreach(
                        kv,
                        np.arange(kv, MAXIMUM_COMPUTATIONAL_FREQUENCY + 1),
                        I,
                        Imax,
                        alpha,
                        beta,
                    )
                )
                for kv in k
            ]
        )

    def _expected_impressions(self, N, alpha, beta):
        """Estimates the expected size of impression inventory for N users.

        Makes use of the identity that the expected value of a negative binomial
        distribution with parameters p and r is pr/(1-p).

        Args:
          N:      total number of people that can potentially be reached.
          alpha:  shape parameter of Gamma distribution.
          beta:   rate parameter of Gamma distribution.
        Returns:
          Expected size of impression inventory for a population of N users.
        """
        return N * alpha / beta

    def _expected_histogram(self, I, Imax, N, alpha, beta, max_freq=10):
        """Computes an estimated histogram.

        Args:
          I:      number of impressions that were purchased.
          Imax:   total size of impression inventory.
          N:      total number of people that can potentially be reached.
          alpha:  shape parameter of Gamma distribution.
          beta:   rate parameter of Gamma distribution.
          max_freq: maximum frequency to report in the output histogram.
        Returns:
          An np.array h where h[i] is the expected number of users who will see i+1
          impressions.
        """
        return N * self._kreach(np.arange(1, max_freq + 1), I, Imax, alpha, beta)

    def _feasible_point(self, h):
        """Returns values of alpha, beta, Imax, N that could feasibly produce h."""
        alpha = 1.0
        N = 2 * np.sum(h)
        I = 2 * np.sum([h[i] * (i + 1) for i in range(len(h))])
        beta = N * alpha / I
        return alpha, beta, I, N

    def _fit_histogram_chi2_distance(self, h, regularization_param=0.01):
        """Chi-squared fit to histogram h.

        Computes parameters alpha, beta, Imax, N such that the histogram hbar of
        of expected frequencies minimizes the metric

          chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i] +
             lambda * (alpha + beta + I_max / I)

        where lambda is a regularization constant and I is the total number of
        impressions that were observed in the data.

        Optimization is performed using the basin hopping algorithm of Wales
        and Doye.  Each optimization step is performed using the BFGS optimizer.

        Args:
          h:  np.array specifying the histogram of observed frequencies.
            h[i] is the number of users that were observed to be reached
            exactly i+1 times.
          regularization_param:  float, the constant lambda that is used in
            the regularization term in the above metric.
        Returns:
          A tuple (Imax, N, alpha, beta) representing the parameters of the
          best fit that was found.
        """
        # The estimated impression count cannot go below the count observed
        # in the data
        Imin = sum([h[i] * (i + 1) for i in range(len(h))]) + 1

        # Choose a reasonable starting point
        alpha0, beta0, I0, N0 = self._feasible_point(h)

        def gamma_obj(params):
            """Objective function for optimization."""
            alpha, beta, Imax = np.exp(params)
            Imax += Imin
            N = Imax * beta / alpha
            hbar = self._expected_histogram(Imin, Imax, N, alpha, beta, len(h))
            obj = np.sum((hbar - h) ** 2 / (hbar + 1e-6)) + regularization_param * (
                alpha + beta + Imax / Imin
            )
            return obj

        result = scipy.optimize.basinhopping(
            gamma_obj,
            np.log((alpha0, beta0, I0 - Imin)),
            niter=MAXIMUM_BASIN_HOPS,
            minimizer_kwargs={"method": "BFGS", "tol": 0.1},
        )

        alpha, beta, I = np.exp(result.x)
        I += Imin
        N = I * beta / alpha

        return I, N, alpha, beta

    def _fit_histogram_fixed_N(self, h, N, regularization_param=0.01):
        """Chi-squared fit to histogram h for fixed N.

        Computes parameters alpha, beta, Imax such that the histogram hbar of
        of expected frequencies minimizes the metric

          chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i] +
             lambda * (alpha + beta + I_max / I)

        where lambda is a regularization constant and I is the total number of
        impressions that were observed in the data.  The value of Imax is
        taken to be Imax = N * beta / alpha, so this really a two parameter
        model.

        Optimization is performed using the basin hopping algorithm of Wales
        and Doye.  Each optimization step is performed using the BFGS optimizer.

        Args:
          h:  np.array specifying the histogram of observed frequencies.
            h[i] is the number of users that were observed to be reached
            exactly i+1 times.
          N:  int, total reachable audience.
          regularization_param:  float, the constant lambda that is used in
            the regularization term in the above metric.
        Returns:
          A tuple (Imax, alpha, beta) representing the parameters of the
          best fit that was found.
        """
        # The estimated impression count cannot go below the count observed
        # in the data
        Imin = sum([h[i] * (i + 1) for i in range(len(h))]) + 1

        # Choose a reasonable starting point.
        alpha0, beta0 = 1.0, 0.5 * N / Imin

        def gamma_obj(params):
            """Objective function for optimization."""
            alpha, beta = np.exp(params)
            Imax = self._expected_impressions(N, alpha, beta)
            if Imax <= Imin:
                return 1e99
            hbar = self._expected_histogram(Imin, Imax, N, alpha, beta, len(h))
            obj = np.sum((hbar - h) ** 2 / (hbar + 1e-6)) + regularization_param * (
                alpha + beta + 1 / beta
            )
            return obj

        result = scipy.optimize.basinhopping(
            gamma_obj,
            np.log((alpha0, beta0)),
            niter=MAXIMUM_BASIN_HOPS,
            minimizer_kwargs={"method": "Nelder-Mead", "tol": 0.1},
        )

        alpha, beta = np.exp(result.x)
        I = self._expected_impressions(N, alpha, beta)

        return I, alpha, beta

    def _fit(self) -> None:
        """Fits a model to the data that was provided in the constructor."""

        h = [
            self._reach_point.frequency(i)
            for i in range(1, self._reach_point.max_frequency)
        ]

        if self._max_reach is None:
            (
                self._max_impressions,
                self._max_reach,
                self._alpha,
                self._beta,
            ) = self._fit_histogram_chi2_distance(
                h, regularization_param=self._regularization_parameter
            )
        else:
            (
                self._max_impressions,
                self._alpha,
                self._beta,
            ) = self._fit_histogram_fixed_N(
                h, self._max_reach, regularization_param=self._regularization_parameter
            )

    def by_impressions(self, impressions: [int], max_frequency: int = 1) -> ReachPoint:
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
        hist = self._expected_histogram(
            impressions[0],
            self._max_impressions,
            self.max_reach,
            self._alpha,
            self._beta,
            MAXIMUM_COMPUTATIONAL_FREQUENCY,
        )
        kplus_reach = list(reversed(np.cumsum(list(reversed(hist)))))
        if self._cpi:
            return ReachPoint(
                impressions, kplus_reach[:max_frequency], [impressions[0] * self._cpi]
            )
        else:
            return ReachPoint(impressions, kplus_reach[:max_frequency])

    def by_spend(self, spends: [int], max_frequency: int = 1) -> ReachPoint:
        """Returns the estimated reach as a function of spend assuming constant CPM

        Args:
          spend: list of floats of length 1, specifying the hypothetical spend.
          max_frequency: int, specifies the number of frequencies for which reach
            will be reported.
        Returns:
          A ReachPoint specifying the estimated reach for this number of impressions.
        """
        if len(spends) != 1:
            raise ValueError("Spend vector must have a length of 1.")
        return self.by_impressions([self.impressions_for_spend(spends[0])],
                                   max_frequency)

    def impressions_for_spend(self, spend: float) -> int:
        if not self._cpi:
            raise ValueError("Impression cost is not known for this ReachPoint.")
        return spend / self._cpi
        
