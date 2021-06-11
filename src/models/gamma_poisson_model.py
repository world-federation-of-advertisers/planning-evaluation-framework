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
from typing import List
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve

# The following constant defines the highest frequency that will be
# used for parameter estimation.
MAXIMUM_COMPUTATIONAL_FREQUENCY = 1000

# The ratio of the number of impressions observed in the data to the
# size of the total impression inventory.  This ratio is kept constant
# as a way of eliminating a free parameter in the model.
IMPRESSION_INVENTORY_RATIO = 0.1


class GammaPoissonModel(ReachCurve):
    """Gamma-Poisson reach curve model for underreported counts.

    The Gamma-Poisson distribution is a discrete distribution defined on the
    non-negative integers.  The probably of observing a count of k is given
    by a Poisson distribution with parameter lambda, where lambda is drawn
    from a Gamma distribution with shape alpha and rate beta.  Explicitly,
       Pr(X = k) = lambda^k e^{-lambda} / k!, where lambda ~ Gamma(alpha, beta),
    k = 0, 1, 2, ...

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

    where C(n, k) is the binomial coefficient n!/[k!(n-k)!].

    Therefore, the probability that a randomly chosen user will be reached
    exactly k times is

       g(k | alpha, beta, I, I_max) =
            sum_{n=k}^{infinity} g(k, n | alpha, beta, I, I_max).

    Thus, if a total of N viewers can potentially be reached, then the expected
    number of people who will be reached k times is

       hbar[i] = N * g(k | alpha, beta, I, I_max).

    Note that the value I is determined from the input and is not a
    model parameter that needs to be fit.  Thus, the apparent number
    of parameters that need to be fit is four.  However, we reduce
    the number of parameters to two as follows.  First, we note that
    the expected value of a Gamma-Poisson distribution with parameters
    alpha and beta is alpha/beta.  Thus, the total number of
    impressions I_max is related to the total number of potential
    viewers N via the relation

      E[I_max] = N * alpha/beta.

    We therefore set N = I_max * beta / alpha.  Next, we note that
    the size of the total impression inventory I_max is essentially
    a free parameter.  In fact, along the line I_max * beta = constant,
    the objective function is nearly constant.  To see this, observe
    that if the available inventory for each user is doubled, then
    the probability that any given user will receive k impressions
    remains essentially constant.  Thus, we set I / I_max to a
    constant value.  This leaves two remaining parameters to optimize,
    alpha and beta.

    To estimate these parameters we use a histogram matching approach.
    That is to say, let h[i] be the number of people that were reached
    i times, and let hbar[i] be the estimated number of people that
    would be reached i times, given alpha, beta and I_max, N.  The
    objective function that we compute is as follows:

      chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i]

    The perceptive reader might recognize this as the chi-squared
    statistic.  There are a couple advantages to using the chi-squared
    objective function.  First, in practice, the parameter estimates
    that are found often give a nearly exact match to the histograms
    that are observed.  Second, a low value of the objective function
    gives some confidence that a statistically good fit has been
    found.
    """

    def __init__(self, data: List[ReachPoint], max_reach=None):
        """Constructs a Gamma-Poisson model of underreported count data.

        Args:
          data:  A list of consisting of a single ReachPoint to which the model is
            to be fit.
          max_reach:  Optional.  If specified, the maximum possible reach that can
            be achieved.
        """
        if len(data) != 1:
            raise ValueError("Exactly one ReachPoint must be specified")
        self._reach_point = data[0]
        self._max_reach = max_reach
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

    def _impression_count(self, h):
        """Number of impressions recorded in the histogram h."""
        return np.sum([h[i] * (i + 1) for i in range(len(h))])

    def _feasible_point(self, h):
        """Returns values of alpha, beta, Imax that could feasibly produce h.

        We start with alpha=1, because a Gamma-Poisson(1,beta) distribution is
        the same as an Exponential-Poisson(beta), which is the model underlying
        the Goerg one-point reach curve model.  The value of Imax is chosen so
        that I/Imax is the constant INVENTORY_IMPRESSION_RATIO.  The value of
        beta represents a reach point where half of the total available audience
        is reached.
        """
        alpha = 1.0
        Imax = self._impression_count(h) / IMPRESSION_INVENTORY_RATIO
        beta = 2 * np.sum(h) * alpha / Imax
        return alpha, beta, Imax

    def _fit_histogram_chi2_distance(self, h):
        """Chi-squared fit to histogram h.

        Computes parameters alpha, beta, Imax, N such that the histogram hbar of
        of expected frequencies minimizes the metric

          chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i]

        Args:
          h:  np.array specifying the histogram of observed frequencies.
            h[i] is the number of users that were observed to be reached
            exactly i+1 times.
        Returns:
          A tuple (Imax, N, alpha, beta) representing the parameters of the
          best fit that was found.
        """
        # Choose a reasonable starting point
        alpha0, beta0, Imax = self._feasible_point(h)
        Iobs = self._impression_count(h)

        def gamma_obj(params):
            """Objective function for optimization."""
            alpha, beta = np.exp(params)
            N = Imax * beta / alpha
            hbar = self._expected_histogram(Iobs, Imax, N, alpha, beta, len(h))
            obj = np.sum((hbar - h) ** 2 / (hbar + 1e-6))
            return obj

        result = scipy.optimize.minimize(
            gamma_obj, np.log((alpha0, beta0)), method="BFGS"
        )

        alpha, beta = np.exp(result.x)
        N = Imax * beta / alpha

        return Imax, N, alpha, beta

    def _fit_histogram_fixed_N(self, h, N):
        """Chi-squared fit to histogram h for fixed N.

        Computes parameters alpha, beta, Imax such that the histogram hbar of
        of expected frequencies minimizes the metric

          chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i].

        The value of Imax is taken to satisfy I / Imax =
        IMPRESSION_INVENTORY_RATIO, where I is number of impressions
        recorded in the histogram h.  The value of beta is taken to
        satisfy Imax = N * alpha / beta.  Thus, this is really a one
        parameter model, and we can use Brent's method to find the
        optimum.

        Args:
          h:  np.array specifying the histogram of observed frequencies.
            h[i] is the number of users that were observed to be reached
            exactly i+1 times.
          N:  int, total reachable audience.
        Returns:
          A tuple (Imax, alpha, beta) representing the parameters of the
          best fit that was found.

        """
        # Choose a reasonable starting point.
        Iobs = self._impression_count(h)
        Imax = Iobs / IMPRESSION_INVENTORY_RATIO

        def gamma_obj(alpha):
            """Objective function for optimization."""
            beta = N * alpha / Imax
            hbar = self._expected_histogram(Iobs, Imax, N, alpha, beta, len(h))
            obj = np.sum((hbar - h) ** 2 / (hbar + 1e-6))
            return obj

        result = scipy.optimize.minimize_scalar(gamma_obj, bounds=[1e-6, np.Inf])
        alpha = result.x
        beta = N * alpha / Imax

        return Imax, alpha, beta

    def _fit(self) -> None:
        """Fits a model to the data that was provided in the constructor."""

        h = [
            self._reach_point.frequency(i)
            for i in range(1, self._reach_point.max_frequency)
        ]

        if self._max_reach is None:
            Imax, N, alpha, beta = self._fit_histogram_chi2_distance(h)
            self._max_impressions = Imax
            self._max_reach = N
            self._alpha = alpha
            self._beta = beta
        else:
            Imax, alpha, beta = self._fit_histogram_fixed_N(h, self._max_reach)
            self._max_impressions = Imax
            self._alpha = alpha
            self._beta = beta

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
        hist = self._expected_histogram(
            impressions[0],
            self._max_impressions,
            self.max_reach,
            self._alpha,
            self._beta,
            MAXIMUM_COMPUTATIONAL_FREQUENCY,
        )
        kplus_reach = self._kplus_reaches_from_frequencies(hist)
        if self._cpi:
            return ReachPoint(
                impressions, kplus_reach[:max_frequency], [impressions[0] * self._cpi]
            )
        else:
            return ReachPoint(impressions, kplus_reach[:max_frequency])

    def by_spend(self, spends: List[int], max_frequency: int = 1) -> ReachPoint:
        """Returns the estimated reach as a function of spend assuming constant CPM

        Args:
          spend: list of floats of length 1, specifying the hypothetical spend.
          max_frequency: int, specifies the number of frequencies for which reach
            will be reported.
        Returns:
          A ReachPoint specifying the estimated reach for this number of impressions.
        """
        if not self._cpi:
            raise ValueError("Impression cost is not known for this ReachPoint.")
        if len(spends) != 1:
            raise ValueError("Spend vector must have a length of 1.")
        return self.by_impressions([int(spends[0] / self._cpi)], max_frequency)
