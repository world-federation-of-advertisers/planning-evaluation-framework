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

from absl import logging
import numpy as np
import scipy.stats
from typing import List
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve

# The following constant defines the highest frequency that will be
# used for parameter estimation.
MAXIMUM_COMPUTATIONAL_FREQUENCY = 1000

class KInflatedGammaPoissonDistribution:
    """k-inflated Gamma-Poisson distribution with specified parameters.

    The k-inflated Gamma-Poisson distribution is like the Gamma-Poisson
    distribution except that the first k values of the PMF are allowed to
    be arbitrary.

    It is given by k+2 parameters: alpha, beta, a_1, a_2, ..., a_k.
    We assume that each a_i >= 0, and that a_1 + a_2 + ... + a_k < 1.
    
    Let GP(n | alpha, beta) be the PMF of the shifted Gamma-Poisson distribution 
    with parameters alpha, beta.  Also, define s = a_1 + a_2 + ... + a_k,
    and define t = \sum_{i=1}^k GP(i | alpha, beta).  Then, the PMF of the
    k-inflated Gamma-Poisson distribution is given as

     kGP(n | alpha, beta, a_1, ..., a_k) 
          = a_n if 0 < n <= k, 
            (1-s) / (1-t) GP(n | alpha, beta) if n > k.
    """
    def __init__(self, alpha, beta, a):
        """k-Inflated Gamma-Poisson with parameters alpha, beta, N, a. """
        if sum(a) > 1.0:
            raise ValueError("Inflation values sum to a value greater than 1.")
        
        self._alpha = alpha
        self._beta = beta
        self._a = a
        
        n = len(a)
        s = np.sum(a)
        t = np.sum(self._gamma_poisson_pmf(np.arange(1, n+1), alpha, beta))
        u = self._gamma_poisson_pmf(np.arange(n+1,MAXIMUM_COMPUTATIONAL_FREQUENCY), alpha, beta)
        pmf_tail = ((1 - s) / (1 - t)) * u
        # self._pmf[n] = kGP(n | alpha, beta, a_1, ..., a_k)
        self._pmf = np.array([0.] + list(a) + list(pmf_tail))

    def _gamma_poisson_pmf(self, n, alpha, beta):
        """PMF of the shifted Gamma-Poisson distribution.

        This implementation makes use of the equivalence between the
        Gamma-Poisson distribution with parameters (alpha, beta) and
        the negative binomial distribution with parameters (p, r) =
        (1 / (1 + beta), alpha).

        Args:
          n:  Value(s) at which the distribution is to be evaluated.
            This can be a scalar or a numpy array.
          alpha: Alpha parameter of the Gamma-Poisson distribution.
          beta: Beta parameter of the Gamma-Poisson distribution.
        Returns:
          f(n | alpha, beta), where f is the Gamma-Poisson distribution.
        """
        # Note scipy.stats.nbinom uses a different parameterization than
        # the Wikipedia parameterization.  Under this parameterization,
        # p -> 1-p, so the parameter 1/(1 + beta) becomes beta/(1 + beta).
        return scipy.stats.nbinom.pmf(n - 1, alpha, 1.0 / (1.0 + beta))

    def pmf(self, n):
        """PMF of the k-inflated Gamma-Poisson distribution.

        Args:
          n:  Value(s) at which the distribution is to be evaluated.
            This can be a scalar or a numpy array.
        Returns:
          kGP(n | alpha, beta, a_1, ..., a_k), where f is the k-inflated 
          Gamma-Poisson distribution.
        """
        return self._pmf[n]
       
    def knreach(self, k, n, p):
        """Probability that a random user has n impressions of which k are shown.

        Computes the function:

            C(n, k) p^k (1 - p)^{n-k} kGP(n | alpha, beta, a).

        where C(n, k) is the binomial coefficient n!/[k!(n-k)!].

        Args:
          k:  Number of impressions that were seen by the user.
          n:  Available inventory for the user.
            This can be either a scalar or a numpy array.
          p:  Probability that a given impression will be chosen.
        Returns:
          Probability that a randomly chosen user will have an inventory
          of n impressions, of which k are shown.
        """
        return scipy.stats.binom.pmf(k, n, p) * self.pmf(n)

    def kreach(self, k, p):
        """Probability that a random user receives k impressions.

        Computes the function:
            sum_{n=k}^{infinity} g(k, n | alpha, beta, I, I_max).
        where g(k, n | alpha, beta, I, I_max) is the probability that
        a randomly chosen user will have an available inventory of n
        impressions, of which k are shown.

        Args:
          k:  np.array specifying number of impressions that were seen by the user.
          p:  Probability that a given impression will be chosen.
        Returns:
          For each k, probability that a randomly chosen user will have an inventory
          of n impressions, of which k are shown.
        """
        return np.array(
            [
                np.sum([self.knreach(kv, np.arange(kv, len(self._pmf)), p)])
                for kv in k
            ]
        )

    def kplusreach(self, k, p):
        """Probability that a random user receives k or more impressions.

        Computed as one minus the probability that the user receives k-1 or
        fewer impressions.

        Args:
          k:  np.array specifying number of impressions that were seen by the user.
          p:  Probability that a given impression will be chosen.
        Returns:
          For each k, probability that a randomly chosen user will have an inventory
          of n impressions, of which k are shown.
        """
        if k == 0:
            return 1.0
        else:
            return 1.0 - np.sum(self.kreach(np.arange(k), p))
            
    def expected_value(self):
        """Expected value of this distribution."""
        return np.sum(np.arange(len(self._pmf)) * self._pmf)


class KInflatedGammaPoissonModel(ReachCurve):
    """k-inflated Gamma-Poisson reach curve model for underreported counts.

    The k-inflated Gamma-Poisson distribution is like the Gamma-Poisson
    distribution except that the first k values of the PMF are allowed to
    be arbitrary.

    It is given by k+3 parameters: alpha, beta, N, a_1, a_2, ..., a_k.
    We assume that each a_i >= 0, and that a_1 + a_2 + ... + a_k < 1.
    
    Let GP(n | alpha, beta) be the PMF of the shifted Gamma-Poisson distribution 
    with parameters alpha, beta.  Also, define s = a_1 + a_2 + ... + a_k,
    and define t = \sum_{i=1}^k GP(i | alpha, beta).  Then, the PMF of the
    k-inflated Gamma-Poisson distribution is given as

     kGP(n | alpha, beta, a_1, ..., a_k) 
          = a_n if n <= k, 
            (1-s) / (1-t) GP(n | alpha, beta) if n > k.

    As with the Gamma-Poisson distribution, we use the chi2 objective
    function for assessing goodness of fit.  That is to say, let h[i]
    be the number of people that were reached i times, and let hbar[i]
    be the estimated number of people that would be reached i times,
    given alpha, beta and N.  The objective function that we
    compute is as follows:

      chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i]

    One question is what value of k should be chosen.  We propose
    to consider successive values of k starting from 0, and to stop
    when a fit is found such that the value of the objective function 
    is smaller than the value of the chi^2 statistic with f_max degrees
    of freedom, where f_max is the highest frequency that is recorded.
    """

    def __init__(self, data: List[ReachPoint], kmax=10):
        """Constructs a Gamma-Poisson model of underreported count data.

        Args:
          data:  A list of consisting of a single ReachPoint to which the model is
            to be fit.
          kmax:  Maximum number of values of the PMF that are allowed to be set
            arbitrarily.
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
        self._kmax = kmax
        self._fit_computed = False
        if data[0].spends:
            self._cpi = data[0].spends[0] / data[0].impressions[0]
        else:
            self._cpi = None

    def _chi2_distance(
            self, h, I, N, dist
    ):
        """Returns distance between actual and expected histograms.

        Computes the metric

            chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i]

        where
            hbar is the histogram that would be expected given parameters
               I, N, dist

        The first part of this objective formula is the chi-squared statistic
        used for determining whether two sets of categorical samples are drawn
        from the same underlying distribution.  Therefore, if the value returned
        by this function is relatively low (say < 20), then there is reason to
        believe that the Gamma-Poisson distribution adequately fits the data.

        Args:
          h: Actual frequency histogram.  h[i] is the number of people
             reached exactly i+1 times, except for h[-1] which is the
             number of people reached len(h) or more times.
          I: Total number of impressions purchased.
          N: Total number of unique users.
          dist:   A KInflatedGammaPoisson distribution.
        Returns:
          chi^2 distance between the actual and expected histograms, plus
          a term for weighting the difference in 1+ reach.
        """
        # Estimate total number of potential impressions
        Imax = N * dist.expected_value()
        if Imax <= I:
            return np.sum(np.array(h) ** 2)

        freqs = N * dist.kreach(np.arange(1, len(h)), I/Imax)
        kplus = N * dist.kplusreach(len(h), I/Imax)
        hbar = np.array(list(freqs) + [kplus])
            
        obj = np.sum((hbar - h) ** 2 / (hbar + 1e-6))

        if np.isnan(obj):
            logging.vlog(2, f"alpha {alpha} beta {beta} N {N} Imax {Imax}")
            logging.vlog(2, f"h    {h}")
            logging.vlog(2, f"hbar {hbar}")
            raise RuntimeError("Invalid value of objective function")

        return obj

    def _exponential_poisson_reach(self, I, N, beta):
        """Estimates reach in an exponential poisson model.

        Estimates reach when I impressions are purchased from a population
        of N users where the impression count of each user is distributed
        according to the exponential poisson distribution with parameter beta.

        Under the exponential poisson model, the probability that a
        user will have n impressions is (1 - p) * p^{n-1}, where p =
        beta/(beta + 1).  This can be worked out by using the
        equivalence between the gamma poisson and the negative
        binomial and setting alpha = r = 1.

        The probability that a user is not reached, given that a fraction q
        of available impressions is purchased, is given by
           sum_{n=1}^{infinity} (1 - q)^n f(n),
        where f(n) is the PMF of the above exponential poisson distribution.
        This simplifies to (1 - p)(1 - q) / (1 - p(1 - q)).

        The probability that a user is reached at least once is one minus
        the above quantity.  Finally, the value of q is taken to be
        I / ((beta + 1) * N), as the average number of impressions that a
        user will have is beta + 1.

        Args:
          I:  The hypothetical number of impressions purchased.
          N:  The audience size.
          beta: The parameter of the exponential-poisson distribution.
        Returns:
          The expected reach.
        """
        return N * I * (beta + 1) / (N + beta * (N + I))

    def _exponential_poisson_beta(self, I, N, R):
        """Estimate beta given impressions, audience size and reach."""
        return (N * (I - R))/(I * R - N * (I - R))

    def _exponential_poisson_N_from_beta(self, I, R, beta):
        """Estimate audience size from impressions, reach and beta."""
        return -(I * beta * R)/((beta + 1) * (R - I))

    def _exponential_poisson_N(self, I, R, mu_scale_factor=1.1):
        """Estimate audience size from I and R.

        We know that the mean mu of the distribution has to be greater than
        I / R.  Since beta = mu - 1, this says that beta >= I / R - 1.
        But how much bigger?  That cannot easily be determined from just one 
        point.  Here, we assume beta = s * (I / R - 1), where s is some
        small scale factor.  From that, we estimate N.
        """
        return R * (mu_scale_factor * I - R) / (mu_scale_factor * (I - R))
        
    def _fit_exponential_poisson_model(self, point):
        """Returns N, alpha, beta of an exponential-poisson model.

        The fit returned by this function is guaranteed to match the 1+ reach
        of the input point.  If additional frequencies are given, there should
        be a reasonable match at these as well, although the match is not 
        guaranteed to be perfect.

        The parameters returned by this function are used to bootstrap the
        full k-Inflated Gamma Poisson model.

        Args:
          point:  A ReachPoint to which an exponential-poisson model is 
            to be fit.
        Returns:
          A pair (N, dist), where N is the estimated audience size and dist
          is an estimated KInflatedGammaPoissonDistribution representing an
          exponential-poisson distribution.
        """
        impressions = point.impressions[0]
        reach = point.reach()
        if len(point.frequencies) > 1:
            # Estimate the mean from the distribution found in the reach point
            s = np.sum(point.frequencies)
            mu = np.sum([(i + 1) * f / s for i, f in enumerate(point.frequencies)])
            beta = (mu - 1) * 1.1    # 1.1 is arbitrary
            N = self._exponential_poisson_N_from_beta(impressions, reach, beta)
        else:
            N = self._exponential_poisson_N(impressions, reach)
            beta = self._exponential_poisson_beta(impressions, N, reach)

        return N, KInflatedGammaPoissonDistribution(1.0, beta, [])

    def _fit_histogram_fixed_length_a(self, h, I, alpha0, beta0, N0, a0):
        """Given a fixed length vector of inflation values, finds optimum fit."""

        def objective(params):
            alpha, beta, N = params[:3]
            a = params[3:]
            if len(a) and np.sum(a) > 1.0:
                return np.sum(h**2)
            dist = KInflatedGammaPoissonDistribution(alpha, beta, a)
            return self._chi2_distance(h, I, N, dist)

        p0 = [alpha0, beta0, N0] + a0
        bounds = ([(1e-6, 100.0), (1e-6,100.0), (Nmin, 100.0 * Nmin)] +
                  [(0.0, 1.0)] * len(a0))
        
        result = scipy.optimize.minimize(
            objective, p0, method="L-BFGS-B", bounds=bounds
        )

        alpha, beta, N = result.x[:3]
        a = result.x[3:]
        dist = KInflatedGammaPoissonDistribution(alpha, beta, a)
        score = self._chi2_distance(h, I, N, dist)

        return alpha, beta, N, a, score

    def _fit(self):
        """Chi-squared fit to histogram h.

        Computes parameters alpha, beta, Imax, N such that the histogram hbar of
        of expected frequencies minimizes the metric

          chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i] +
             w * (\sum(h[i] - hbar[i]))^2

        where w is the one_plus_reach_weight.

        Experience shows that finding the optimal value depends on starting from
        a point where the estimated audience size is not too far from the true
        audience size.  So, this code retries the optimization for various different
        values of the estimated audience size.  The number of such attempts is given
        by nstarting_points.
        """
        N0, dist = self._fit_exponential_poisson_model(self._reach_point)

        if self._reach_point.max_frequency == 1:
            self._max_reach = N0
            self._max_impressions = N0 * dist.expected_value()
            self._dist = dist
            self._fit_computed = True
            return
        
        h = self._reach_point.frequencies_with_kplus_bucket
        N, alpha, beta, a, score = self._fit_histogram_fixed_length_a(h, alpha0, beta0, N0, [])
        
        while chi2.cdf(score, len(h)) > 0.95 and len(a) < self._kmax:
            a += [self._pmf(len(a)+1, alpha, beta)]
            N, alpha, beta, a, score = self._fit_histogram_fixed_length_a(h, alpha0, beta0, N0, [])

        self._max_reach = N
        self._dist = KInflatedGammaPoissonDistribution(alpha, beta, a)
        self._max_impressions = N * dist.expected_value()
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
        if not self._fit_computed:
            self._fit()
        p = impressions[0] / self._max_impressions
        freqs = self._max_reach * self._dist.kreach(np.arange(1, max_frequency), p)
        kplus = self._max_reach * self._dist.kplusreach(max_frequency, p)
        hist = list(freqs) + [kplus]
        kplus_reaches = np.cumsum(hist[::-1])[::-1]
        if self._cpi:
            return ReachPoint(
                impressions, kplus_reaches, [impressions[0] * self._cpi]
            )
        else:
            return ReachPoint(impressions, kplus_reaches)

    def by_spend(self, spends: List[int], max_frequency: int = 1) -> ReachPoint:
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
        return self.by_impressions(
            [self.impressions_for_spend(spends[0])], max_frequency
        )

    def impressions_for_spend(self, spend: float) -> int:
        if not self._cpi:
            raise ValueError("Impression cost is not known for this ReachPoint.")
        return spend / self._cpi
