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


class KInflatedGammaPoissonModel(ReachCurve):
    """k-inflated Gamma-Poisson reach curve model for underreported counts.

    The k-inflated Gamma-Poisson distribution is like the Gamma-Poisson
    distribution except that the first k values of the PMF are allowed to
    be arbitrary.

    It is given by k+2 parameters: alpha, beta, a_1, a_2, ..., a_k.
    We assume that each a_i >= 0, and that a_1 + a_2 + ... + a_k < 1.
    
    Let GP(n | alpha, beta) be the PMF of the Gamma-Poisson distribution 
    with parameters alpha, beta.  Also, define s = a_1 + a_2 + ... + a_k,
    and define t = \sum_{i=1}^k GP(i | alpha, beta).  Then, the PMF of the
    k-inflated Gamma-Poisson distribution is given as

     kGP(n | alpha, beta, a_1, ..., a_k) 
          = a_n if n <= k, 
            (s / t) GP(n | alpha, beta) if n > k.

    As with the Gamma-Poisson distribution, we use the chi2 objective
    function for assessing goodness of fit.  That is to say, let h[i]
    be the number of people that were reached i times, and let hbar[i]
    be the estimated number of people that would be reached i times,
    given alpha, beta and N.  The objective function that we
    compute is as follows:

      chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i]

    One question is what value of k should be chosen.  We proposed
    to consider successive values of k starting from 0, and to stop
    when a fit is found such that the value of the objective function 
    is smaller than the value of the chi^2 statistic with f_max degrees
    of freedom, where f_max is the highest frequency that is recorded.
    """

    def __init__(self, data: List[ReachPoint], max_reach=None, kmax=10):
        """Constructs a Gamma-Poisson model of underreported count data.

        Args:
          data:  A list of consisting of a single ReachPoint to which the model is
            to be fit.
          max_reach:  Optional.  If specified, the maximum possible reach that can
            be achieved.
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
        self._max_reach = max_reach
        self._kmax = kmax
        if data[0].spends:
            self._cpi = data[0].spends[0] / data[0].impressions[0]
        else:
            self._cpi = None

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
        return scipy.stats.nbinom.pmf(n - 1, alpha, beta / (1.0 + beta))

    def _pmf(self, n, alpha, beta, a):
        """PMF of the k-inflated Gamma-Poisson distribution.

        This implementation makes use of the equivalence between the
        Gamma-Poisson distribution with parameters (alpha, beta) and
        the negative binomial distribution with parameters (p, r) =
        (1 / (1 + beta), alpha).

        Args:
          n:  Value(s) at which the distribution is to be evaluated.
            This can be a scalar or a numpy array.
          alpha: Alpha parameter of the Gamma-Poisson distribution.
          beta: Beta parameter of the Gamma-Poisson distribution.
          a: The array of fixed PMF values.  In other words,
            if n <= len(a), then _logpmf is a[n-1].
        Returns:
          f(n | alpha, beta), where f is the k-inflated Gamma-Poisson 
          distribution.
        """
        if not len(a):
            return self._gamma_poisson_pmf(n, alpha, beta)
        elif n <= len(a):
            return a[n-1]
        
        s = np.sum(a)
        t = np.sum(self._gamma_poisson_pmf(np.arange(1, len(a)+1), alpha, beta))
        u = self._gamma_poisson_pmf(n, alpha, beta)
        return ((1 - s) / (1 - t)) * u
                   
    def _knreach(self, k, n, I, Imax, alpha, beta, a):
        """Probability that a random user has n impressions of which k are shown.

        Computes the function:

            C(n, k) (I / I_max)^k (1 - I/I_max)^{n-k} f(n | alpha, beta).

        where C(n, k) is the binomial coefficient n!/[k!(n-k)!].

        Args:
          k:  Scalar number of impressions that were seen by the user.
          n:  Available inventory for the user.
            This can be either a scalar or a numpy array.
          I:  Total number of impressions shown to all users.
          Imax: Total size of impression inventory.
          alpha: Parameter of Gamma-Poisson distribution.
          beta: Parameter of Gamma-Poisson distribution.
          a: The array of fixed PMF values.  In other words,
            if n <= len(a), then _logpmf is a[n-1].
        Returns:
          Probability that a randomly chosen user will have an inventory
          of n impressions, of which k are shown.
        """
        kprob = scipy.stats.binom.pmf(k, n, I / Imax)
        return kprob * self._pmf(n, alpha, beta, a)

    def _kreach(self, k, I, Imax, alpha, beta, a):
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
          a: The array of fixed PMF values.  In other words,
            if n <= len(a), then _logpmf is a[n-1].
        Returns:
          For each k, probability that a randomly chosen user will have an inventory
          of n impressions, of which k are shown.
        """
        return np.array(
            [
                np.sum(
                    [
                        self._knreach(kv, n, I, Imax, alpha, beta, a)
                        for n in np.arange(kv, MAXIMUM_COMPUTATIONAL_FREQUENCY + 1)
                    ]
                    
                )
                for kv in k
            ]
        )

    def _expected_impressions(self, N, alpha, beta, a):
        """Estimates the expected size of impression inventory for N users.

        Makes use of the identity that the expected value of a negative binomial
        distribution with parameters p and r is pr/(1-p).

        Args:
          N:      total number of people that can potentially be reached.
          alpha:  shape parameter of Gamma distribution.
          beta:   rate parameter of Gamma distribution.
          a: The array of fixed PMF values.  In other words,
            if n <= len(a), then _logpmf is a[n-1].
        Returns:
          Expected size of impression inventory for a population of N users.
        """
        # N * (1.0 + alpha * beta)
        return N * np.sum([n * self._pmf(n, alpha, beta, a)
                           for n in np.arange(1, MAXIMUM_COMPUTATIONAL_FREQUENCY)])

    def _expected_histogram(self, I, Imax, N, alpha, beta, a, max_freq=10):
        """Computes an estimated histogram.

        Args:
          I:      number of impressions that were purchased.
          Imax:   total size of impression inventory.
          N:      total number of people that can potentially be reached.
          alpha:  shape parameter of Gamma distribution.
          beta:   rate parameter of Gamma distribution.
          a: The array of fixed PMF values.  In other words,
            if n <= len(a), then _logpmf is a[n-1].
          max_freq: maximum frequency to report in the output histogram.
        Returns:
          An np.array h where h[i] is the expected number of users who will see i+1
          impressions.
        """
        return N * self._kreach(np.arange(1, max_freq + 1), I, Imax, alpha, beta, a)

    def _impression_count(self, h):
        """Number of impressions recorded in the histogram h."""
        return np.sum([h[i] * (i + 1) for i in range(len(h))])

    def _histogram_from_reach_point(self, point):
        """Returns a frequency histogram from a reach point.

        Args:
          point: A ReachPoint
        Returns:
          A list h[], where h[i] is the number of people reached exactly i
          times.  Note that this histogram will have length one less than the
          list of k+ reach values in ReachPoint, because the last k-plus
          reach value in the ReachPoint represents people reached at various
          different frequencies.
        """
        h = [point.frequency(i) for i in range(1, point.max_frequency)]
        return h

    def _histogram_with_kplus_bucket(self, point):
        """Computes a histogram whose last bucket is k+ reach.

        Args:
          point:  A ReachPoint for which the histogram is to be
            computed.
        Returns:
          A list h[] where h[i] is the number of people reached
          exactly i+1 times, except for h[-1], which records the
          number of people reached len(h) or more times.
        """
        # A partial histogram of people who were reached exactly k times.
        h = self._histogram_from_reach_point(point)

        # The number of people who were reach len(h) or more times.  These
        # people are not counted in the histogram h.
        kplus = len(h) + 1
        kplus_reach = point.reach(kplus)

        # Append kplus_reach to h.  So, the last value of h is treated
        # specially.
        h.append(kplus_reach)
        h = np.array(h)
        return h

    def _chi2_distance_with_one_plus_reach(
            self, h, I, N, alpha, beta, a, one_plus_reach_weight=0.0
    ):
        """Returns distance between actual and expected histograms.

        Computes the metric

            chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i] +
               w * (\sum_i (h[i] - hbar[i]))^2

        where
            hbar is the histogram that would be expected given parameters
               I, N, alpha, beta
            w is the one_plus_reach_weight that is used to determine how
               much to weight an exact match of 1+ reach.

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
          alpha:  Parameter of the Gamma-Poisson distribution.
          beta:  Second parameter of the Gamma-Poisson distribution.
          a: The array of fixed PMF values.  In other words,
            if n <= len(a), then _logpmf is a[n-1].
          one_plus_reach_weight:  Weight to be given to the squared
            difference in 1+ reach values for the actual and estimated
            histograms.
        Returns:
          chi^2 distance between the actual and expected histograms, plus
          a term for weighting the difference in 1+ reach.
        """
        # Estimate total number of potential impressions
        Imax = self._expected_impressions(N, alpha, beta, a)
        if Imax <= I:
            return np.sum(np.array(h) ** 2)

        hbar = list(self._expected_histogram(I, Imax, N, alpha, beta, a, len(h) - 1))

        # Compute expected number of people in the k+ bucket.
        kplus_prob = 1.0 - np.sum(
            self._kreach(np.arange(0, len(h)), I, Imax, alpha, beta, a)
        )
        hbar.append(kplus_prob * N)
        hbar = np.array(hbar)

        actual_oneplus = np.sum(h)
        predicted_oneplus = np.sum(hbar)
        oneplus_error = (
            one_plus_reach_weight * (actual_oneplus - predicted_oneplus) ** 2
        )

        obj = np.sum((hbar - h) ** 2 / (hbar + 1e-6)) + oneplus_error

        if np.isnan(obj):
            logging.vlog(2, f"alpha {alpha} beta {beta} N {N} Imax {Imax}")
            logging.vlog(2, f"h    {h}")
            logging.vlog(2, f"hbar {hbar}")
            raise RuntimeError("Invalid value of objective function")

        return obj

    def _fit_histogram_chi2_distance(
        self, point, one_plus_reach_weight=0.0, nstarting_points=10
    ):
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

        Args:
          point:  A ReachPoint whose histogram is to be approximated.
          one_plus_reach_weight:  Weight to be given to the squared
            difference in 1+ reach values for the actual and estimated
            histograms.
          nstarting_points: The number of different starting points that
            should be tried when performing optimizations.
        Returns:
          A tuple (Imax, N, alpha, beta, a) representing the parameters of the
          best fit that was found.
        """
        ### TODO(matthewclegg): Rewrite this
        
        # There must be at least this many people in the final model
        Nmin = point.reach(1)

        # This is the number of impressions that were actually served.
        I0 = point.impressions[0]

        # The histogram of the data.
        h = self._histogram_with_kplus_bucket(point)

        # Best score found so far of the objective function.
        best_score = 1e99

        # Calculate the set of audience sizes that are used as initial
        # starting points.
        Nvalues = Nmin / (np.arange(1, nstarting_points + 1) / (nstarting_points + 1))

        # If the objective function for an optimization attempt is smaller than
        # this value, then we have found a model that fits the data.  There is no
        # need to continue searching for a better fit.
        early_stopping_value = scipy.stats.chi2.ppf(0.98, len(h))

        for N0 in Nvalues:
            # Choose a reasonable starting point for optimization.
            Imax0, alpha0, beta0 = self._fit_histogram_fixed_N(
                point, N0, one_plus_reach_weight
            )

            def gamma_obj(params):
                """Objective function for optimization."""
                alpha, beta, N = params
                return self._chi2_distance_with_one_plus_reach(
                    h, I0, N, alpha, beta, one_plus_reach_weight
                )

            result = scipy.optimize.minimize(
                gamma_obj,
                (alpha0, beta0, N0),
                method="L-BFGS-B",
                bounds=((1e-6, 100.0), (1e-6, 100.0), (Nmin, 100.0 * Nmin)),
            )

            if result.fun < best_score:
                best_alpha, best_beta, best_N = result.x
                best_score = result.fun

            if best_score < early_stopping_value:
                break

        best_Imax = self._expected_impressions(best_N, best_alpha, best_beta)
        return best_Imax, best_N, best_alpha, best_beta

    def _fit(self) -> None:
        """Fits a model to the data that was provided in the constructor."""

        Imax, N, alpha, beta, a = self._fit_histogram_chi2_distance(self._reach_point)
        self._max_impressions = Imax
        self._max_reach = N
        self._alpha = alpha
        self._beta = beta
        self._a = a

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
            self._a,
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
        if len(spends) != 1:
            raise ValueError("Spend vector must have a length of 1.")
        return self.by_impressions(
            [self.impressions_for_spend(spends[0])], max_frequency
        )

    def impressions_for_spend(self, spend: float) -> int:
        if not self._cpi:
            raise ValueError("Impression cost is not known for this ReachPoint.")
        return spend / self._cpi
