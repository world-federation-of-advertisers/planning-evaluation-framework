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
        if data[0].impressions[0] < 0.001:
            raise ValueError("Attempt to create model with 0 impressions")
        if data[0].impressions[0] < data[0].reach(1):
            raise ValueError(
                "Cannot have a model with fewer impressions than reached people"
            )
        self._reach_point = data[0]
        self._max_reach = max_reach
        if data[0].spends:
            self._cpi = data[0].spends[0] / data[0].impressions[0]
        else:
            self._cpi = None

    def _logpmf(self, n, alpha, beta):
        """Log of the PMF of the shifted Gamma-Poisson distribution.

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
        return scipy.stats.nbinom.logpmf(n - 1, alpha, 1.0 / (1.0 + beta))

    def _knreach(self, k, n, I, Imax, alpha, beta):
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
        return N * (1.0 + alpha * beta)

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
        self, h, I, N, alpha, beta, one_plus_reach_weight=1.0
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
          one_plus_reach_weight:  Weight to be given to the squared
            difference in 1+ reach values for the actual and estimated
            histograms.
        Returns:
          chi^2 distance between the actual and expected histograms, plus
          a term for weighting the difference in 1+ reach.
        """
        # Estimate total number of potential impressions
        Imax = self._expected_impressions(N, alpha, beta)
        if Imax <= I:
            return np.sum(np.array(h) ** 2)

        hbar = list(self._expected_histogram(I, Imax, N, alpha, beta, len(h) - 1))

        # Compute expected number of people in the k+ bucket.
        kplus_prob = 1.0 - np.sum(
            self._kreach(np.arange(0, len(h)), I, Imax, alpha, beta)
        )
        hbar.append(kplus_prob * N)
        hbar = np.array(hbar)

        actual_oneplus = np.sum(h)
        predicted_oneplus = np.sum(hbar)
        oneplus_error = one_plus_reach_weight * (actual_oneplus - predicted_oneplus) ** 2

        obj = np.sum((hbar - h) ** 2 / (hbar + 1e-6)) + oneplus_error

        if np.isnan(obj):
            logging.vlog(2, f"alpha {alpha} beta {beta} N {N} Imax {Imax}")
            logging.vlog(2, f"h    {h}")
            logging.vlog(2, f"hbar {hbar}")
            raise RuntimeError("Invalid value of objective function")

        return obj

    def _fit_histogram_chi2_distance(
        self, point, one_plus_reach_weight=1.0, nstarting_points=10
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
          A tuple (Imax, N, alpha, beta) representing the parameters of the
          best fit that was found.
        """
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

    def _find_starting_point_for_fixed_N(self, h, I0, N, one_plus_reach_weight):
        """Searches for a good initial starting point for optimization.

        The standard optimizers unfortunately often fail to find the
        global minimum if they are started from suboptimal starting points.
        The surface represented by the objective function has large flat
        regions, and it often contains a valley with a false minimum that
        is adjacent to the basin containing the true minimum.  However,
        it seems to be the case that the basin containing the true minimum
        is often located relatively close to the origin.  This function performs
        a simple downhill search for a local optimum starting from a point
        close to the origin.  The value returned by this function can then
        be used as the starting point for a more thorough optimization
        attempt.

        Args:
          h:  Histogram with kplus reach bucket for the reach point
              that is to be fit.
          I0: The number of impressions that were shown to users.
          N:  The total audience size.
          one_plus_reach_weight:  Weight to be given to the squared
            difference in 1+ reach values for the actual and estimated
            histograms.
        Returns:
          alpha, beta: The values of the parameters that gave the best
          score of the objective function when performing a simple
          downhill search.
        """

        # Search for local optimum on a grid whose initial spacing is
        # given by grid_width
        grid_width = 0.05

        # Try to start from a point that is near the origin but also
        # in the feasible region
        alpha = beta = 0.1
        if I0 > N:
            alpha = beta = 1.1 * np.sqrt(I0 / N - 1) + grid_width

        score = 1e99
        while grid_width > 0.01:
            improvement_found = False
            local_best_score = score
            for delta in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                test_alpha = alpha + delta[0] * grid_width
                test_beta = beta + delta[1] * grid_width
                test_score = self._chi2_distance_with_one_plus_reach(
                    h, I0, N, test_alpha, test_beta, one_plus_reach_weight
                )
                if test_score < local_best_score:
                    improvement_found = True
                    local_best_alpha = test_alpha
                    local_best_beta = test_beta
                    local_best_score = test_score
            if improvement_found:
                alpha, beta, score = local_best_alpha, local_best_beta, local_best_score
            else:
                grid_width *= 0.5

        return alpha, beta

    def _fit_histogram_fixed_N(self, point, N, one_plus_reach_weight=1.0):
        """Chi-squared fit to histogram h.

        Computes parameters alpha, beta, Imax, N such that the histogram hbar of
        of expected frequencies minimizes the metric

          chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i] +
             w * (\sum(h[i] - hbar[i]))^2

        where w is the one_plus_reach_weight.

        Args:
          point:  A ReachPoint whose histogram is to be approximated.
          N:  int, total reachable audience.
          one_plus_reach_weight:  Weight to be given to the squared
            difference in 1+ reach values for the actual and estimated
            histograms.
        Returns:
          A tuple (Imax, alpha, beta) representing the parameters of the
          best fit that was found.
        """
        # This is the number of impressions that were actually served.
        I0 = point.impressions[0]

        # The histogram of the data.
        h = self._histogram_with_kplus_bucket(point)

        # Look for a good starting point for optimization
        alpha0, beta0 = self._find_starting_point_for_fixed_N(
            h, I0, N, one_plus_reach_weight
        )

        def gamma_obj(params):
            """Objective function for optimization."""
            alpha, beta = params
            score = self._chi2_distance_with_one_plus_reach(
                h, I0, N, alpha, beta, one_plus_reach_weight
            )
            return score

        result = scipy.optimize.minimize(
            gamma_obj,
            (alpha0, beta0),
            method="L-BFGS-B",
            bounds=((1e-6, 100.0), (1e-6, 100.0)),
        )

        best_alpha, best_beta = result.x
        best_Imax = self._expected_impressions(N, best_alpha, best_beta)
        return best_Imax, best_alpha, best_beta

    def _fit(self) -> None:
        """Fits a model to the data that was provided in the constructor."""

        if self._max_reach is None:
            Imax, N, alpha, beta = self._fit_histogram_chi2_distance(self._reach_point)
            self._max_impressions = Imax
            self._max_reach = N
            self._alpha = alpha
            self._beta = beta
        else:
            Imax, alpha, beta = self._fit_histogram_fixed_N(
                self._reach_point, self._max_reach
            )
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
        if len(spends) != 1:
            raise ValueError("Spend vector must have a length of 1.")
        return self.by_impressions(
            [self.impressions_for_spend(spends[0])], max_frequency
        )

    def impressions_for_spend(self, spend: float) -> int:
        if not self._cpi:
            raise ValueError("Impression cost is not known for this ReachPoint.")
        return spend / self._cpi
