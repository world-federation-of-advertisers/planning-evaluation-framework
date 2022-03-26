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
import numpy.typing as npt
import scipy.stats
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve

# The following constant defines the highest frequency that will be
# used for parameter estimation.  This is chosen to be large enough so that
# model truncation is ignorable.
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

    def __init__(self, alpha: float, beta: float, a: List[float]):
        """k-Inflated Gamma-Poisson with parameters alpha, beta, a."""
        if sum(a) > 1.0:
            # In theory, it should be an error to create a distribution where
            # the values of sum to a value > 1, but in practice the easiest
            # solution is just to scale down the sum to 1.
            a = np.array(a) / np.sum(a)

        self._alpha = alpha
        self._beta = beta
        self._a = a

        n = len(a)
        s = np.sum(a)
        t = np.sum(self._gamma_poisson_pmf(np.arange(1, n + 1), alpha, beta))
        # Note that MAXIMUM_COMPUTATIONAL_FREQUENCY is a relatively large value such
        # that values of the PMF beyond this value can be ignored.
        u = self._gamma_poisson_pmf(
            np.arange(n + 1, MAXIMUM_COMPUTATIONAL_FREQUENCY), alpha, beta
        )
        if t >= 1:
            pmf_tail = 0.0 * u
        else:
            pmf_tail = ((1 - s) / (1 - t)) * u
        # self._pmf[n] = kGP(n | alpha, beta, a_1, ..., a_k)
        # Note, we prepend 0 to self._pmf to make indexing more convenient.
        self._pmf = np.array([0.0] + list(a) + list(pmf_tail))
        # Force the series to sum to 1.
        self._pmf[-1] = 1.0 - np.sum(self._pmf[:-1])

    def _gamma_poisson_pmf(
        self, n: Union[int, Iterable[int]], alpha: float, beta: float
    ) -> Union[float, npt.ArrayLike]:
        """PMF of the shifted Gamma-Poisson distribution.

        This implementation makes use of the equivalence between the
        Gamma-Poisson distribution with parameters (alpha, beta) and
        the negative binomial distribution with parameters (p, r) =
        (1 / (1 + beta), alpha).

        Args:
          n:  Value(s) at which the distribution is to be evaluated.
            This can be a scalar or a numpy array.
          alpha: float, alpha parameter of the Gamma-Poisson distribution.
          beta: float, beta parameter of the Gamma-Poisson distribution.
        Returns:
          f(n | alpha, beta), where f is the PMF of the Gamma-Poisson distribution.
        """
        # Note scipy.stats.nbinom uses a different parameterization than
        # the Wikipedia parameterization.  Under this parameterization,
        # p -> 1-p, so the parameter 1/(1 + beta) becomes beta/(1 + beta).
        return scipy.stats.nbinom.pmf(n - 1, alpha, 1.0 / (1.0 + beta))

    def pmf(self, n: Union[int, Iterable[int]]) -> Union[float, npt.ArrayLike]:
        """PMF of the k-inflated Gamma-Poisson distribution.

        Args:
          n:  Value(s) at which the distribution is to be evaluated.
            This can be a scalar or a numpy array.
        Returns:
          kGP(n | alpha, beta, a_1, ..., a_k), where f is the k-inflated
          Gamma-Poisson distribution.
        """
        return self._pmf[n]

    def knreach(
        self, k: int, n: Union[int, Iterable[int]], p: float
    ) -> Union[float, npt.ArrayLike]:
        """Probability that a random user has n impressions of which k are shown.

        Computes the function:

            C(n, k) p^k (1 - p)^{n-k} kGP(n | alpha, beta, a).

        where C(n, k) is the binomial coefficient n!/[k!(n-k)!].

        Args:
          k:  (C, ) ndarray. Numbers of impressions that were seen by the user.
          n:  (M, ) ndarray. Inventory size for the user.
          p:  float. Probability that a given impression will be chosen.
        Returns:
          (C, M) ndarray. Probability that a randomly chosen user will have an
          inventory of n impressions, of which k are shown.
        """
        kprob = scipy.stats.binom.pmf(k.reshape(-1, 1), n.reshape(1, -1), p)
        return kprob * self.pmf(n)

    def kreach(self, k: Iterable[float], p: float) -> npt.ArrayLike:
        """Probability that a random user receives k impressions.

        Computes the function:
            sum_{n=k}^{infinity} g(k, n | alpha, beta, I, I_max).
        where g(k, n | alpha, beta, I, I_max) is the probability that
        a randomly chosen user will have an available inventory of n
        impressions, of which k are shown.

        Args:
          k:  (C, ) ndarray specifying number of impressions that were seen by the user.
          p:  float, probability that a given impression will be chosen.
        Returns:
          (C, ) ndarray.For each k, probability that a randomly chosen user
          will have an inventory of n impressions, of which k are shown.
        """
        k = np.asarray(k)
        mat = self.knreach(k, np.arange(len(self._pmf)), p)  # shape:CxM
        upper_triangular = np.triu(mat)
        return np.sum(upper_triangular, axis=-1)

    def kplusreach(self, k: int, p: float) -> npt.ArrayLike:
        """Probability that a random user receives k or more impressions.

        Computed as one minus the probability that the user receives k-1 or
        fewer impressions.

        Args:
          k:  int, specifying number of impressions that were seen by the user.
          p:  float, probability that a given impression will be chosen.
        Returns:
          For each k, probability that a randomly chosen user will have an inventory
          of n impressions, of which k are shown.
        """
        if k == 0:
            return 1.0
        else:
            return 1.0 - np.sum(self.kreach(np.arange(k), p))

    def expected_value(self) -> float:
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
    is smaller than the critical value of the chi^2 statistic with
    f_max degrees of freedom, where f_max is the highest frequency
    that is recorded.
    """

    def __init__(
        self,
        data: List[ReachPoint],
        kmax: int = 10,
        debug: bool = False,
        extrapolation_multiplier: float = 1.0,
    ):
        """Constructs a Gamma-Poisson model of underreported count data.

        Args:
          data:  A list consisting of the single ReachPoint to which the model is
            to be fit.
          kmax:  Maximum number of values of the PMF that are allowed to be set
            arbitrarily.
          debug:  If True, prints debug information as models are fit.
          extrapolation_multiplier:  Float.  If specified, then a penalty term is
            introduced that penalizes models where the expected number of impressions
            in the inventory is less than extrapolation_multiplier * the observed number
            of impressions in the data.
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
        self._kmax = kmax
        self._debug = debug
        self._fit_computed = False
        self._extrapolation_multiplier = extrapolation_multiplier
        self._extrapolation_value = extrapolation_multiplier * data[0].impressions[0]
        if data[0].spends:
            self._cpi = data[0].spends[0] / data[0].impressions[0]
        else:
            self._cpi = None

    def _chi2_distance(
        self,
        h: npt.ArrayLike,
        I: float,
        N: float,
        dist: KInflatedGammaPoissonDistribution,
    ) -> float:
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
            # It is necessary to treat this as a special case because the
            # probability that an impression will be served is I / Imax.
            # If I / Imax > 1, then the computation of the histogram fails.
            return np.sum(np.array(h) ** 2)

        freqs = N * dist.kreach(np.arange(1, len(h)), I / Imax)
        kplus = N * dist.kplusreach(len(h), I / Imax)
        hbar = np.array(list(freqs) + [kplus])

        obj = np.sum((hbar - h) ** 2 / (hbar + 1e-6))

        if np.isnan(obj):
            logging.vlog(2, f"alpha {dist._alpha} beta {dist._beta} N {N} Imax {Imax}")
            logging.vlog(2, f"h    {h}")
            logging.vlog(2, f"hbar {hbar}")
            raise RuntimeError("Invalid value of objective function")

        return obj

    def _exponential_poisson_reach(self, I: float, N: float, beta: float) -> float:
        """Estimates reach in an exponential poisson model.

        Estimates reach when I impressions are purchased from a population
        of N users where the impression count of each user is distributed
        according to the exponential poisson distribution with parameter beta.

        Under the exponential poisson model, the probability f(n) that a
        user will have n impressions is f(n) = (1 - p) * p^{n-1}, where p =
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
        if I >= (beta + 1) * N:
            return N
        return N * I * (beta + 1) / (N + beta * (N + I))

    def _exponential_poisson_beta(self, I: float, N: float, R: float) -> float:
        """Estimate beta given impressions, audience size and reach.

        This is computed by backsolving the formula for reach used in
        _exponential_poisson_reach for beta in terms of I, N and R.

        Args:
          I:  The hypothetical number of impressions purchased.
          N:  The audience size.
          R:  The 1+ reach that was obtained.
        Returns:
          The corresponding parameter beta of the exponential-poisson
          model that achieves reach R when I impressions are delivered
          to an audience of size N.
        """
        if I * R - N * (I - R) <= 0:
            return 1e3
        return (N * (I - R)) / (I * R - N * (I - R))

    def _exponential_poisson_N_from_beta(
        self, I: float, R: float, beta: float
    ) -> float:
        """Estimate audience size from impressions, reach and beta.

        This is computed by backsolving the formula for reach used in
        _exponential_poisson_reach for N in terms of I, R and beta.

        Args:
          I:  The hypothetical number of impressions purchased.
          R:  The 1+ reach that was obtained.
          beta: The parameter of the exponential-poisson distribution.
        Returns:
          The corresponding parameter N of the exponential-poisson
          model that achieves reach R when I impressions are delivered
          to an audience of size N with parameter beta.
        """
        if I < R + 1e-3:
            return 100.0 * R
        return -(I * beta * R) / ((beta + 1) * (R - I))

    def _exponential_poisson_N(self, I: float, R: float, mu_scale_factor=1.5) -> float:
        """Estimate audience size from I and R.

        We know that the mean mu of the distribution has to be greater than
        I / R.  Since beta = mu - 1, this says that beta >= I / R - 1.
        But how much bigger?  That cannot easily be determined from just one
        point.  Here, we assume beta = s * (I / R - 1), where s is some
        small scale factor that is at least one.  From that, we estimate N.
        """
        if mu_scale_factor < 1.0:
            raise ValueError("mu_scale_factor must be at least 1.0")
        elif I < R + 1e-3:
            return 100.0 * R
        return R * (mu_scale_factor * I - R) / (mu_scale_factor * (I - R))

    def _fit_exponential_poisson_model(
        self, point: ReachPoint, Imin: float = None
    ) -> Tuple[float, KInflatedGammaPoissonDistribution]:
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
          Imin:  Minimum number of inventory impressions that the returned
            model should have.
        Returns:
          A pair (N, dist), where N is the estimated audience size and dist
          is an estimated KInflatedGammaPoissonDistribution representing an
          exponential-poisson distribution.
        """
        impressions = point.impressions[0]
        if Imin is not None and impressions < Imin:
            impressions = Imin
        reach = point.reach()
        if len(point.frequencies) > 1:
            # Estimate the mean from the distribution found in the reach point
            mu = impressions / min(reach, impressions - 1)
            beta = (mu - 1) * 1.2  # 1.2 is arbitrary
            N = self._exponential_poisson_N_from_beta(impressions, reach, beta)
        else:
            N = self._exponential_poisson_N(impressions, reach)
            beta = self._exponential_poisson_beta(impressions, N, reach)

        # Check if the chosen values of N, beta result in an impossibly low
        # number of impressions.  If so, adjust them upwards.
        Imax = N * (beta + 1)
        while Imax < impressions:
            N = max(1, 1.1 * N)
            beta = self._exponential_poisson_beta(impressions, N, reach)
            Imax = N * (beta + 1)

        return N, KInflatedGammaPoissonDistribution(1.0, beta, [])

    def _fit_histogram_fixed_length_a(
        self,
        h: npt.ArrayLike,
        I: float,
        Imin: float,
        N0: float,
        alpha0: float,
        beta0: float,
        a0: Iterable[float],
    ) -> Tuple[float, KInflatedGammaPoissonDistribution, float]:
        """Given a fixed length vector of inflation values, finds optimum fit."""

        p0 = list([alpha0, beta0, N0]) + list(a0)
        Nmin = np.sum(h)
        bounds = [(1e-4, 1000.0), (1e-4, 10000.0), (Nmin, 100.0 * Nmin)] + [
            (0.0, 1.0)
        ] * len(a0)

        def objective(params):
            alpha, beta, N = params[:3]
            a = params[3:]
            for i in range(len(params)):
                if params[i] < bounds[i][0] or bounds[i][1] < params[i]:
                    return np.sum(np.array(h) ** 2) + Imin ** 2
            if len(a) and np.sum(a) > 1.0:
                return np.sum(np.array(h) ** 2) + Imin ** 2
            dist = KInflatedGammaPoissonDistribution(alpha, beta, a)
            score = self._chi2_distance(h, I, N, dist)
            Imax = N * dist.expected_value()
            if Imax < Imin:
                score += (Imin - Imax) ** 2
            return score

        result = scipy.optimize.minimize(objective, p0, method="Nelder-Mead")

        alpha, beta, N = result.x[:3]
        a = result.x[3:]
        dist = KInflatedGammaPoissonDistribution(alpha, beta, a)
        score = self._chi2_distance(h, I, N, dist)

        return N, dist, score

    def print_fit_header(self):
        if not self._debug:
            return
        print(
            "{:15s} {:>7s} {:>7s} {:>8s} {:>8s} {:>2s}  distribution".format(
                " ", "score", "N", "alpha", "beta", "n"
            )
        )

    def print_fit(
        self,
        msg: str,
        score: float,
        N: float,
        alpha: float,
        beta: float,
        a: List[float],
    ):
        if not self._debug:
            return
        dist = KInflatedGammaPoissonDistribution(alpha, beta, a)
        dist_str = " ".join([f"{dist.pmf(i):7.4f}" for i in range(1, 10)])
        print(
            f"{msg:15s} {score:7.2f} {N:7.0f} {alpha:8.3f} {beta:8.3f} {len(a):2d} {dist_str}"
        )

    def _fit_point(
        self, point: ReachPoint, Imin: float
    ) -> Tuple[float, KInflatedGammaPoissonDistribution]:
        """Chi-squared fit to histogram h.

        Computes parameters alpha, beta, Imax, N such that the histogram hbar of
        of expected frequencies minimizes the metric

          chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i]

        subject to the constraint the number of impressions in the fitted model
        is at least Imin.

        Args:
          point:  The point to which the data is to be fit.
          Imin: The minimum number of impressions that the resulting
            fit should have.
        Returns:
          N, dist of the fit that was found, where N is the estimated audience size
          and dist is the best fitting k-inflated Gamma Poisson distribution.
          The expected number of impressions of this pair will be at least
          Imin.
        """
        N0, dist = self._fit_exponential_poisson_model(point, Imin)
        h = point.frequencies_with_kplus_bucket
        I = point.impressions
        score = self._chi2_distance(h, I, N0, dist)
        self.print_fit_header()
        self.print_fit("exp-poisson", score, N0, dist._alpha, dist._beta, dist._a)

        if point.max_frequency == 1:
            return N0, dist

        N, dist, score = self._fit_histogram_fixed_length_a(
            h, I, Imin, N0, dist._alpha, dist._beta, []
        )
        self.print_fit("start", score, N, dist._alpha, dist._beta, dist._a)

        while scipy.stats.chi2.cdf(score, len(h)) > 0.95 and len(dist._a) < self._kmax:
            a = list(dist._a) + [dist.pmf(len(dist._a) + 1)]
            N, dist, score = self._fit_histogram_fixed_length_a(
                h, I, Imin, N, dist._alpha, dist._beta, a
            )
            self.print_fit(f"{len(a)}", score, N, dist._alpha, dist._beta, dist._a)

        return N, dist

    def _fit(self, Imin: float = None):
        """Chi-squared fit to histogram h.

        Computes parameters alpha, beta, Imax, N such that the histogram hbar of
        of expected frequencies minimizes the metric

          chi2(h, hbar) = \sum_i (h[i] - hbar[i])^2 / hbar[i]

        subject to the constraint the number of impressions in the fitted model
        is at least Imin.
        """
        if self._fit_computed:
            return

        if Imin is None:
            Imin = self._extrapolation_value

        N, dist = self._fit_point(self._reach_point, Imin)

        self._max_reach = N
        self._dist = dist
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

        self._fit(impressions[0] + 1)

        p = min(impressions[0] / self._max_impressions, 1.0)
        freqs = self._max_reach * self._dist.kreach(np.arange(1, max_frequency), p)
        kplus = self._max_reach * self._dist.kplusreach(max_frequency, p)
        hist = list(freqs) + [kplus]
        kplus_reaches = np.cumsum(hist[::-1])[::-1]
        if self._cpi:
            return ReachPoint(impressions, kplus_reaches, [impressions[0] * self._cpi])
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
