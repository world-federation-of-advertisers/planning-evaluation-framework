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
"""Analytical variance formulas for sampling + LiquidLegions."""

import numpy as np
from numpy import exp as exp
from scipy.special import expi as ei


class AnalyticalAccuracyEvaluator:

    """Analytical formulas to quickly evaluate the accuracy of sampling + LiquidLegions.

    To estimate reach and frequency histogram using sampling + LiquidLegions, we
        1. ask each publisher to sample a fraction of vids using the same hash.
        2. ask each publisher to obtain the the LiquidLegions of the sampled, reached vids, with the same
            decay rate and number of registers
        3. Use MPC + SameKeyAggregator protocol to estimate reach and relative frequency histogram of the
            sampled ids.  Scale up the reach estimate.  Use the relative frequency histogram of the sample
            as the estimate for population.
    This class computes the standard deviations of the estimates of
        - reach, and
        - proportion of each frequency bucket.
    TODO(jiayu): derive the stds for k+ reaches and shuffle distance.

    Please read Section of 1 the companion doc:
    https://drive.google.com/file/d/1FNeMZnSYNMcsAMtzFY2ZyM_Tl1YYoEid/view?usp=sharing&resourcekey=0-NTSWpzbL2HkumqI-AUZlIg
    for precise problem description and formulas.
    """

    def __init__(self, a, m, pi, ssreach, ssfreq):
        """Specify parameters that determine the accuracy.

        Args:
            a: decay rate of LiquidLegions.
            m: number of registers in LiquidLegions.
            pi: sampling fraction, which equals 1 / <number of sampling buckets>.
            ssreach: variance of noise added on the number of non-empty registers.
            ssfreq: variance of noise added on each bucket of the frequency histogram of active registers.
        """
        self.a = a
        self.m = int(m)
        self.pi = pi
        self.ssreach = ssreach
        self.ssfreq = ssfreq
        self.c = a / m / (1 - exp(-a))

    def _local_multiply(self, n, multiplier):
        """An operation repeatedly used in the formulas."""
        return multiplier * self.c * n

    def _local_exp(self, n, multiplier):
        """An operation repeatedly used in the formulas."""
        return exp(-self._local_multiply(n, multiplier))

    def _local_scaled_exp(self, n, multiplier):
        """An operation repeatedly used in the formulas."""
        return self._local_multiply(n, multiplier) * self._local_exp(n, multiplier)

    def _local_ei(self, n, multiplier):
        """An operation repeatedly used in the formulas."""
        return ei(-self._local_multiply(n, multiplier))

    def var_n_hat(self, n):
        """Variance of the reach estimate.  Formula (1) in the companion doc.

        Args:
            n: the true reach.

        Returns:
            the variance of reach estimate when the true reach is n.
        """
        a, m, le, lei = self.a, self.m, self._local_exp, self._local_ei
        y = self._ey(n)
        res = lei(y, 1) - lei(y, 2) - lei(y, exp(-a)) + lei(y, 2 * exp(-a))
        res *= self.a * n ** 2 / self.m
        res /= (le(y, exp(-a)) - le(y, 1)) ** 2
        return res - n

    def std_n_hat(self, n):
        """Standard deviation of the reach estimate."""
        return np.sqrt(self.var_n_hat(n))

    def relative_std_n_hat(self, n):
        """Relative (to the true cardinality) standard deviation of the reach estimate."""
        return self.std_n_hat(n) / n

    def _theta_1(self, x):
        """Formula (9) in the companion doc."""
        a, m, le = self.a, self.m, self._local_exp
        return m / a / x * (le(x, exp(-a)) - le(x, 1))

    def _theta_2(self, x):
        """Formula (10) in the companion doc."""
        a, m, le, lse = self.a, self.m, self._local_exp, self._local_scaled_exp
        res = lse(x, exp(-a)) - lse(x, 1) + le(x, exp(-a)) - le(x, 1)
        return res * m / a / x ** 2

    def _theta_1_prime(self, x):
        """Formula (11) in the companion doc."""
        a, m, c, le = self.a, self.m, self.c, self._local_exp
        res = m / a / x * (-c * exp(-a) * le(x, exp(-a)) + c * le(x, 1))
        return res - m / a / x ** 2 * (le(x, exp(-a)) - le(x, 1))

    def _ey(self, n):
        """E(Y) in formula (11)."""
        return n * self.pi

    def _vy(self, n):
        """Var(Y) in formula (11)."""
        return n * self.pi * (1 - self.pi)

    def _ea(self, ey):
        """Expected number of active registers.  Formula (6)."""
        return ey * self._theta_1(ey - 1)

    def get_ea(self, n):
        """Print expected number of active registers.  May be used as it is an important metric."""
        return self._ea(self._ey(n))

    def _va(self, ey, vy):
        """Variance of number of active registers.  Formula (7)."""
        t1, t2, t1p = self._theta_1, self._theta_2, self._theta_1_prime
        res = ey * t1(ey - 1)
        res -= ey ** 2 * t2(2 * ey - 4)
        res += ey ** 3 * (t2(ey - 2)) ** 2
        return res + vy * (t1(ey - 1) + ey * t1p(ey - 1)) ** 2

    def _va_prime(self, va, max_freq):
        """Var(A') in formula (5)."""
        return va + max_freq * self.ssfreq

    def _vak_prime(self, ea, va, n, nk):
        """Var(A_k') in formula (5)."""
        res = ((nk / n) ** 2 - nk * (n - nk) / n ** 3) * va
        res += nk * (n - nk) / n ** 3 * ea * (n - ea)
        return res + self.ssfreq

    def _cov_a_prime_ak_prime(self, va, n, nk):
        """Cov(A', A_k') in formula (5)."""
        return nk / n * va + self.ssfreq

    def _cov_mat(self, ea, va, n, nk, max_freq):
        """The 2 * 2 matrix in formula (3)."""
        m11 = self._vak_prime(ea, va, n, nk)
        m12 = self._cov_a_prime_ak_prime(va, n, nk)
        m22 = self._va_prime(va, max_freq)
        return np.array([[m11, m12], [m12, m22]])

    def _nabla(self, rk, ea):
        """The 2 * 1 vector in formula (4)."""
        return np.array([[1 / ea], [-rk / ea]])

    def var_rk_hat(self, n, nk, max_freq):
        """Variance of each element of relative frequency histogram, i.e., <k-reach> / <1+ reach> at each frequency level k.

        Formula (3) in the companion doc.

        Args:
            n: the true reach, i.e., 1+ reach.
            nk: the true k-reach.
            max_freq: the maximum frequency in the estimate of frequency histogram.

        Returns:
            Variance of each element of relative frequency histogram.
        """
        ey, vy = self._ey(n), self._vy(n)
        ea, va = self._ea(ey), self._va(ey, vy)
        cov_mat = self._cov_mat(ea, va, n, nk, max_freq)
        nabla = self._nabla(nk / n, ea)
        return (np.dot(nabla.transpose(), np.dot(cov_mat, nabla)))[0, 0]

    def std_rk_hat(self, n, nk, max_freq):
        """Standard deviation of each element of relative frequency histogram."""
        return np.sqrt(self.var_rk_hat(n, nk, max_freq))
