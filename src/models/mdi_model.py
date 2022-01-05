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
"""Minimum Deviation from Independence Multi-Publisher Model"""

import functools
from cvxopt.solvers import qp
from cvxopt import matrix
import numpy as np
import scipy.special

from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface


class MDITreeNode(ReachSurface):
  def __init__(self, total_audience_size, max_adjusted_frequency, max_frequency):
    pass

  def inventory_frequencies(self, max_frequency=None):
    raise RuntimeError("Not Implemented")

  def inventory_histogram(self, max_frequency=None):
    if max_frequency is None:
      max_frequency = self._max_frequency
    return self._total_audience_size * self.inventory_frequencies(max_frequency)

  def campaign_frequencies(self, probs, max_frequency=None):
    raise RuntimeError("Not Implemented")

  def campaign_histogram(self, probs, max_frequency=None):
    if max_frequency is None:
      max_frequency = self._max_frequency
    return self._total_audience_size * self.campaign_frequencies(probs, max_frequency)

  def fit(self, reach_points):
    pass

class MDITreeLeafNode(MDITreeNode):
  def __init__(self, curve, total_audience_size, max_spend, max_frequency=20):
    last_reach_point = curve.by_spend([max_spend], max_frequency)
    self._freq = np.zeros(shape=(max_frequency+1))
    for i, f in enumerate(last_reach_point.frequencies):
        self._freq[i+1] = f
    self._freq[-1] += last_reach_point.reach() - np.sum(self._freq)
    self._freq[0] = total_audience_size - last_reach_point.reach()
    self._freq = self._freq / total_audience_size
    self._total_audience_size = total_audience_size
    self._publisher_count = 1
    self._impression_count_vector = np.array(last_reach_point.impressions)
    self._max_spend_vector = np.array([max_spend])
    self._max_frequency = max_frequency

  def inventory_frequencies(self, max_frequency=None):
    if max_frequency is None:
      max_frequency = self._max_frequency
    if max_frequency < len(self._freq):
      return self._freq[:max_frequency+1]
    else:
      return np.array(list(self._freq) + [0.] * (max_frequency - len(self._freq)+ 1))

  def inventory_histogram(self, max_frequency=None):
    if max_frequency is None:
      max_frequency = self._max_frequency
    return self._total_audience_size * self.inventory_frequencies(max_frequency)

  def campaign_frequencies(self, probs, max_frequency=None):
    if max_frequency is None:
      max_frequency = self._max_frequency
    if len(probs) != 1:
      raise ValueError(f"probs must be of length 1.  Got {probs}")

    p = probs[0]
    q = 1 - p
    g = [0.0] * (max_frequency+1)
    for k in range(len(g)):
      for l in range(k, len(self._freq)):
         g[k] = g[k] + scipy.special.binom(l, k) * p**k * q**(l-k) * self._freq[l]
    return np.array(g)

  def campaign_histogram(self, probs, max_frequency=None):
    if max_frequency is None:
      max_frequency = self._max_frequency
    return self._total_audience_size * self.campaign_frequencies(probs, max_frequency)

  def impression_probability(self, probs):
    """Probability that an impression will be delivered to the campaign."""
    if len(probs) != 1:
      raise ValueError(f"probs must be of length 1.  Got {probs}")
    return probs[0]

  def by_impression_probability(self, prob, max_frequency=1):
    hist = self.campaign_histogram(prob)[1:]
    kplus_reaches = list(np.cumsum(hist[::-1]))[::-1]
    if max_frequency < len(kplus_reaches)-1:
        kplus_reaches = kplus_reaches[:max_frequency]
    elif len(kplus_reaches) < max_frequency:
        kplus_reaches += [0] * (max_frequency - len(kplus_reaches))
    impressions = self._impression_count_vector * prob
    spend = self._max_spend_vector * prob[0]
    return ReachPoint(impressions, kplus_reaches, list(spend))

  def by_impressions(self, impressions, max_frequency=1):
    """Expected number of people reached as a function of impressions."""
    return self.by_impression_probability(
        impressions/self._impression_count_vector, max_frequency)

  def by_spend(self, spends, max_frequency=1):
    return self.by_impression_probability(spends/self._max_spend_vector, max_frequency)

  def _fit(self):
    pass

class MDITreeInternalNode(MDITreeNode):
  def __init__(self, left_child, right_child, total_audience_size, max_frequency=20, basis_dimension=3):
    self._left_child = left_child
    self._right_child = right_child
    self._total_audience_size = total_audience_size
    self._basis = np.zeros(shape=(basis_dimension, basis_dimension))
    self._publisher_count = left_child._publisher_count + right_child._publisher_count
    self._impression_count_vector = np.concatenate([
        left_child._impression_count_vector,
        right_child._impression_count_vector
    ])
    self._impression_count = np.sum(self._impression_count_vector)
    self._max_spend_vector = np.concatenate([
      left_child._max_spend_vector,
      right_child._max_spend_vector                                        
    ])
    self._max_frequency = max_frequency
    self._basis_dimension = basis_dimension

  def inventory_frequencies(self, max_frequency=None):
    if max_frequency is None:
      max_frequency = self._max_frequency
    f_S = self._left_child.inventory_frequencies(max_frequency)
    f_T = self._right_child.inventory_frequencies(max_frequency)
    f = np.array([0.] * len(f_S))
    for k in range(len(f)):
      for l in range(k+1):
        if l < self._basis.shape[0] and k-l < self._basis.shape[1]:
          f[k] += f_S[l] * f_T[k-l] + self._basis[l, k-l]
        else:
          f[k] += f_S[l] * f_T[k-1]
    return f

  def campaign_frequencies(self, probs, max_frequency=None):
    if max_frequency is None:
      max_frequency = self._max_frequency
    g = np.array([0.] * (max_frequency+1))
    S_probs = probs[:self._left_child._publisher_count]
    T_probs = probs[self._left_child._publisher_count:]
    p_S = self._left_child.impression_probability(S_probs)
    p_T = self._right_child.impression_probability(T_probs)
    q_S = 1 - p_S
    q_T = 1 - p_T
    g_S = self._left_child.campaign_frequencies(S_probs, max_frequency)
    g_T = self._right_child.campaign_frequencies(T_probs, max_frequency)
    for k in range(len(g)):
      for k_1 in range(k+1):
        k_2 = k - k_1
        g[k] += g_S[k_1] * g_T[k_2]
        for l_1 in range(k_1, self._basis.shape[0]):
          for l_2 in range(k_2, self._basis.shape[1]):
            g[k] += (scipy.special.binom(l_1, k_1) * 
                     scipy.special.binom(l_2, k_2) *
                     p_S ** k_1 * q_S ** (l_1 - k_1) * 
                     p_T ** k_2 * q_T ** (l_2 - k_2) *
                     self._basis[l_1, l_2])
    # print(f"campaign freqs {probs} -> ({g.shape}) ({np.sum(g)}) {g[:5]}")
    g[-1] += 1 - np.sum(g)
    return g

  def by_impression_probability(self, prob, max_frequency=1):
    hist = self.campaign_histogram(prob)[1:]
    kplus_reaches = list(np.cumsum(hist[::-1]))[::-1]
    if max_frequency < len(kplus_reaches)-1:
        kplus_reaches = kplus_reaches[:max_frequency]
    elif len(kplus_reaches) < max_frequency:
        kplus_reaches += [0] * (max_frequency - len(kplus_reaches))
    impressions = np.array(prob) * self._impression_count_vector
    spend = np.array(prob) * self._max_spend_vector
    return ReachPoint(impressions, kplus_reaches, list(spend))

  def by_impressions(self, impressions, max_frequency=1):
    """Expected number of people reached as a function of impressions."""
    return self.by_impression_probability(
        np.array(impressions)/self._impression_count_vector,
        max_frequency)

  def by_spend(self, spends, max_frequency=1):
    return self.by_impression_probability(
        np.array(spends)/self._max_spend_vector, max_frequency)

  # @cache
  def impression_probability(self, probs):
    p_S = self._left_child.impression_probability(probs[:self._left_child._publisher_count])
    p_T = self._right_child.impression_probability(probs[self._left_child._publisher_count:])
    I_S = np.sum(self._left_child._impression_count_vector)
    I_T = np.sum(self._right_child._impression_count_vector)
    p_R = (p_S * I_S + p_T * I_T) / (I_S + I_T)
    return p_R

  def _fit(self, reach_points, hlambda=1.0):
    # Given a collection of campaign reach points, computes the best fitting 
    # basis matrix for this set of reach points.  This is accomplished by
    # encoding the problem as a convex optimization problem of the form:
    #   Minimize ||Bx - g||^2 + lambda ||x||^2
    #   Subject to 
    #     (1) L <= x <= U
    #     (2) R x = 0, C x = 0,
    # In this representation, 
    #    B is a matrix representing the coefficients
    #      of the basis elements, 
    #    g represents the deviation between the observed frequency and the
    #      frequency that is estimated under independence,
    #    x is the vector of basis elements that are to be estimated,
    #    L is the lower bound on each basis element,
    #    U is the upper bound on each basis element,
    #    R is a matrix expressing the row sums of the basis elements,
    #    C is a matrix expressing the column sums of the basis elements.
    #    hlambda is a regularization parameter.
    #
    # To solve this problem, we formulate it instead as a quadratic programming
    # problem,
    #    Minimize (1/2) x^T P x + q^T x
    #    Subject to the constraints
    #      (1) L <= x <= U
    #      (2) R x = 0, C x = 0
    # This can then be solved using quadratic programming optimizer, for example,
    # cvxopt.solvers.qp.  In this formulation, the objects P and q are taken as
    #    P = B^T B + lambda I
    #    q = -B^T g
    # I suspect that using L1-regularization would give an improvement, but I am
    # not aware of any implementations in Python.  The following paper might be
    # a possible starting point:
    #   Solntsev, Stefan, Jorge Nocedal, and Richard H. Byrd. 
    #   "An algorithm for quadratic ℓ1-regularized optimization with a 
    #   flexible active-set strategy." 
    #   Optimization Methods and Software 30.6 (2015): 1213-1237.
    # Assumes _fit() has already been called on the child nodes.

    # Calculate number of rows that will be present in B
    Brows = 0
    for point in reach_points:
      Brows += min(point.max_frequency, self._max_frequency)

    Bdim = self._basis_dimension
    print(f"Bdim = {Bdim}")
    B = np.zeros(shape=(Brows, Bdim**2))
    g = np.zeros(shape=(Brows, 1))
    L = np.zeros(shape=(Bdim**2, 1))
    U = np.zeros(shape=(Bdim**2, 1))

    # Construct B and g
    i = 0
    last_nonzero = -1
    for point in reach_points:
      probs = np.array(point.impressions) / self._impression_count_vector
      S_probs = probs[:self._left_child._publisher_count]
      T_probs = probs[self._left_child._publisher_count:]
      p_S = self._left_child.impression_probability(S_probs)
      p_T = self._right_child.impression_probability(T_probs)
      q_S = 1 - p_S
      q_T = 1 - p_T
      print(f"p_S = {p_S}, q_S = {q_S}, p_T = {p_T}, q_T = {q_T}")
      g_S = self._left_child.campaign_frequencies(S_probs, self._max_frequency)
      g_T = self._right_child.campaign_frequencies(T_probs, self._max_frequency)
      print(f"g_S = {g_S}")
      print(f"g_T = {g_T}")
      max_freq = min(point.max_frequency, self._max_frequency)
      print(f"max_freq = {max_freq}")
      for k in range(1, max_freq):
        g[i] = point.frequency(k) / self._total_audience_size
        for k_1 in range(k+1):
          k_2 = k - k_1
          g[i] -= g_S[k_1] * g_T[k_2]
          for l_1 in range(k_1, Bdim):
            for l_2 in range(k_2, Bdim):
              z = (
                  scipy.special.binom(l_1, k_1) * 
                  scipy.special.binom(l_2, k_2) *
                  p_S ** k_1 * q_S ** (l_1 - k_1) * 
                  p_T ** k_2 * q_T ** (l_2 - k_2))
              B[i][l_1 * Bdim + l_2] += z
              # print(f"B[{i}][{l_1 * Bdim + l_2} ( {k_1}:{l_1}; {k_2}:{l_2})] += {z}")
        if not np.all(B[i] == 0.0):
          last_nonzero = i
        i += 1
        # print(f"B[{i}] = {B[i]}")
    g = g[:(last_nonzero+1)]
    B = B[:(last_nonzero+1)]
    print(f"g = {g}")
    # print(f"B = {B}")

    # Construct lower and upper bounds for basis elements
    f_S = self._left_child.inventory_frequencies(max(self._max_frequency, Bdim))
    f_T = self._right_child.inventory_frequencies(max(self._max_frequency, Bdim))
    print(f"L.shape = {L.shape}")
    print(f"U.shape = {U.shape}")
    for k_1 in range(Bdim):
      for k_2 in range(Bdim):
        # print(f"k1={k_1}, k2={k_2}, ix={k_1*Bdim + k_2}")
        L[k_1 * Bdim + k_2] = f_S[k_1] * f_T[k_2]
        U[k_1 * Bdim + k_2] = 1 - f_S[k_1] * f_T[k_2]

    print(f"L = {L}")
    print(f"U = {U}")

    # Construct row sum and column sum constraints
    R = np.zeros(shape=(Bdim, Bdim**2))
    C = np.zeros(shape=(Bdim-1, Bdim**2))
    for k_1 in range(Bdim):
      for k_2 in range(Bdim):
        R[k_1, (k_1 * Bdim + k_2)] = 1.0

    for k_1 in range(Bdim-1):
      for k_2 in range(Bdim):
        C[k_1, (k_2 * Bdim + k_1)] = 1.0

    print(f"R = {R}")
    print(f"C = {C}")

    # Build full constraint matrix
    GM = matrix(np.concatenate([-np.identity(L.shape[0]),
                                np.identity(U.shape[0])]))
    Gv = matrix(np.concatenate([L, U]))

    AM = matrix(np.concatenate([R, C]))
    Av = matrix(np.zeros(shape=(R.shape[0] + C.shape[0], 1)))

    # Perform optimization
    P = matrix(np.matmul(B.T, B) + hlambda * np.identity(Bdim**2))
    q = matrix(-np.matmul(B.T, g))
    print(f"P = {P}")
    print(f"q = {q}")
    print(f"GM = {GM}")
    print(f"Gv = {Gv}")
    print(f"AM = {AM}")
    print(f"Av = {Av}")
    sol = qp(P, q, GM, Gv, AM, Av)
    # sol = qp(P, q, GM, Gv)

    # Copy optimization results into self._basis
    for k_1 in range(Bdim):
      for k_2 in range(Bdim):
        self._basis[k_1][k_2] = sol['x'][k_1 * Bdim + k_2]

    print(f"sol[x] = {sol['x']}")
    print(f"self._basis = {self._basis}")
