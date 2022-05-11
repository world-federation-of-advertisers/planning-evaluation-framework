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
"""Dirac mixture multi publisher model."""

from absl import logging
import numpy as np
import copy
from typing import Callable, List, Tuple, Union, Dict
from wfa_planning_evaluation_framework.models.reach_point import ReachPoint
from wfa_planning_evaluation_framework.models.reach_curve import ReachCurve
from wfa_planning_evaluation_framework.models.reach_surface import ReachSurface
from wfa_planning_evaluation_framework.models.dirac_mixture_single_publisher_model import (
    UnivariateMixedPoissonOptimizer,
)


class MultivariateMixedPoissonOptimizer:
    """Fit multivariate mixed Poisson distribution."""

    def __init__(
        self,
        observable_directions: np.ndarray,
        frequency_histograms_on_observable_directions: np.ndarray,
        prior_marginal_frequency_histograms: np.ndarray,
        ncomponents: int = 1000,
        rng: np.random.Generator = np.random.default_rng(0),
    ):
        """Construct an optimizer for multivariate mixed Poisson distribution.

        Args:
            observable_directions:  An <n * p> matrix where n = #training points,
                p = #pubs.

                Each training point has
                    input = an impression vector, and
                    output = the frequency historgam under the impression vector.
                The i-th row of the observable_directions matrix represents the
                impression vector of the i-th training point.
                In reality, the impression vector could be like [1e6, 2e6],
                which means 1e6 impressions at pub 1 and 2e6 impressions at pub 2.
                Computing with these raw, large numbers of impresssions may
                introduce numerical errors. As such, we standardize impression
                vectors as "directions".  For example, suppose
                - training point 1 has impression vector = [1e6, 2e6],
                - training point 2 has impression vector = [1e6, 0],
                - training point 3 has impression vector = [0, 2e6],
                then we say that
                - the "baseline impression vector" = [1e6, 2e6],
                - the 3 training points have observable_directions = [
                    [1, 1],
                    [1, 0],
                    [0, 1]
                ] respectively.
                Observable_directions is also a more compact representation
                Compared to impression vectors.

                As another example, observable_directions = [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1]
                ] means that we observe the frequency histogram on all the subsets
                of 3 publishers.  Rows 1-3 are single-pub observable_directions,
                rows 4-6 indicate 2-pub unions, and rows 7 indicates 3-pub union.

                As of Apr 2022, all the training points from halo_simulator will
                have a binary observable_direction (like the above example).
                We allow non-binary observable_direction in the future.
            frequency_histograms_on_observable_directions:  An <n * (F + 1)> matrix
                where n = #training points, F = maximum frequency.
                Its i-th row is the observed frequency histogram at the i-th
                row of observable_directions, i.e., the i-th training point.
            prior_marginal_frequency_histograms:  A <p * (F + 1)> matrix where
                p = #pubs, F = maximum frequency.

                Its i-th row is a marginal frequency histogram at the i-th
                publisher under the baseline impression (see the description of
                the `observable_directions` arg).  In other words, its rows are
                frequency_histograms at the single-pub observable_directions =
                [1, 0, ..., 0], [0, 1, 0, ..., 0], ..., [0, ..., 0, 1],
                respectively.  As of Apr 2022, halo_simulator generates a training
                point on each of these single pub observable_direction, so,
                this `prior_marginal_frequency_histograms` arg just consists of
                partial rows of `frequency_histograms_on_observable_directions`.
                It's not necessarily the case in the future though.  We require
                always specifying this `prior_marginal_frequency_histograms`
                arg so that the model is runnable regardless of what are provided
                by halo_simulator.

                Being called "prior_marginal_frequency_histograms", these
                frequency histograms are allowed to be estimates (say, from models
                or historical data), instead of actual observations.
            ncomponents:  Number of components in the Poisson mixture.
            rng:  Random Generator for the random sampling of high dimensional
                components.
        """
        self.observable_directions = observable_directions
        self.p = observable_directions.shape[1]

        for hist in frequency_histograms_on_observable_directions:
            UnivariateMixedPoissonOptimizer.validate_frequency_histogram(hist)
        self.observed_pmf_matrix = self.normalize_rows(
            frequency_histograms_on_observable_directions
        )
        self.max_freq = self.observed_pmf_matrix.shape[1] - 1

        for hist in prior_marginal_frequency_histograms:
            UnivariateMixedPoissonOptimizer.validate_frequency_histogram(hist)
        self.marginal_pmfs = self.normalize_rows(prior_marginal_frequency_histograms)
        self.check_dimension_campatibility()

        self.components = self.weighted_random_sampling(
            ncomponents, self.marginal_pmfs, rng
        )
        self.fitted = False

    def check_dimension_campatibility(self):
        """Check the compatibility of dimensions among the matrix attributes."""
        if self.marginal_pmfs.shape[0] != self.p:
            raise ValueError("Inconsistent number of publishers")
        if self.observable_directions.shape[0] != self.observed_pmf_matrix.shape[0]:
            raise ValueError("Inconsistent number of directions")
        # Minor note: for flexibity, we don't require self.marginal_pmfs to have
        # the same max_freq as self.observed_pmf_matrix.

    @classmethod
    def normalize_rows(cls, matrix: np.ndarray) -> np.ndarray:
        """Normalize a matrix so that each row sums up to 1.

        Args:
            matrix: Any m * n array for any m, n.

        Returns:
            An m * n array B such that B[i, :] = A[i, :] / sum(A[i, :])
            for any i, where A is the given matrix.
        """
        row_sums = matrix.sum(axis=1)
        # Using numpy broadcasting
        return matrix / row_sums[:, np.newaxis]

    @classmethod
    def weighted_random_sampling(
        cls,
        ncomponents: int,
        marginal_pmfs: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(0),
    ) -> np.ndarray:
        """Randomly sample components based on marginal pmfs.

        Each coordinate of the component is independently sampled based on
        the corresponding marginal pmf.

        Args:
            ncomponents:  Number of components to sample.
                (Precisely, we sample these many components plus a special
                component: the component of all zeros. This is to
                reflect the never-reached users.)
            marginal_pmfs:  Prior pmf of frequency at each publisher.
            rng:  Random generator.

        Returns:
            A <C * p> matrix where C = #components and p = #pubs.
            Each row is a component, i.e., a vector of Poisson means at all pubs.
        """
        p, max_f = marginal_pmfs.shape
        # For any frequency level 0 <= f <= <max frequency>,
        # and any pub i,
        # Pr(the i-th coordinate of a random component falls in [f, f + 1))
        # = marginal_pmfs[i] [f].
        # And with in [f, f + 1), the coordinate is uniformly distributed.
        # So the codes below first randomly choose an integer f according to the
        # marginal pmf, and then add rng.random() to make it a random float
        # in [f, f + 1).
        sample = np.array(
            [
                rng.choice(a=range(max_f), p=pmf, size=ncomponents)
                for pmf in marginal_pmfs
            ]
        ) + rng.random(size=(p, ncomponents))
        return np.hstack((np.zeros((p, 1)), sample)).transpose()

    @classmethod
    def obtain_pmf_matrix(
        cls,
        component_vector: np.ndarray,
        observable_directions: np.ndarray,
        max_freq: int,
    ) -> np.ndarray:
        """Obtain the pmfs on each observable direction of a component.

        Args:
            component_vector:  A length <p> vector where p = #pubs.
                A vector of Poisson means at all the pubs.  A component can be
                interpreted as a user group.  Its i-th coordinate means the
                average frequency at pub i of this user group.
            observable_directions:   An <n * p> matrix where n = #training points,
                p = #pubs.  Each row indicates the impression vector of a training
                point.  See the description of __init__ for more details.
            max_freq:  Maximum frequency.

        Returns:
            A <C * (F + 1)> matrix where C = #components.
            the i-th row is the pmf of the given component at the i-th direction
            of observable_directions.
        """
        projected_poisson_means = observable_directions.dot(component_vector)
        return np.array(
            [
                UnivariateMixedPoissonOptimizer.truncated_poisson_pmf_vec(
                    poisson_mean=mean, max_freq=max_freq
                )
                for mean in projected_poisson_means
            ]
        )

    def fit(self):
        """Fits a univariate mixed Poisson distribution."""
        if self.fitted:
            return
        self.component_pmf_matrices = [
            self.obtain_pmf_matrix(
                component_vector=component,
                observable_directions=self.observable_directions,
                max_freq=self.max_freq,
            )
            for component in self.components
        ]
        # self.ws saves the weights of each component
        self.ws = UnivariateMixedPoissonOptimizer.solve_optimal_weights(
            observed_arr=self.observed_pmf_matrix,
            component_arrs=self.component_pmf_matrices,
            distance=UnivariateMixedPoissonOptimizer.cross_entropy,
        )
        self.fitted = True

    def predict(
        self, hypothetical_direction: np.ndarray, customized_max_freq: int = None
    ) -> np.ndarray:
        """Predict frequency histogram of a hyperthetical campaign.

        Args:
            hypothetical_direction:  A length <p> vector.
                Please see the description of the `observable_directions` arg in
                `__init__()`, especially the concept of baseline impression vector.
                Suppose baseline impression vector = [1e6, 2e6, 3e6].  A
                hypothetical_direction = [0.5, 1, 2] means that we want to predict
                a hypothetical campaign with impression vector = [0.5 * 1e6,
                1 * 2e6, 2 * 3e6].
            customized_max_freq:  If specified, the predicted frequency
                histogram will be capped by this maximum frequency. It can be
                different from the maximum frequency of the training data.

        Returns:
            A length <F + 1> vector, where F is the maximum frequency.
            Its f-th coordinate is the pmf at frequency f of the hypothetical campiaign,
            for 0 <= f < = F.
        """
        if len(hypothetical_direction) != self.p:
            raise ValueError("Hypothetical direction does match number of publishers")
        if not self.fitted:
            self.fit()
        projected_components = [
            component.dot(hypothetical_direction) for component in self.components
        ]
        projected_component_pmfs = [
            UnivariateMixedPoissonOptimizer.truncated_poisson_pmf_vec(
                poisson_mean=c,
                max_freq=(
                    self.max_freq
                    if customized_max_freq is None
                    else customized_max_freq
                ),
            )
            for c in projected_components
        ]
        return sum([w * pmf for w, pmf in zip(self.ws, projected_component_pmfs)])


class DiracMixtureMultiPublisherModel(ReachSurface):
    """Dirac mixture multi publisher k+ model."""

    def __init__(
        self,
        reach_points: List[ReachPoint],
        reach_curves: List[ReachCurve] = None,
        single_publisher_reach_agreement: bool = True,
        ncomponents: int = None,
        rng: np.random.Generator = np.random.default_rng(0),
    ):
        """Constructs a Dirac mixture multi publisher model.

        Args:
            reach_points: A length <n> vector where n = #(training points).
                List of ReachPoints on which the model is to be trained.
            reach_curves: A length <p> list where p = #pubs.
                List of ReachCurves to be used in model fitting and prediction.
                Explicitly, it can
                    - force single publisher agreement if required
                    - possibly provide (additional) single publisher points
                        for training
                This argument is optional.  If this argument is None, then
                the multi pub model is trained purely based on discrete
                reach points, and the single pub models will be induced from
                this multi pub model.
            single_publisher_reach_agreement:  Specifies if we want to force
                single publisher agreements when reach_curves are given.
                In this class we have implemented a method to force single
                pub agreement for 1+ reach, but not for k+ reaches.  In other
                words, agreement in reach but not frequency.  As such, this
                argument is called "single_publisher_reach_agreement" instead
                of "single_publisher_agreement".
            ncomponents:  Number of components in the Poisson mixture.  If not
                specified, then follow the default choice min(5000, 200 * p**2).
            rng:  Random Generator for the random sampling of high dimensional
                components.
        """
        super().__init__(data=reach_points)
        self.p = len(reach_points[0].impressions)
        # We require at least a ReachPoint to have universe size.
        # We choose the common_universe_size to be the max universe size of the
        # given ReachPoints and reset each training point to have this common
        # universe size.
        sizes = [
            rp._universe_size for rp in self._data if rp._universe_size is not None
        ]
        if len(sizes) == 0:
            raise ValueError(
                "The model requires at least one ReachPoint to have universe size."
            )
        self.common_universe_size = max(sizes)
        for rp in self._data:
            rp._universe_size = self.common_universe_size
        if reach_curves is None:
            self.single_publisher_reach_agreement = False
        else:
            self.single_publisher_reach_agreement = single_publisher_reach_agreement
            self.ensure_compatible_num_publishers(reach_curves, reach_points)
            self.reach_curves = copy.deepcopy(reach_curves)
        self.ncomponents = (
            min(5000, 200 * self.p ** 2) if ncomponents is None else ncomponents
        )
        self.rng = rng
        self._fit_computed = False

    @staticmethod
    def ensure_compatible_num_publishers(
        reach_curves: List[ReachCurve], reach_points: List[ReachPoint]
    ):
        """Check if the number of publishers match in different inputs."""
        p = len(reach_curves)
        for rp in reach_points:
            if len(rp.impressions) != p:
                raise ValueError(
                    "Number of publishers in reach point does not match length of reach curves"
                )

    @staticmethod
    def select_single_publisher_points(
        reach_points: List[ReachPoint],
    ) -> Dict[int, List[ReachPoint]]:
        """Select reach points that describe the reach on a single publisher.

        Args:
            reach_points:  Any list of ReachPoints.

        Returns:
            A dictionary of which the value of key = i is the sub-list of
            reach points that involve only pub i.
        """
        single_pub_points = {}
        p = len(reach_points[0].impressions)
        for rp in reach_points:
            involved_pubs = [i for i in range(p) if rp.impressions[i] > 0]
            if len(involved_pubs) == 1:
                pub = involved_pubs[0]
                single_pub_points[pub] = single_pub_points.get(pub, []) + [rp]
        return single_pub_points

    @classmethod
    def obtain_marginal_frequency_histograms(
        cls, reach_points: List[ReachPoint]
    ) -> Tuple[np.ndarray]:
        """One way to obtain prior_marginal_frequency_histograms as an input of MultivariateMixedPoissonOptimizer.

        This and the next two methods are for translating the given
        reach points to the inputs of MultivariateMixedPoissonOptimizer.

        Args:
            reach_points:  Any list of ReachPoint.

        Return:
            A length <2> tuple
            (prior_marginal_frequency_histograms, baseline_impression_vector).

            This method assumes that each single pub frequency histogram is
            already observed, and collected in the given ReachPoints (self._data).
            This method extracts the single pub frequeny histograms and put them in
            a matrix, as the `prior_marginal_frequency_histograms` argument of
            MultivariateMixedPoissonOptimizer.

            The impressions at these single publisher points are also returned.
            They will be used as baseline_impression_vector (see the description
            of MultivariateMixedPoissonOptimizer).
        """
        single_pub_points = cls.select_single_publisher_points(reach_points)
        p = len(reach_points[0].impressions)
        prior_marginal_frequency_histograms = []
        baseline_impression_vector = []
        for i in range(p):
            if i not in single_pub_points:
                raise AssertionError(f"Cannot find single pub reach point for pub {i}")
            rp = single_pub_points[i][0]
            prior_marginal_frequency_histograms.append(rp.zero_included_histogram)
            baseline_impression_vector.append(rp.impressions[i])
        return (
            np.array(prior_marginal_frequency_histograms),
            np.array(baseline_impression_vector),
        )

    @staticmethod
    def obtain_observable_directions(
        reach_points: List[ReachPoint], baseline_impression_vector: np.ndarray
    ) -> np.ndarray:
        """Obtain observable_directions as an input of MultivariateMixedPoissonOptimizer.

        As mentioned in the description of MultivariateMixedPoissonOptimizer.__init__(),
        the observable_directions are obtained by standardizing the impression vector of
        each training point with the baseline_impression_vector.
        """
        return (
            np.array([list(rp.impressions) for rp in reach_points])
            / baseline_impression_vector[np.newaxis, :]
        )

    @classmethod
    def obtain_frequency_histograms_on_observable_directions(
        cls, reach_points: List[ReachPoint]
    ) -> np.ndarray:
        """Obtain frequency_histograms_on_observable_directions as an input of MultivariateMixedPoissonOptimizer."""
        return np.array([rp.zero_included_histogram for rp in reach_points])

    def _fit(self):
        if self._fit_computed:
            return
        prior_hists, self.baseline_imps = self.obtain_marginal_frequency_histograms(
            reach_points=self._data
        )
        obs_dirs = self.obtain_observable_directions(
            reach_points=self._data, baseline_impression_vector=self.baseline_imps
        )
        hists_on_obs_dirs = self.obtain_frequency_histograms_on_observable_directions(
            reach_points=self._data
        )
        while self.ncomponents > 0:
            self.optimizer = MultivariateMixedPoissonOptimizer(
                observable_directions=obs_dirs,
                frequency_histograms_on_observable_directions=hists_on_obs_dirs,
                prior_marginal_frequency_histograms=prior_hists,
                ncomponents=self.ncomponents,
                rng=self.rng,
            )
            try:
                self.optimizer.fit()
                break
            except Exception as inst:
                # There is a tiny chance of exception when cvxpy mistakenly
                # thinks the problem is non-convex due to numerical errors.
                # If this occurs, it is likely that we have a large number
                # of components.  In this case, try reducing the number of
                # components.
                logging.vlog(1, f"Optimizer failure: {inst}")
                self.ncomponents = self.ncomponents // 2
                continue
        self._fit_computed = True

    def by_impressions_no_single_pub_reach_agreement(
        self, impressions: List[int], max_frequency: int = 1
    ) -> ReachPoint:
        """Predicts reach as a function of impressions, without single pub reach agreement.

        Args:
            impressions:  A length <p> list where p = #pubs.  The impression vector for
                which we want to predict.
            max_frequency: int, specifies the number of frequencies for which reach
                will be reported.

        Returns:
            A ReachPoint specifying the predicted reach for this number of impressions.
        """
        self._fit()
        hypothetical_direction = np.array(
            [
                hypothetical / baseline
                for hypothetical, baseline in zip(impressions, self.baseline_imps)
            ]
        )
        predicted_relative_freq_hist = self.optimizer.predict(
            hypothetical_direction=hypothetical_direction,
            customized_max_freq=max_frequency,
        )
        relative_kplus_reaches_from_zero = np.cumsum(
            predicted_relative_freq_hist[::-1]
        )[::-1]
        kplus_reaches = (
            (self.common_universe_size * relative_kplus_reaches_from_zero[1:])
            .round(0)
            .astype("int32")
        )
        return ReachPoint(impressions, kplus_reaches)

    @staticmethod
    def backsolve_impression(
        curve: Callable[[int], int], target_reach: int, starting_impression: int
    ) -> int:
        """From a reach curve, backsolves the impression from a target reach.

        The backsolving is done by a customized bisection search.
        This method is to be used in the later
        `by_impressions_with_single_pub_reach_agreement` method.

        Args:
            curve:  A reach curve, i.e., function from impression (short for #impressions)
                to reach.  Here reach only means the 1+ reach.
            target_reach:  The target reach.
            starting_impression:  Starting point when searching the impression.

        Returns:
            A value of impression such that curve(impression) equals to
            target_reach.
        """
        if target_reach == 0:
            return 0
        probe = max(2, starting_impression)
        count = 0
        while curve(probe) < target_reach:
            probe *= 2
            count += 1
            if count >= 10:
                raise ValueError(
                    "Cannot achieve target reach.  Does the given reach curve has a upper bound?"
                )
        right = probe
        left = 0 if count == 0 else int(probe / 2)
        # Loop termination criterion:  right - left <= 1, or either left or right hits
        # the target reach.
        while not (
            curve(left) == target_reach
            or curve(right) == target_reach
            or right - left <= 1
        ):
            mid = (left + right) / 2
            mid = int(round(mid))
            if curve(mid) < target_reach:
                left = mid
            else:
                right = mid
        if abs(curve(left) - target_reach) < abs(curve(right) - target_reach):
            return left
        return right

    @staticmethod
    def induced_single_pub_curve(
        surface: Callable[[List[int]], int], num_pubs: int, pub_index: int
    ) -> Callable[[int], int]:
        """Returns a induced single pub reach curve from a multi pub reach surface.

        This is another method to be used in the later
        `by_impressions_with_single_pub_reach_agreement` method.

        Args:
            surface:  A function from a multi pub impression vector to reach.
            num_pubs:  Number of publishers.
            pub_index:  We want the induced reach curve on this pub.

        Returns:
            The induced reach curve on the given `pub_index`.  That is, the function
            of reach on the impression at this given pub, while having zero
            impressions at all other pubs.
        """

        def curve(num_impressions: int) -> int:
            impressions = [0] * num_pubs
            impressions[pub_index] = num_impressions
            return surface(impressions)

        return curve

    def by_impressions_with_single_pub_reach_agreement(
        self, impressions: List[int], max_frequency: int = 1
    ) -> ReachPoint:
        """Predicts reach as a function of impressions, with single pub reach agreement.

        With single pub reach agreement, the induced reach curve (see description of the
        previous method) of the predicted reach surface aligns with the given reach curve
        at each pub.
        This is done by first predicting without single pub reach agreement and then
        applying an adjustment.  Mathematically, let
        - S(x, y) be the predicted surface without single pub reach agreement,
        - s1(x) = S(x, 0), s2(y) = S(0, y) be the induced curves of S,
        - c1(x), c2(y) be the given curves,
        Then the adjusted surface is
        S'(x, y) = S(s1^{-1}(c1(x)), s2^{-1}(c2(y))).
        It can be seen that S'(x, 0) = c1(x), S'(0, y) = c2(y), thus single pub reach
        agreement.  (Note that the agreement is on 1+ but not necessarily k+ reach.
        Per confirmation in a WFA meeting, k+ reach agreement is not a requirement.)

        Args:
            impressions:  A length <p> list where p = #pubs.  The impression vector for
                which we want to predict.
            max_frequency: int, specifies the number of frequencies for which reach
                will be reported.

        Returns:
            A ReachPoint specifying the predicted reach for this number of impressions.
        """
        self._fit()
        target_single_pub_reaches = [
            self.reach_curves[i].by_impressions([impressions[i]]).reach(1)
            for i in range(self.p)
        ]
        reach_surface = lambda x: self.by_impressions_no_single_pub_reach_agreement(
            x, max_frequency
        ).reach(1)
        induced_single_pub_curves = [
            self.induced_single_pub_curve(
                surface=reach_surface, num_pubs=self.p, pub_index=i
            )
            for i in range(self.p)
        ]
        adjusted_impressions = [
            self.backsolve_impression(
                curve=induced_single_pub_curves[i],
                target_reach=target_single_pub_reaches[i],
                starting_impression=impressions[i],
            )
            for i in range(self.p)
        ]
        return self.by_impressions_no_single_pub_reach_agreement(
            impressions=adjusted_impressions, max_frequency=max_frequency
        )

    def by_impressions(
        self, impressions: List[int], max_frequency: int = 1
    ) -> ReachPoint:
        """Predicts reach either with or without single pub reach agreement."""
        if self.single_publisher_reach_agreement:
            return self.by_impressions_with_single_pub_reach_agreement(
                impressions, max_frequency
            )
        return self.by_impressions_no_single_pub_reach_agreement(
            impressions, max_frequency
        )
