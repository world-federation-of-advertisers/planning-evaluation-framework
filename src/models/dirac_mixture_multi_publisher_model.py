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
from time import time
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
        ncomponents: int = 1000,
        dilution: float = 0,
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
                For each frequency histogram H, H[0] indicates the non-reach,
                H[f] indicates the reach at frequency f for f = 1, ..., F - 1,
                and H[F] indicates the F+ reach.
            ncomponents:  Number of components in the Poisson mixture.
            dilution:  An arg to control the weights when sampling components.
                The same argument as in the in_bound_grid method of the
                UnivariateMixedPoissonOptimizer class for single pub model.
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
        if self.observable_directions.shape[0] != self.observed_pmf_matrix.shape[0]:
            raise ValueError("Inconsistent number of directions")
        self.max_freq = self.observed_pmf_matrix.shape[1] - 1
        self.rng = rng
        self.ncomponents = ncomponents
        self.dilution = dilution
        self.components = self.weighted_sampling_from_marginals()
        self.fitted = False

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

    def find_single_pub_pmfs(self) -> List[int]:
        """Find all the indices in self.observable_directions that indicate a single pub.

        As of June 2022, restrict the definition of a single pub direction as:
        vectors like [1, 0, 0], [0, 1, 0] or [0, 0, 1], i.e., binary vectors that
        are 1 at exactly one coordinate.

        (Note: Later, if we have multi campaigns to train the model, we might have
        directions such as [0.6, 0, 0] which might be also considered as single
        pub directions.  In that case, we may be multiple single pub directions
        at one pub which adds the complexity.  We will refactor the codes if that
        happens.)

        Returns:
            A length <p> list where p = #pubs.
            For each 1 <= i <= p, its i-th element is a length <F + 1> array, where
            F = max freq, which indicates the frequency pmf at a single pub direction
            at pub i.

        Raises:
            AssertionError:  As of June 2022, assume that self.observable_directions
                include a single pub direction for every pub.  Otherwise, raise
                an error.
        """
        n, p = self.observable_directions.shape
        single_pub_direction_indices = {}
        numerical_err = 1e-6
        for k in range(n):
            dir = self.observable_directions[k]
            involved_pubs = np.where(dir > numerical_err)[0]
            if len(involved_pubs) == 1:
                pub = involved_pubs[0]
                if abs(dir[pub] - 1) < numerical_err:
                    single_pub_direction_indices[pub] = k
        if len(single_pub_direction_indices) < p:
            raise AssertionError(
                "Not every pub has an observable single pub direction."
            )
        return [
            self.observed_pmf_matrix[single_pub_direction_indices[i]] for i in range(p)
        ]

    def weighted_sampling_from_marginals(self) -> np.ndarray:
        """Randomly sample components based on the given single pub points.

        Each component of the Dirac mixture model is a length <p> vector where
        p = #pubs.  In this class, we randomly sample a bunch of components
        from the p-dimensional space.  The assumption is that when the number
        of components is large enough, randomly sampled components with optimized
        weights can well approximate the components in the true model.

        In the Dirac mixture single pub model, we sampled components using
        weighted grid search.  The main idea was:  A Poisson distribution has
        high pmf around its mean parameter.  So, if we observe a high pmf at a
        frequency level, we tend to draw more Poisson components around it.

        Following the same idea, we also do weighted sampling of components in the
        multi pub model.  However, unlike the single pub case, in the multi pub
        case we do not have straightforward weights for sampling, since we do
        not observe the joint distribution of cross-pub frequency.  So, as a proxy,
        instead of directly sampling components in the high dimensional space, we
        independently sample each coordinate of the component according to the
        observed single pub distributions, like what we did in the single pub.

        This approach assumes that we observe the single pub distribution at each
        pub.  In this class, these single pub distributions can be obtained by
        identifying the single pub directions in self.observable_directions.

        Returns:
            A <C * p> matrix where C = #components and p = #pubs.
            Each row is a component, i.e., a vector of Poisson means at all pubs.
        """
        marginal_pmfs = self.find_single_pub_pmfs()
        # We are sampling from a mixture of the independent combination of marginal
        # pmfs and the uniform distribution, where the weight given to the uniform
        # distribution is given by self.dilution.
        water = np.array([self.dilution / (self.max_freq + 1)] * (self.max_freq + 1))
        diluted_marginal_pmfs = [
            pmf * (1 - self.dilution) + water for pmf in marginal_pmfs
        ]
        sample = np.array(
            [
                self.rng.choice(
                    a=range(self.max_freq + 1), p=pmf, size=self.ncomponents
                )
                for pmf in diluted_marginal_pmfs
            ]
        ).astype("float")
        # The above sample are integers.  We further add uniformly distributed
        # numbers in (0, 1) to make the component coordinates decimals.
        sample += self.rng.random(size=(self.p, self.ncomponents))
        # Always include the zero component to reflect non-reach.
        return np.hstack((np.zeros((self.p, 1)), sample)).transpose()

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
        """Predict frequency histogram of a hypothetical campaign.

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
        dilution: float = 0,
        rng: np.random.Generator = np.random.default_rng(0),
        universe_size: int = None,
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
            dilution:  An arg to control the weights when sampling components.
                The same argument as in the in_bound_grid method of the
                UnivariateMixedPoissonOptimizer class for single pub model.
            rng:  Random Generator for the random sampling of high dimensional
                components.
        """
        super().__init__(data=reach_points)
        self.p = len(reach_points[0].impressions)
        # We require at least a ReachPoint to have universe size.
        # We choose the common_universe_size to be the max universe size of the
        # given ReachPoints and reset each training point to have this common
        # universe size.
        if universe_size is None:
            sizes = [
                rp._universe_size for rp in self._data if rp._universe_size is not None
            ]
            if len(sizes) == 0:
                raise ValueError(
                    "The model requires at least one ReachPoint to have universe size."
                )
            self.common_universe_size = max(sizes)
        else:
            self.common_universe_size = universe_size
        for rp in self._data:
            rp._universe_size = self.common_universe_size
        if reach_curves is None:
            self.single_publisher_reach_agreement = False
            self._reach_curves = None
        else:
            self.single_publisher_reach_agreement = single_publisher_reach_agreement
            self.ensure_compatible_num_publishers(reach_curves, reach_points)
            self._reach_curves = copy.deepcopy(reach_curves)
        self.ncomponents = (
            min(5000, 200 * self.p**2) if ncomponents is None else ncomponents
        )
        self.dilution = dilution
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
    def obtain_observable_directions(
        reach_points: List[ReachPoint],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Obtain observable_directions as an input of MultivariateMixedPoissonOptimizer.

        Args:
            reach_point:  Any list of ReachPoint.

        Returns:
            A tuple (baseline_impression_vector, observable_directions).
            - baseline_impression_vector is a length <p> array where p = #pubs.
                baseline_impression_vector[i] equals the maximum number of impressions
                at pub i among the given reach_points.
            - observable_directions is a <n * p> matrix where n = #ReachPoint in the given
                list, and p = #pubs.  Its k-th row the observable direction of the k-th
                ReachPoint, i.e., the impression vector at this ReachPoint divided by
                the baseline_impression_vector.
            See the description of MultivariateMixedPoissonOptimizer.__init__() for why
            we convert impression vectors to observable directions.
        """
        raw_impression_vectors = np.array([list(rp.impressions) for rp in reach_points])
        # Construct baseline_impression_vector by selecting the maximum num_impressions
        # at each pub
        baseline_impression_vector = np.max(raw_impression_vectors, axis=0)
        return (
            baseline_impression_vector,
            raw_impression_vectors / baseline_impression_vector[np.newaxis, :],
        )

    def _fit(self):
        if self._fit_computed:
            return
        self.baseline_imps, observable_dirs = self.obtain_observable_directions(
            self._data
        )
        hists_on_observable_dirs = np.array(
            [rp.zero_included_histogram for rp in self._data]
        )
        # Add single pub training curves from reach curves
        if self._reach_curves is not None:
            for i in range(self.p):
                direction = [0] * self.p
                direction[i] = 1
                observable_dirs = np.vstack((observable_dirs, direction))
                rp = self._reach_curves[i].by_impressions(
                    impressions=[self.baseline_imps[i]],
                    max_frequency=self._data[0].max_frequency,
                )
                rp._universe_size = self.common_universe_size
                hists_on_observable_dirs = np.vstack(
                    (hists_on_observable_dirs, rp.zero_included_histogram)
                )
        while self.ncomponents > 0:
            self.optimizer = MultivariateMixedPoissonOptimizer(
                observable_directions=observable_dirs,
                frequency_histograms_on_observable_directions=hists_on_observable_dirs,
                ncomponents=self.ncomponents,
                dilution=self.dilution,
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
        left = 0 if count == 0 else probe // 2
        while left < right - 1:
            mid = left + (right - left) // 2
            if curve(mid) == target_reach:
                return mid
            elif curve(mid) < target_reach:
                left = mid
            else:
                right = mid
        if target_reach - curve(left) < curve(right) - target_reach:
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
        t1 = time()
        self._fit()
        t2 = time()
        print(f'1-2: {round(t2 - t1, 1)} seconds.')
        target_single_pub_reaches = [
            self._reach_curves[i].by_impressions([impressions[i]]).reach(1)
            for i in range(self.p)
        ]
        t3 = time()
        print(f'2-3: {round(t3 - t2, 1)} seconds.')
        reach_surface = lambda x: self.by_impressions_no_single_pub_reach_agreement(
            x, max_frequency
        ).reach(1)
        t4 = time()
        print(f'3-4: {round(t4 - t3, 1)} seconds.')
        induced_single_pub_curves = [
            self.induced_single_pub_curve(
                surface=reach_surface, num_pubs=self.p, pub_index=i
            )
            for i in range(self.p)
        ]
        t5 = time()
        print(f'4-5: {round(t5 - t4, 1)} seconds.')
        adjusted_impressions = [
            self.backsolve_impression(
                curve=induced_single_pub_curves[i],
                target_reach=target_single_pub_reaches[i],
                starting_impression=impressions[i],
            )
            for i in range(self.p)
        ]
        t6 = time()
        print(f'5-6: {round(t6 - t5, 1)} seconds.')
        surface = self.by_impressions_no_single_pub_reach_agreement(
            impressions=adjusted_impressions, max_frequency=max_frequency
        )
        t7 = time()
        print(f'6-7: {round(t7 - t6, 1)} seconds.')
        return surface

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

    def evaluate_single_pub_kplus_reach_agreement(
        self,
        scaling_factor_choices: List[float] = [0.5, 0.75, 1, 1.5, 2],
        max_frequency: int = 1,
    ) -> Dict[float, Dict[str, List[float]]]:
        if not self.single_publisher_reach_agreement:
            return {}
        metrics = {}
        for scaling_factor in scaling_factor_choices:
            metrics[scaling_factor] = {}
            single_pub_model_predictions = [
                curve.by_impressions(scaling_factor * imp, max_frequency)
                for curve, imp in zip(self._reach_curves, self.baseline_imps)
            ]
            multi_pub_model_predictions = []
            for i in range(self.p):
                imps = [0] * self.p
                imps[i] = scaling_factor * self.baseline_imps[i]
                multi_pub_model_predictions.append(
                    self.by_impressions_with_single_pub_reach_agreement(
                        imps, max_frequency
                    )
                )
            relative_differences = np.array(
                [
                    # if x = 0 but y != 0, treat the relative error as 100%
                    [abs(y - x) / x if x > 0 else 1 for x, y in zip(a, b)]
                    for a, b in zip(
                        single_pub_model_predictions, multi_pub_model_predictions
                    )
                ]
            )
            metrics[scaling_factor]["mean"] = np.mean(relative_differences, axis=0)
            metrics[scaling_factor]["q90"] = np.quantile(
                relative_differences, 0.9, axis=0
            )
            metrics[scaling_factor]["max"] = np.max(relative_differences, axis=0)
        return metrics
