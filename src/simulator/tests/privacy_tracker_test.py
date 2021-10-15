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
"""Tests for privacy_tracker.py."""

from absl.testing import absltest
from absl.testing import parameterized

from wfa_planning_evaluation_framework.simulator import privacy_tracker


class PrivacyTrackerTest(parameterized.TestCase):
    def test_empty_object(self):
        tracker = privacy_tracker.PrivacyTracker()
        self.assertEqual(tracker.privacy_consumption.epsilon, 0)
        self.assertEqual(tracker.privacy_consumption.delta, 0)
        self.assertEqual(tracker.mechanisms, [])

    def test_single_event(self):
        tracker = privacy_tracker.PrivacyTracker()
        tracker.append(
            privacy_tracker.NoisingEvent(
                privacy_tracker.PrivacyBudget(0.1, 0.01),
                privacy_tracker.DP_NOISE_MECHANISM_LAPLACE,
                {"sensitivity": 1},
            )
        )
        self.assertEqual(tracker.privacy_consumption.epsilon, 0.1)
        self.assertEqual(tracker.privacy_consumption.delta, 0.01)
        self.assertEqual(
            tracker.mechanisms, [privacy_tracker.DP_NOISE_MECHANISM_LAPLACE]
        )

    def test_two_events(self):
        tracker = privacy_tracker.PrivacyTracker()
        tracker.append(
            privacy_tracker.NoisingEvent(
                privacy_tracker.PrivacyBudget(0.1, 0.01),
                privacy_tracker.DP_NOISE_MECHANISM_LAPLACE,
                {"sensitivity": 1},
            )
        )
        tracker.append(
            privacy_tracker.NoisingEvent(
                privacy_tracker.PrivacyBudget(0.2, 0.015),
                privacy_tracker.DP_NOISE_MECHANISM_GAUSSIAN,
                {},
            )
        )
        self.assertAlmostEqual(tracker.privacy_consumption.epsilon, 0.3)
        self.assertAlmostEqual(tracker.privacy_consumption.delta, 0.025)
        self.assertEqual(
            tracker.mechanisms,
            [
                privacy_tracker.DP_NOISE_MECHANISM_LAPLACE,
                privacy_tracker.DP_NOISE_MECHANISM_GAUSSIAN,
            ],
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "strictly_not_contained",
            "sampling_bucket": 0.7,
            "expected_epsilon": 0,
            "expected_delta": 0,
        },
        {
            "testcase_name": "strictly_contained",
            "sampling_bucket": 0.2,
            "expected_epsilon": 0.1,
            "expected_delta": 0.01,
        },
        {
            "testcase_name": "border_starting_point",
            "sampling_bucket": 0.1,
            "expected_epsilon": 0.1,
            "expected_delta": 0.01,
        },
        {
            "testcase_name": "border_ending_point",
            "sampling_bucket": 0.6,
            "expected_epsilon": 0,
            "expected_delta": 0,
        },
    )
    def test_privacy_consumption_for_sampling_bucket_single_event(
        self, sampling_bucket, expected_epsilon, expected_delta
    ):
        tracker = privacy_tracker.PrivacyTracker()
        tracker.append(
            privacy_tracker.NoisingEvent(
                privacy_tracker.PrivacyBudget(0.1, 0.01),
                privacy_tracker.DP_NOISE_MECHANISM_LAPLACE,
                {},
                privacy_tracker.SamplingBucketIndices(0.1, 0.5),
            )
        )
        bucket_budget_consumption = tracker.privacy_consumption_for_sampling_bucket(
            sampling_bucket
        )
        self.assertAlmostEqual(bucket_budget_consumption.epsilon, expected_epsilon)
        self.assertAlmostEqual(bucket_budget_consumption.delta, expected_delta)

    @parameterized.named_parameters(
        {
            "testcase_name": "strictly_not_contained",
            "sampling_bucket": 0.4,
            "expected_epsilon": 0,
            "expected_delta": 0,
        },
        {
            "testcase_name": "strictly_contained_before_wrap_around",
            "sampling_bucket": 0.8,
            "expected_epsilon": 0.1,
            "expected_delta": 0.01,
        },
        {
            "testcase_name": "strictly_contained_after_wrap_around",
            "sampling_bucket": 0.2,
            "expected_epsilon": 0.1,
            "expected_delta": 0.01,
        },
        {
            "testcase_name": "border_starting_point",
            "sampling_bucket": 0.7,
            "expected_epsilon": 0.1,
            "expected_delta": 0.01,
        },
        {
            "testcase_name": "border_ending_point",
            "sampling_bucket": 0.3,
            "expected_epsilon": 0,
            "expected_delta": 0,
        },
    )
    def test_privacy_consumption_for_sampling_bucket_single_event_wrap_around(
        self, sampling_bucket, expected_epsilon, expected_delta
    ):
        tracker = privacy_tracker.PrivacyTracker()
        tracker.append(
            privacy_tracker.NoisingEvent(
                privacy_tracker.PrivacyBudget(0.1, 0.01),
                privacy_tracker.DP_NOISE_MECHANISM_LAPLACE,
                {},
                privacy_tracker.SamplingBucketIndices(0.7, 0.6),
            )
        )
        bucket_budget_consumption = tracker.privacy_consumption_for_sampling_bucket(
            sampling_bucket
        )
        self.assertAlmostEqual(bucket_budget_consumption.epsilon, expected_epsilon)
        self.assertAlmostEqual(bucket_budget_consumption.delta, expected_delta)

    @parameterized.named_parameters(
        {
            "testcase_name": "not_contained",
            "sampling_bucket": 0.05,
            "expected_epsilon": 0,
            "expected_delta": 0,
        },
        {
            "testcase_name": "contained_in_one",
            "sampling_bucket": 0.625,
            "expected_epsilon": 0.2,
            "expected_delta": 0.015,
        },
        {
            "testcase_name": "contained_in_both",
            "sampling_bucket": 0.5,
            "expected_epsilon": 0.3,
            "expected_delta": 0.025,
        },
    )
    def test_privacy_consumption_for_sampling_bucket_two_events(
        self, sampling_bucket, expected_epsilon, expected_delta
    ):
        tracker = privacy_tracker.PrivacyTracker()
        tracker.append(
            privacy_tracker.NoisingEvent(
                privacy_tracker.PrivacyBudget(0.1, 0.01),
                privacy_tracker.DP_NOISE_MECHANISM_LAPLACE,
                {},
                privacy_tracker.SamplingBucketIndices(0.1, 0.5),
            )
        )
        tracker.append(
            privacy_tracker.NoisingEvent(
                privacy_tracker.PrivacyBudget(0.2, 0.015),
                privacy_tracker.DP_NOISE_MECHANISM_GAUSSIAN,
                {},
                privacy_tracker.SamplingBucketIndices(0.35, 0.3),
            )
        )
        bucket_budget_consumption = tracker.privacy_consumption_for_sampling_bucket(
            sampling_bucket
        )
        self.assertAlmostEqual(bucket_budget_consumption.epsilon, expected_epsilon)
        self.assertAlmostEqual(bucket_budget_consumption.delta, expected_delta)

    @parameterized.named_parameters(
        {
            "testcase_name": "no_intersection",
            "smallest_index_first_event": 0.1,
            "smallest_index_second_event": 0.3,
            "expected_epsilon": 0.2,
            "expected_delta": 0.015,
        },
        {
            "testcase_name": "sharing_starting_and_end_point",
            "smallest_index_first_event": 0.1,
            "smallest_index_second_event": 0.2,
            "expected_epsilon": 0.2,
            "expected_delta": 0.015,
        },
        {
            "testcase_name": "proper_intersection",
            "smallest_index_first_event": 0.1,
            "smallest_index_second_event": 0.15,
            "expected_epsilon": 0.3,
            "expected_delta": 0.025,
        },
    )
    def test_privacy_consumption_two_events_with_sampling_buckets(
        self,
        smallest_index_first_event,
        smallest_index_second_event,
        expected_epsilon,
        expected_delta,
    ):
        tracker = privacy_tracker.PrivacyTracker()
        tracker.append(
            privacy_tracker.NoisingEvent(
                privacy_tracker.PrivacyBudget(0.1, 0.01),
                privacy_tracker.DP_NOISE_MECHANISM_LAPLACE,
                {},
                privacy_tracker.SamplingBucketIndices(smallest_index_first_event, 0.1),
            )
        )
        tracker.append(
            privacy_tracker.NoisingEvent(
                privacy_tracker.PrivacyBudget(0.2, 0.015),
                privacy_tracker.DP_NOISE_MECHANISM_GAUSSIAN,
                {},
                privacy_tracker.SamplingBucketIndices(smallest_index_second_event, 0.1),
            )
        )
        self.assertAlmostEqual(tracker.privacy_consumption.epsilon, expected_epsilon)
        self.assertAlmostEqual(tracker.privacy_consumption.delta, expected_delta)


if __name__ == "__main__":
    absltest.main()
