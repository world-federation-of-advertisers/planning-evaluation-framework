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
"""Tests for publisher_data_file.py."""

from absl.testing import absltest
from numpy.random import RandomState
from tempfile import TemporaryDirectory

from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)
from wfa_planning_evaluation_framework.data_generators.homogeneous_impression_generator import (
    HomogeneousImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data_file import (
    PublisherDataFile,
)


class PublisherDataFileTest(absltest.TestCase):
    def test_properties(self):
        pdf = PublisherDataFile([(1, 0.01), (2, 0.02), (1, 0.04)], "test")
        self.assertEqual(pdf.impressions, 3)
        self.assertEqual(pdf.spend, 0.04)
        self.assertEqual(pdf.reach, 2)
        self.assertEqual(pdf.name, "test")

    def test_user_counts_by_impressions(self):
        pdf = PublisherDataFile([(1, 0.01), (1, 0.04), (2, 0.02)])
        self.assertEqual(pdf.user_counts_by_impressions(0), {})
        self.assertEqual(pdf.user_counts_by_impressions(1), {1: 1})
        self.assertEqual(pdf.user_counts_by_impressions(2), {1: 1, 2: 1})
        self.assertEqual(pdf.user_counts_by_impressions(3), {1: 2, 2: 1})

    def test_user_counts_by_spend(self):
        pdf = PublisherDataFile([(1, 0.01), (1, 0.04), (2, 0.02)])
        self.assertEqual(pdf.user_counts_by_spend(0), {})
        self.assertEqual(pdf.user_counts_by_spend(0.01), {1: 1})
        self.assertEqual(pdf.user_counts_by_spend(0.015), {1: 1})
        self.assertEqual(pdf.user_counts_by_spend(0.03), {1: 1, 2: 1})
        self.assertEqual(pdf.user_counts_by_spend(0.07), {1: 2, 2: 1})

    def test_read_and_write_publisher_data(self):
        pdf = PublisherDataFile([(1, 0.01), (2, 0.02), (1, 0.04)], "test")
        with TemporaryDirectory() as d:
            pdf.write_publisher_data_file(d)
            new_pdf = PublisherDataFile.read_publisher_data_file("{}/test".format(d))
            self.assertEqual(new_pdf.impressions, 3)
            self.assertEqual(new_pdf.spend, 0.04)
            self.assertEqual(new_pdf.reach, 2)
            self.assertEqual(new_pdf.name, "test")

    def test_generate_publisher_data_file(self):
        pdf = PublisherDataFile.generate_publisher_data_file(
            HomogeneousImpressionGenerator(3, 5), FixedPriceGenerator(0.01), "test"
        )
        self.assertEqual(pdf.reach, 3)
        self.assertEqual(pdf.name, "test")


if __name__ == "__main__":
    absltest.main()
