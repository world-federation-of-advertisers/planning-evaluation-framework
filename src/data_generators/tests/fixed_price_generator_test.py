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
"""Tests for fixed_price_generator.py."""

from absl.testing import absltest

from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)


class FixedPriceGeneratorTest(absltest.TestCase):
    def test_fixed_price_generator(self):
        generator = FixedPriceGenerator(0.01)
        annotated_ids = generator([1, 1, 2])
        self.assertEqual(annotated_ids, [(1, 0.01), (1, 0.02), (2, 0.03)])


if __name__ == "__main__":
    absltest.main()
