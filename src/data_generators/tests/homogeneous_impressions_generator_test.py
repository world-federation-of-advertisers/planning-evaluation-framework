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
"""Tests for homogeneous_impression_generator.py."""

from absl.testing import absltest
from numpy.random import RandomState

from wfa_planning_evaluation_framework.data_generators.homogeneous_impression_generator import (
    HomogeneousImpressionGenerator,
)


class ImpressionGeneratorTest(absltest.TestCase):
    def test_homogeneous_impression_generator(self):
        generator = HomogeneousImpressionGenerator(3, 5, RandomState(1))
        generated_ids = generator()
        self.assertEqual(set(generated_ids), set([0, 1, 2]))
        self.assertLen(generated_ids, 11)


if __name__ == "__main__":
    absltest.main()
