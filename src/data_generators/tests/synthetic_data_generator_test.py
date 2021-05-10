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
"""Tests for synthetic_data_generator.py."""

from absl.testing import absltest
from tempfile import TemporaryDirectory

from wfa_planning_evaluation_framework.data_generators.synthetic_data_generator import (
    SyntheticDataGenerator)
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.test_synthetic_data_design_config import TestSyntheticDataDesignConfig


class SyntheticDataGeneratorTest(absltest.TestCase):

  def test_synthetic_data_generator(self):
    with TemporaryDirectory() as d:
      generator = SyntheticDataGenerator(d, TestSyntheticDataDesignConfig)
      data_design = generator()
      self.assertEqual(data_design.count, 3)
      self.assertEqual(data_design.names, [
          "independent_homog_p=10_numpub=3_rs=0",
          "independent_homog_p=10_numpub=3_rs=1",
          "independent_homog_p=10_numpub=3_rs=2"
      ])


if __name__ == "__main__":
  absltest.main()
