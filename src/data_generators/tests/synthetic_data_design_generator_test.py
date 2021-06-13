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
import numpy as np
from tempfile import TemporaryDirectory

from wfa_planning_evaluation_framework.data_generators.synthetic_data_design_generator import (
    SyntheticDataDesignGenerator,)
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign

from wfa_planning_evaluation_framework.data_generators import lhs_data_design_example
from wfa_planning_evaluation_framework.data_generators import simple_data_design_example


class SyntheticDataDesignGeneratorTest(absltest.TestCase):

  def test_simple_design(self):
    simple_design = simple_data_design_example.generate_data_design_config(np.random.default_rng(seed=1))
    self.assertLen(list(simple_design), 27)

  def test_lhs_design(self):
    lhs_design = lhs_data_design_example.generate_data_design_config(np.random.default_rng(seed=1))
    self.assertLen(list(lhs_design), 10)

  def test_synthetic_data_generator_simple_design(self):
    with TemporaryDirectory() as d:
      data_design_generator = SyntheticDataDesignGenerator(
        d, 1, simple_data_design_example.__file__, False)
      data_design_generator()
      dd = DataDesign(d)
      self.assertEqual(dd.count, 27)

  def test_synthetic_data_generator_lhs_design(self):
    with TemporaryDirectory() as d:
      data_design_generator = SyntheticDataDesignGenerator(
        d, 1, lhs_data_design_example.__file__, False)
      data_design_generator()
      dd = DataDesign(d)
      self.assertEqual(dd.count, 10)
      

if __name__ == "__main__":
  absltest.main()
