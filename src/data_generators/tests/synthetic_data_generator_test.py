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
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.data_design_parameters import (
    DataDesignParameters, DataSetParameters, OverlapModelParameters,
    ImpressionGeneratorParameters, PricingGeneratorParameters)
from wfa_planning_evaluation_framework.data_generators.pricing_generator import PricingGenerator
from wfa_planning_evaluation_framework.data_generators.impression_generator import ImpressionGenerator
from wfa_planning_evaluation_framework.data_generators.publisher_data import PublisherData


class SyntheticDataGeneratorTest(absltest.TestCase):

  def test_synthetic_data_generator(self):
    with TemporaryDirectory() as d:
      generator = SyntheticDataGenerator(
          DataDesignParameters(
              num_reps=2,
              data_set_parameters=DataSetParameters(
                  num_publishers=3,
                  largest_publisher_size=100,
                  publisher_size_decay_rate=0.9,
                  pricing_generator_params=PricingGeneratorParameters(
                      pricing_generator="fixed", fixed_price_cost=0.5),
                  impression_generator_params=ImpressionGeneratorParameters(
                      impression_generator="homogenous",
                      num_users=10,
                      homogenous_lambda=0.1),
              overlap_model_params=OverlapModelParameters(overlap_model="test")),
              output_folder=d))
      data_design = generator()
      self.assertEqual(data_design.count, 2)
      self.assertEqual(data_design.names, ["homog_p=10_rep=3_rs=0", "homog_p=10_rep=3_rs=1"])


if __name__ == "__main__":
  absltest.main()
