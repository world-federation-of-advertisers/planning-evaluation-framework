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

from wfa_planning_evaluation_framework.data_generators.synthetic_data_design_generator import (
    SyntheticDataDesignGenerator,
)
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.test_synthetic_data_design_config import (
    TestSyntheticDataDesignConfig,
)
from wfa_planning_evaluation_framework.data_generators.test_synthetic_data_design_config2 import (
    TestSyntheticDataDesignConfig2,
)
from wfa_planning_evaluation_framework.data_generators.test_lhs_synthetic_data_design_config import (
    TestLHSSyntheticDataDesignConfig,
)


class SyntheticDataDesignGeneratorTest(absltest.TestCase):
    def validate_generated_test_data(self, generated_data_set):
        self.assertEqual(generated_data_set.publisher_count, 3)
        self.assertEqual(generated_data_set._data[0].max_reach, 1000)
        self.assertEqual(generated_data_set._data[1].max_reach, 707)
        # Because of rounding down we don't get exactly 500
        self.assertEqual(generated_data_set._data[2].max_reach, 499)

    def test_synthetic_data_generator_lhs_dataset(self):
        with TemporaryDirectory() as d:
            generator = SyntheticDataDesignGenerator(
                d, 1, TestLHSSyntheticDataDesignConfig
            )
            data_design = generator()
            self.assertEqual(data_design.count, 6)

    def test_synthetic_data_generator_single_dataset_single_publisher(self):
        with TemporaryDirectory() as d:
            generator = SyntheticDataDesignGenerator(
                d, 1, TestSyntheticDataDesignConfig2
            )
            data_design = generator()
            self.assertEqual(data_design.count, 1)
            # Random signatures at rs=xxx will change when any underliying operation
            # that uses a RandomState changes. Thus, this test will need update.
            # However, it is necessary to set these values to ensure deterministic
            # behavior.
            expected_names = [
                "num_publishers=1_largest_publisher_size=1000_largest_to_smallest_publisher_ratio=0.5_rs=11942",
            ]
            self.assertEqual(data_design.names, expected_names)
            generated_data_set = data_design.by_name(expected_names[0])
            self.assertEqual(generated_data_set.publisher_count, 1)
            self.assertEqual(generated_data_set._data[0].max_reach, 1000)

    def test_synthetic_data_generator_multiple_publishers(self):
        with TemporaryDirectory() as d:
            generator = SyntheticDataDesignGenerator(
                d, 1, TestSyntheticDataDesignConfig
            )
            data_design = generator()
            self.assertEqual(data_design.count, 3)
            # Random signatures at rs=xxx will change when any underliying operation
            # that uses a RandomState changes. Thus, this test will need update.
            # However, it is necessary to set these values to ensure deterministic
            # behavior.
            expected_names = [
                "num_publishers=3_largest_publisher_size=1000_largest_to_smallest_publisher_ratio=0.5_rs=34865",
                "num_publishers=3_largest_publisher_size=1000_largest_to_smallest_publisher_ratio=0.5_rs=46756",
                "num_publishers=3_largest_publisher_size=1000_largest_to_smallest_publisher_ratio=0.5_rs=62066",
            ]
            self.assertEqual(data_design.names, expected_names)
            [
                self.validate_generated_test_data(data_design.by_name(name))
                for name in expected_names
            ]


if __name__ == "__main__":
    absltest.main()
