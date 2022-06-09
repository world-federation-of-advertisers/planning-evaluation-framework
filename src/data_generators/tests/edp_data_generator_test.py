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
import pandas as pd

from wfa_planning_evaluation_framework.data_generators.edp_data_generator import (
    EdpDataGenerator,
)


class EdpDataGeneratorTest(absltest.TestCase):
    def test_single_publisher_single_demo(self):
        data_design = pd.DataFrame(
            {
                "publisher_id": [2],
                "mc_id": ["mc-5"],
                "gender": ["M"],
                "age_group": ["18-24"],
                "social_grade": ["ABC1"],
                "date": ["2022/01/02"],
                "mean": [5.0],
                "std": [10.0],
                "cardinality": [3],
            }
        )

        expected_data = pd.DataFrame(
            {
                "Publisher ID": [2] * 5,
                "Event ID": [0, 1, 2, 3, 4],
                "Sex": ["M"] * 5,
                "Age Group": ["18-24"] * 5,
                "Social Grade": ["ABC1"] * 5,
                "Date": ["2022/01/02"] * 5,
                "Complete": ["1"] * 5,
                "VID": [0, 2, 2, 1, 0],
            }
        )
        data = EdpDataGenerator(data_design, 1).generate_data()
        pd.testing.assert_frame_equal(data, expected_data)

    def test_single_publisher_multiple_demos(self):
        data_design = pd.DataFrame(
            {
                "publisher_id": [2] * 3,
                "mc_id": ["mc-5"] * 3,
                "gender": ["M", "F", "M"],
                "age_group": ["18-24", "25-34", "35-44"],
                "social_grade": ["ABC1", "ABC2", "ABC3"],
                "date": ["2022/01/02", "2022/01/03", "2022/01/03"],
                "mean": [5.0] * 3,
                "std": [10.0] * 3,
                "cardinality": [2] * 3,
            }
        )

        expected_data = pd.DataFrame(
            {
                "Publisher ID": [2] * 7,
                "Event ID": [0, 1, 2, 3, 4, 5, 6],
                "Sex": ["M", "M", "M", "F", "F", "M", "M"],
                "Age Group": ["18-24"] * 3 + ["25-34"] * 2 + ["35-44"] * 2,
                "Social Grade": ["ABC1"] * 3 + ["ABC2"] * 2 + ["ABC3"] * 2,
                "Date": ["2022/01/02"] * 3 + ["2022/01/03"] * 4,
                "Complete": ["1"] * 7,
                "VID": list([0, 0, 1, 3, 2, 5, 4]),
            }
        )
        data = EdpDataGenerator(data_design, 1).generate_data()
        pd.testing.assert_frame_equal(data, expected_data)

    def test_multiple_publishers_multiple_demos(self):
        data_design = pd.DataFrame(
            {
                "publisher_id": [1, 2, 1],
                "mc_id": ["mc-5"] * 3,
                "gender": ["M", "M", "F"],
                "age_group": ["18-24", "18-24", "25-34"],
                "social_grade": ["ABC1", "ABC1", "ABC2"],
                "date": ["2022/01/02", "2022/01/02", "2022/01/03"],
                "mean": [5.0] * 3,
                "std": [10.0] * 3,
                "cardinality": [2] * 3,
            }
        )

        expected_data = pd.DataFrame(
            {
                "Publisher ID": [1, 1, 2, 2, 2, 1, 1],
                "Event ID": [0, 1, 2, 3, 4, 5, 6],
                "Sex": ["M", "M", "M", "M", "M", "F", "F"],
                "Age Group": ["18-24"] * 5 + ["25-34"] * 2,
                "Social Grade": ["ABC1"] * 5 + ["ABC2"] * 2,
                "Date": ["2022/01/02"] * 5 + ["2022/01/03"] * 2,
                "Complete": ["1"] * 7,
                "VID": list([0, 3, 1, 2, 3, 5, 4]),
            }
        )
        data = EdpDataGenerator(data_design, 1).generate_data()
        pd.testing.assert_frame_equal(data, expected_data)


if __name__ == "__main__":
    absltest.main()
