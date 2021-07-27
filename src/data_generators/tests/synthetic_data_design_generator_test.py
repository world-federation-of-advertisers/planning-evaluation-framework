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
from unittest.mock import patch

from wfa_planning_evaluation_framework.data_generators.synthetic_data_design_generator import (
    SyntheticDataDesignGenerator,
)
from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.data_set_parameters import (
    GeneratorParameters,
)
from wfa_planning_evaluation_framework.data_generators.independent_overlap_data_set import (
    IndependentOverlapDataSet,
)
from wfa_planning_evaluation_framework.data_generators import lhs_data_design_example
from wfa_planning_evaluation_framework.data_generators import m3_data_design
from wfa_planning_evaluation_framework.data_generators import (
    analysis_example_data_design,
)
from wfa_planning_evaluation_framework.data_generators import simple_data_design_example
from wfa_planning_evaluation_framework.data_generators import single_publisher_design

TEST_LEVELS = {
    "largest_publisher_size": [8, 16],
    "overlap_generator_params": [
        GeneratorParameters(
            "Independent",
            IndependentOverlapDataSet,
            {"largest_pub_to_universe_ratio": 0.5, "random_generator": 1},
        ),
    ],
}


class SyntheticDataDesignGeneratorTest(absltest.TestCase):
    def test_simple_design(self):
        simple_design = simple_data_design_example.generate_data_design_config(
            np.random.default_rng(seed=1)
        )
        self.assertLen(list(simple_design), 27)

    def test_lhs_design(self):
        lhs_design = lhs_data_design_example.generate_data_design_config(
            np.random.default_rng(seed=1)
        )
        self.assertLen(list(lhs_design), 10)

    def test_m3_design_size(self):
        m3_design = m3_data_design.generate_data_design_config(
            np.random.default_rng(seed=1)
        )
        self.assertLen(list(m3_design), 100)

    def test_analysis_example_design_size(self):
        analysis_example_design = (
            analysis_example_data_design.generate_data_design_config(
                np.random.default_rng(seed=1)
            )
        )
        self.assertLen(list(analysis_example_design), 32)

    @patch(
        "wfa_planning_evaluation_framework.data_generators.m3_data_design.LEVELS",
        new=TEST_LEVELS,
    )
    @patch(
        "wfa_planning_evaluation_framework.data_generators.m3_data_design.NUM_SAMPLES_FOR_LHS",
        new=2,
    )
    def test_m3_design_generate_universe_size(self):
        test_design = m3_data_design.generate_data_design_config(
            np.random.default_rng(seed=1)
        )
        x = next(test_design).overlap_generator_params.params["universe_size"]
        y = next(test_design).overlap_generator_params.params["universe_size"]
        self.assertCountEqual([x, y], [16, 32])

    def test_single_publisher_design(self):
        sp_design = single_publisher_design.generate_data_design_config(
            np.random.default_rng(seed=1)
        )
        self.assertLen(
            list(sp_design),
            56,
            "Expected single pub design to have {} datasets but it had {}".format(
                56, len(list(sp_design))
            ),
        )

    def test_synthetic_data_generator_simple_design(self):
        with TemporaryDirectory() as d:
            data_design_generator = SyntheticDataDesignGenerator(
                d, simple_data_design_example.__file__, 1, False
            )
            data_design_generator()
            dd = DataDesign(d)
            self.assertEqual(dd.count, 27)

    def test_synthetic_data_generator_lhs_design(self):
        with TemporaryDirectory() as d:
            data_design_generator = SyntheticDataDesignGenerator(
                d, lhs_data_design_example.__file__, 1, False
            )
            data_design_generator()
            dd = DataDesign(d)
            self.assertEqual(dd.count, 10)


if __name__ == "__main__":
    absltest.main()
