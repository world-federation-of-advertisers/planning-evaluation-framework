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
"""Installs wfa_planning_evaluation_framework."""
from setuptools import setup

setup(
    name='wfa_planning_evaluation_framework',
    version='0.01',
    description='Framework for Evaluating Cross-Publisher Planning Models',
    python_requires='>=3.6',
    packages=[
        'wfa_planning_evaluation_framework',
        'wfa_planning_evaluation_framework.models',
        'wfa_planning_evaluation_framework.data_generators',
        'wfa_planning_evaluation_framework.simulator',
        'wfa_planning_evaluation_framework.driver',
    ],
    package_dir={'wfa_planning_evaluation_framework': 'src'}
)
