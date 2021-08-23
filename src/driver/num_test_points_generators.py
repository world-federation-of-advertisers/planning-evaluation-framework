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
"""Functions to generate npoints in Latin/Uniform test point generataors."""

# A dictionary of functions that maps the number of publishers p
# to the number of testing points.

NUM_TEST_POINTS_GENERATORS = {
    "10p": lambda p: 10 * p,
    "100p": lambda p: 100 * p,
    "2p^2": lambda p: 2 * p ** 2,
    "10p^2": lambda p: 10 * p ** 2,
}
