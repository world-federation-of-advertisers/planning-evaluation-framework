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

# A class of functions that maps the number of publishers p
# to the number of testing points.


class NumTestPointsGenerators:
    def p_times_10(p):
        return 10 * p

    def p_times_10(p):
        return 100 * p

    def p_sq_times_2(p):
        return 2 * p ** 2

    def p_sq_times_10(p):
        return 10 * p ** 2
