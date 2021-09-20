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
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "wfa_planning_evaluation_framework"
DESCRIPTION = "Framework for Evaluating Cross-Publisher Planning Models."
URL = "https://github.com/world-federation-of-advertisers/planning-evaluation-framework"
EMAIL = ""
AUTHOR = ""
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.0.1"

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy==1.20.2",
    "pandas==1.2.5",
    "absl-py==0.12.0",
    "typing-extensions==3.7.4.3",
    "pathos==0.2.7",
    "fsspec==2021.7.0",
    "gcsfs==2021.7.0",
    "google-cloud-storage==1.42.0",
    "cloudpathlib==0.6.0",
    "pyfarmhash==0.2.2",
    "pyDOE==0.3.8",
    "scipy==1.6.2",
    "tqdm==4.47.0",
    "lxml==4.5.2",
    "cvxopt==1.2.6",
    "cvxpy==1.1.12",
    "dp-accounting==0.0.1",
]

# What packages are optional?
EXTRAS = {}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


# This call to setup() does all the work
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=[
        "wfa_planning_evaluation_framework",
        "wfa_planning_evaluation_framework.models",
        "wfa_planning_evaluation_framework.data_generators",
        "wfa_planning_evaluation_framework.simulator",
        "wfa_planning_evaluation_framework.driver",
    ],
    package_dir={"wfa_planning_evaluation_framework": "src"},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
