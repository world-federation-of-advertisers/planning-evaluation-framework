# Copyright 2022 The Private Cardinality Estimation Framework Authors
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
"""Generates synthetic data for use in testing the Halo system.

Generates a synthetic data set in a format that can be used for testing
the Halo system.  For an example of the data format, see the following
file:

  https://github.com/world-federation-of-advertisers/cross-media-measurement/blob/main/src/main/k8s/testing/data/synthetic-labelled-events.csv


Example usage:

  python3 edp_data_generator.py \
    --data_design_config=<data_design> \
    --output=<output_file> \
    [--random_seed=<int>]

where

  output_file is the name of the output file where the data will be written.
  data_design_config is the name of a python file that specifies the underlying
  data design.  This file should be a CSV file with the following columns:

    publisher_id,mc_id,gender,age_group,social_grade,date,mean,std,cardinality

  where

    publisher_id: A string or int uniquely identifying the EDP
    mc_id: A string or int uniquely identifying the measurement consumer
    gender: A string identifying the gender (typically either "M" or "F")
    age_group: A string identifying the age group (example: "18-24")
    social_grade: A string identifying the social grade of this user (example: "ABC1")
    date: The date the impressions were shown, given as YYYY/MM/DD.
    mean: The average number of impressions per viewer
    std:  The standard deviation of the number of impressions per viewer.  Note: this
      must be larger than the mean.
    cardinality: The target number of unique people reached in this group.

The output is a file containing one row per impression.  The columns in the
output file are as follows:

    Publisher ID,Event ID,Sex,Age Group,Social Grade,Date,Complete,VID

For each combination of (publisher_id, mc_id, gender, age_group,
social_grade, date), approximately cardinality many rows are generated
in the output.  The distribution of impressions per viewer follows a
negative binomial distribution with specified mean and standard
deviation.  If multiple publishers are specified in the input, then
the overlap is determined using an independent Gaussian copula.  In
this case, due to the way that data is sampled from the Gaussian
copula, the cardinality of the users in the synthetically generated
output may not exactly match the input values.

TODO: Extend this to include campaign_id's.
TODO: Extend this to include watch times.

"""

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from statsmodels.distributions.copula.other_copulas import IndependenceCopula
from typing import Dict
from typing import List
from wfa_planning_evaluation_framework.data_generators.copula_data_set import (
    CopulaDataSet,
)
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet
from wfa_planning_evaluation_framework.data_generators.fixed_price_generator import (
    FixedPriceGenerator,
)
from wfa_planning_evaluation_framework.data_generators.heterogeneous_impression_generator import (
    HeterogeneousImpressionGenerator,
)
from wfa_planning_evaluation_framework.data_generators.publisher_data import (
    PublisherData,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("output", None, "File where synthetic data will be written")
flags.DEFINE_string(
    "data_design_config", None, "Name of CSV file containing data design configuration"
)
flags.DEFINE_integer("random_seed", 1, "Seed for the random number generator")

# Not used currently, but required as a parameter to generate publisher data
PRICE_PER_IMPRESSION = 0.1


class EdpDataGenerator:
    """Generates a synthetic data for use in testing Halo."""

    def __init__(
        self,
        data_design_config: str,
        random_seed: int = 1,
    ):
        """Constructor for EdpDataGenerator.

        Args:
          data_design_config:  pd.DataFrame, specifies the configuration of the
            data design.
          random_seed:  Int, a value used to initialize the random number
            generator.
        """
        self._data_design_config = data_design_config
        self._random_generator = np.random.default_rng(random_seed)
        np.random.seed(random_seed)

    def _generate_data_for_row(self, i: int, pub_id: str) -> PublisherData:
        """Generates a data set for a single edp and demo group.

        Args:
          i: The row of self._data_design_config for which the data should
            by genereated.
          pub_id: Publisher id
        Returns:
          A PublisherData object representing the list of impressions associated
          to this demo group.
        """
        mean = self._data_design_config["mean"].iloc[i]
        std = self._data_design_config["std"].iloc[i]
        n = self._data_design_config["cardinality"].iloc[i]
        if mean >= std**2+1:
            raise ValueError(
                "Invalid values for mean and std. "
                "Mean must be less than std**2+1. "
                f"Got mean {mean}, std {std} at row {i}"
            )

        p = 1 - (mean - 1) / std**2
        alpha = (mean - 1) * (1 - p) / p
        beta = p / (1 - p)

        data = HeterogeneousImpressionGenerator(
            n,
            gamma_shape=alpha,
            gamma_scale=beta,
            random_generator=self._random_generator,
        )()
        publisher_data = PublisherData(
            FixedPriceGenerator(PRICE_PER_IMPRESSION)(data), name=pub_id
        )
        return publisher_data

    def _generate_cross_publisher_data_for_demo_groups(self) -> Dict:
        """Generates cross-publisher datasets for each distinct demo group."""
        publishers = {}
        for i in range(self._data_design_config.shape[0]):
            demo_bucket = tuple(
                self._data_design_config[
                    ["mc_id", "gender", "age_group", "social_grade", "date"]
                ].iloc[i]
            )
            pub_id = self._data_design_config["publisher_id"].iloc[i]
            publisher_data = self._generate_data_for_row(i, pub_id)
            if not demo_bucket in publishers:
                publishers[demo_bucket] = [publisher_data]
            else:
                publishers[demo_bucket].append(publisher_data)

        cross_publisher_data = {}
        for demo_group in publishers:
            if len(publishers[demo_group]) <= 1:
                dataset = DataSet(publishers[demo_group], str(demo_group))
            else:
                dataset = CopulaDataSet(
                    unlabeled_publisher_data_list=publishers[demo_group],
                    copula_generator=IndependenceCopula(),
                    random_generator=self._random_generator,
                )
            cross_publisher_data[demo_group] = dataset

        return cross_publisher_data

    def _cross_publisher_data_to_impression_dataframe(self, data: Dict) -> pd.DataFrame:
        """Converts a dictionary of cross-publisher datasets to a dataframe."""
        impression_data = []
        starting_vid = 0
        max_event_id = 0
        vid_start = 0
        max_vid = 0
        for demo_bucket in data:
            mc_id, gender, age_group, social_grade, date = demo_bucket
            for pub in data[demo_bucket]._data:
                pub_id = pub.name
                impressions = [id + starting_vid for id, _ in pub._data]
                n = len(impressions)
                df = pd.DataFrame(
                    {
                        "Publisher ID": [pub_id] * n,
                        "Event ID": list(
                            range(max_event_id, max_event_id + len(impressions))
                        ),
                        "Sex": [gender] * n,
                        "Age Group": [age_group] * n,
                        "Social Grade": [social_grade] * n,
                        "Date": [date] * n,
                        "Complete": ["1"] * n,
                        "VID": [i + vid_start for i in impressions],
                    }
                )
                max_event_id += len(impressions)
                impression_data.append(df)
                max_vid = max(max_vid, max(df["VID"]))
            vid_start = max_vid + 1

        return pd.concat(impression_data).reset_index(drop=True)

    def generate_data(self) -> pd.DataFrame:
        """Generates a synthetic data set."""
        data = self._generate_cross_publisher_data_for_demo_groups()
        return self._cross_publisher_data_to_impression_dataframe(data)


def main(argv):
    data_design = pd.read_csv(FLAGS.data_design_config)
    data = EdpDataGenerator(data_design, FLAGS.random_seed).generate_data()
    data.to_csv(FLAGS.output, index=False)


if __name__ == "__main__":
    app.run(main)
