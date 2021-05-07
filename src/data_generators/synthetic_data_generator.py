from absl import app
from absl import flags

from wfa_planning_evaluation_framework.data_generators.data_design import DataDesign
from wfa_planning_evaluation_framework.data_generators.data_set import DataSet

def main(argv):

  data_design = DataDesign(dirpath="somePath")
  for data_set_config in data_set_configs:
    data_design.add(generateDataSet(data_set_config))

def generateDataSet(poisson_lambda) -> DataSet:
  pdf = PublisherData.generate_publisher_data(
    HomogeneousImpressionGenerator(3, 5), FixedPriceGenerator(0.01), "test"
  )
  data_set1 = DataSet([pdf11, pdf12], "ds1")

if __name__ == '__main__':
  app.run(main)

