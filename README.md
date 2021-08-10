# Planning Evaluation Framework

## Overview

This repo contains the code for the Planning Evaluation Framework, which
evaluates planning models on both synthetic and real data. A planning model
predicts the effectiveness of an advertising campaign that is spread across
multiple publishers, by estimating the number of people that would be reached at
various frequencies as a function of the amount spent with each publisher. The
purpose of the Planning Evaluation Framework is to

*   Provide a means for identifying the best candidate planning modeling strategies
    to be used in the Halo measurement system;
*   Provide the ability for publishers to evaluate the quality of models when
    applied to their own internal data sets; and,
*   Provide the ability to simulate the behavior of the Halo system on realistic
    advertiser workflows.

### Workflow


A typical workflow consists of the following steps:

1.  Generate a Data Design.
2.  Configure an Experimental Design.
3.  Run the Evaluator to create the results of evaluating the experiments
    against the data.
4.  Analyze the results in Colab.

A Data Design specifies the set of data that will be used for evaluating models.
It is given by a directory structure. Each subdirectory of the Data Design
defines a Dataset. A Dataset in turn consists of a collection of Publisher Data
Files. A Publisher Data File simulates the impression viewing history of a
single publisher. It is a CSV file, where each line is of the form

```
   vid, spend
```

where vid is a virtual user id, and spend is the total amount that would have
to be spent for this impression to be shown.

A Data Design can be synthetic, or it can represent actual publisher data.
Synthetic data designs can be generated using the program
`data_generators/synthetic_data_generator.py`. Alternatively, if you have other
data available in your organization, you can construct your own data design. 

An Experimental Design specifies the modeling strategies that will be
evaluated and the parameters that will be used for evaluation. An
Experimental Design is given by a Python class. An Experimental Design
is subdivided into Experiments, which are further broken down into
Trials. An Experiment consists of a collection of modeling strategies
and model parameters that are run against a single Dataset. A Trial
consists of simulating the outcome of one modeling strategy using one
Dataset and one set of parameter values. 

Once a Data Design and an Experimental Design have been specified, the Evaluator
can be used to run simulations of the experimental design versus the data
design. The final output of the Evaluator is a DataFrame, containing one row per
Trial.

The results of an Evaluator run can then be analyzed in Colab. For some example
colabs, see the `analyzers` directory.

### Quickstart

It is recommended to use a virtual environment with Python version 3.8+ for this project. If you already
have one, you can skip to the next step. The quickest way to set up a virtual
environment is by running:

```
python3 -m venv env
source env/bin/activate
```

To install the dependencies, run:

```
pip install -r requirements.txt
```

To execute the Planning Evaluation Framework, we have found it most convenient to
modify PYTHONPATH and create a symbolic link to the source code, following these steps:

1. Add PYTHONPATH to _.bash_profile_ (or _.zshrc_ depending which shell you use) as following:

```
PYTHONPATH=$HOME/dir_which_contains_planning_evaluation_framework_repo:.
export PYTHONPATH
```

Then run `source path_to/.bash_profile` or `source path_to/.zshrc_` in the terminal.

2. Create a Symlink named `wfa_planning_evaluation_framework` at the directory which contains your planning-evaluation-framework repo with command:
    
```
ln -s path_to/planning-evaluation-framework/src/ dir_which_contains_planning_evaluation_framework_repo/wfa_planning_evaluation_framework
```

### Example

This section walks you through the steps to set up the evaluation framework and
to run a sample evaluation using synthetic data.  In this section, we will assume that
the current working directory is the `src` subdirectory of the evaluation framework.

To start with, you should create a directory that you will use for working:

```
DIR=<some-path-that-will-contain-data-and-results>
mkdir -p $DIR
```

The `data_generators` subdirectory contains several example configuration files
that can be used to generate synthetic data sets:

* `simple_data_design_example.py`: A very simple example.
* `single_publisher_data_design.py`: The data design that is used for validating the single publisher models.
* `m3_data_design.py`: The data design that is used for validating the M3 model.

In this example, we will generate data for the single publisher models and then analyze the
performance of these models.  The next command invokes the synthetic data generator.  


```
python3 data_generators/synthetic_data_design_generator.py \
  --output_dir=$DIR/data \
  --data_design_config=data_generators/single_publisher_design.py
```

After running this command, you should see that the directory `$DIR/data` contains many
subdirectories.  The directory `$DIR/data` is a Data Design, and each of the subdirectories
within it represents a Dataset.  Each Dataset in turn contains a collection of files representing
synthetic impression logs.  There is one such file per synthetic publisher.  These files are
just CSV files, so you can view them.

An Experimental Design specifies a collection of different models and parameters.  An Experiment
consists of running each of these against every Dataset in a Data Design.  Several example
Experimental Designs can be found in the `driver` subdirectory:

* `sample_experimental_design.py`:  A very simple example.
* `single_publisher_design.py`: An experimental design that compares Goerg's one point model
against the Gamma-Poisson model in a variety of settings.
* `m3_first_round_experimental_design.py`: An experimental design for evaluating the proposed
M3 model.

The following command will evaluate the Experiments defined in `driver/single_publisher_design.py`
against the Datasets that were created in the previous step:


```
python3 driver/experiment_driver.py \
  --data_design_dir=$DIR/data \
  --experimental_design=driver/single_publisher_design.py \
  --output_file=$DIR/results \
  --intermediates_dir=$DIR/intermediates \
  --cores=0
```

Setting `cores=0` enables multithreading.  Even so, the above design takes very long to run, so you
may want to reduce both the number of Datasets and the number of Experiments by (temporarily)
modifying the respective configuration files.  To see verbose output as the evaluation proceeds,
try adding the parameter `--v==3`.  Once the evaluation is complete, the results will be recorded
in a CSV file named `$DIR/results`.  You can then load this into colab and explore the results that
were obtained.  For some example colabs, see the `analysis` directory.

### Directory Structure

The components of the Planning Evaluation Framework are arranged into the
following subdirectories:

*   `models`: classes that provide mathematical models of different aspects of
    the planning problem.
*   `data_generators`: generate synthetic data sets for evaluation by the
    models.
*   `simulator`: classes that simulate the operation of the Halo system; they
    apply modeling strategies to data sets and return reach surfaces.
*   `driver`: a program that performs many simulations for various combinations
    of modeling strategies and data sets, returning a data frame of results.
*   `analysis`: notebooks that generate interpretable results from the evaluation results.

### Contributing

#### Code formatting

All of the code in this project is formatted with the [black](https://black.readthedocs.io/en/stable/)
tool, using the following command line:

```
black file1.py file2.py ...
```

#### Steps for merging a pull request

1. `git checkout main` and `git pull` to get the latest changes. 
2. `git checkout target_branch`. Make sure it is the latest. 
3. `git rebase main`. Resolve conflicts if there is any. 
4. Run all unit tests with command `find . | egrep ".*tests/.*.py" | xargs -n 1 python3`
5. If everything is okay, `git push -f` to push your rebased branch to the server. 
6. Go to the [web interface](https://github.com/world-federation-of-advertisers/planning-evaluation-framework) and click `merge pull request`.

