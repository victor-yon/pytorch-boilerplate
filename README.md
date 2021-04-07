# pyTorch Boilerplate

Generic code template for pytorch project.

## Install

Required `python >= 3.8` and `pip`

```shell script
pip install -r requirements.txt
echo "run_name: tmp" >> settings.yaml
```

## Template quick start

1. Remove the `.idea/` directory if you don't use pyCharm IDE.
2. In `utils/logger.py` at the last line, set the current project name in: `logger_name='<project name>'`.
3. Create your own dataset in `datasets/` or remove the folder if you use a dataset already implemented in pytorch.
4. Create your own neural network in `networks/`.
5. Replace the dataset and the network in `main.py`.
6. Remove / add settings in `utils/settings.py`
7. Run `run.py`
8. Update the title / description of this README file, and remove this section.

## Run

Run a single _training_ + _testing_ based on the current settings:

```shell
python3 run.py
```

Run a list _training_ + _testing_ based on the current settings and the planner configuration:

```shell
python3 runs_planner.py
```

## Settings

An extendable list of settings with their default values can be found in the file `utils/settings.py`.

Their values can be overridden by:

- the local setting file (`./settings.yaml`)
- the environment variables
- the arguments of the command line (`--help` to see the list)

## File structure

* __datasets/__: Directory that contains the list of available datasets classes that extend the
  pytorch [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class.
* __networks/__: Directory that contains the list of available neural network that extend the
  pytorch [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class.
* __out/__: Directory where named run output will be saved. A subdirectory will be generated for each new run.
* __plots/__: Directory that contains all plotting scripts.
* __utils/__: Directory that contains utility scripts, using them is optional, but they simplify some tasks.
  * __logger.py__: Class wrapper to handle console and file logging.
  * __metrics.py__: Script to compute and save metrics about the network and the run.
  * __output.py__: Script to manage the run output (log file, images, results, ...).
  * __planner.py__: Class to create an iterable list run settings.
  * __progress_bar.py__: Class to handle visual progress bar.
  * __settings.py__: Dataclass to store and load settings values. Create a singleton at the first import.
  * __timer.py__: Class wrapper to handle, print and save timers.
* __main.py__: Simple script to set up dataset and neural network before to start a run.
* __run.py__: Full script that contains the run steps logic.
* __runs_analyse.py__: Script to aggregate and plot result from several runs.
* __runs_planner.py__: Script to start a list of runs with different settings one by one.
* __settings.yaml__: Local settings file to override default values. This file shouldn't be committed.
* __test.py__: Script that contains the network testing logic.
* __train.py__: Script that contains the network training logic.
