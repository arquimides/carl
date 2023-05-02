# causal_rl

This is the code repository to run the experiments for the under-review article "Carl: A Synergistic Framework for Causal Reinforcement Learning". You can download the paper [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4409869) at SSRN.

## Citation

For early citation, please use:

Méndez-Molina, Arquímides and Morales, Eduardo and L. Enrique, Sucar, Carl: A Synergistic Framework for Causal Reinforcement Learning. Available at SSRN: https://ssrn.com/abstract=4409869 or http://dx.doi.org/10.2139/ssrn.4409869

## Installation

1. Clone this repository to your workspace.
2. The project uses Python and R, so you need to install them.
3. We recommend you to create a python virtual environment. Once created and activated, install the dependencies using the following command:
  ```
  pip install -r requirements.txt
  ```
4. Install the Gym environments: Find the gym folder inside your venv and copy the folder "our gym environments" inside the dir "gym/envs", at the same level of other pre-installed folders like "atari", "mujoco", ect. Then add the followings entries at the end of the `__init__.py` file located in "gym/envs":

```
register(
    id="CoffeeTaskEnv-v0",
    entry_point="gym.envs.our_gym_environments.coffee_task_env:CoffeeTaskEnv",
    nondeterministic=True,
)

register(
    id="TaxiSmallEnv-v0",
    entry_point="gym.envs.our_gym_environments.taxi_small_env:TaxiSmallEnv",
    nondeterministic=True,
)

register(
    id="TaxiBigEnv-v0",
    entry_point="gym.envs.our_gym_environments.taxi_big_env:TaxiBigEnv",
    nondeterministic=True,
)
```
5. Install the required R libs (bnlearn and Rgraphviz):

```
install.packages("https://www.bnlearn.com/releases/bnlearn_latest.tar.gz", repos = NULL, type = "source")
```
```
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("Rgraphviz")
```


## Usage

To run all experiments in the paper, just run the `crl.py` script. You can manually edit the desired experiments to run by modifying the script in lines form 592 to 611. To modify any parameter of any experiment you need to modify the `config.py` file inside the "experiments_configurations" folder.


## Examples outputs

The outputs for the experiments will be saved to the "experiment_results" folder.

