# causal_rl

This is the code repository to run the experiments for the under-review article "Carl: A Synergistic Framework for Causal Reinforcement Learning". 

<!-- You can download the paper [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4409869) at SSRN.
 ## Citation
 For early citation, please use:
 Méndez-Molina, Arquímides and Morales, Eduardo and L. Enrique, Sucar, Carl: A Synergistic Framework for Causal Reinforcement Learning. Available at SSRN: # https://ssrn.com/abstract=4409869 or http://dx.doi.org/10.2139/ssrn.4409869 -->

## Installation

1. Clone this repository to your workspace.
2. The project uses Python and R, so you need to install them.
3. We recommend you to create a python virtual environment. Once created and activated, install the dependencies using the following command:
  ```
  pip install -r requirements.txt
  ```
4. Install the required R libs (bnlearn and Rgraphviz): Execute the following commands in R console.

```
install.packages("https://www.bnlearn.com/releases/bnlearn_latest.tar.gz", repos = NULL, type = "source")
```
```
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("Rgraphviz")
```

5. Install the Gym environments: This project uses custom OpenIA Gym environments to test an agent performing CARL. The code for the environments is a standalone project that you can clone and download at https://github.com/arquimides/our_gym_environments . In the project page you can find the instructions to install the environments in the same virtual-env you are using.

## Usage

To run all experiments in the paper, just run the `crl.py` script. You can manually edit the desired experiments to run by modifying the script in line 695. To modify any parameter of any experiment you need to modify the `config.py` file inside the "experiments_configurations" folder.


## Examples outputs

The outputs for the experiments will be saved to the "experiment_results" folder.

