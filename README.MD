# Causality Aware Reinforcement Learning (CARL)

This is the code repository to run the experiments reported in our lastest research paper:

A. Méndez-Molina, E. F. Morales and L. E. Sucar, ‘CARL: A Synergistic Framework for Causal Reinforcement Learning’ IEEE Access, vol. 11, pp. 126 462–126 481, 2023. [doi:
10.1109/ACCESS.2023.3331728](doi:10.1109/ACCESS.2023.3331728)

<!-- You can download the paper [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4409869) at SSRN.
 ## Citation
 For early citation, please use:
 Méndez-Molina, Arquímides and Morales, Eduardo and L. Enrique, Sucar, Carl: A Synergistic Framework for Causal Reinforcement Learning. Available at SSRN: # https://ssrn.com/abstract=4409869 or http://dx.doi.org/10.2139/ssrn.4409869 -->

## Installation

1. Clone this repository to your local workspace.
```
git clone https://github.com/arquimides/carl.git
```

2. The project uses Python and R, so you need to install them.
    
    2.1 For Python we recommend to use Anaconda or similar package manager to manage the project dependencies in an isolated environment. To install Anaconda you can follow these steps:
    
    2.1.1 First download the latest Anaconda distribution for your operating system (Example, for Linux):
    ```
    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
    ```
    2.1.2 Install Anaconda running the following command. Replace Anaconda3-2023.09-0-Linux-x86_64.sh with the actual filename of the installer you downloaded. Follow the prompts to complete the installation.
    ```
    bash Anaconda3-2023.09-0-Linux-x86_64.sh
    ```
    2.1.3 After the installation is complete, you may need to initialize Conda. You can do this by closing and reopening your terminal, or by running: 
    ```
    source ~/.bashrc
    ```
    2.1.4 Create a conda environment for this project. We test with python 3.10. Example:
    ```
    conda create --name carl_env python=3.10
    ```
    2.1.5 Finally active the conda environment to be ready to install all the dependencies later
    ```
    conda activate carl_env
    ```
    
    2.2 For R we recommend to install the latest version from CRAN. To do that follow the instructions at https://cran.r-project.org/bin/linux/ubuntu/fullREADME.html#installation
    
3. Install Python project dependencies install the dependencies using the following command:
```
pip install -r requirements-cleanrl-requirements.txt
``` 
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

