# [Chromatin Structures from Integrated AI and Polymer Physics Model](https://doi.org/10.1101/2024.11.27.624905)

![test](https://github.com/ERSchultz/GNN_HiC_to_Structure/blob/main/overview.png)

## Installation and dependencies


We include a bash script (bin/create_environment.sh) to create an anaconda environment and install all necessary dependencies. A reference environment file is provided in environment.yml. 

We also require https://github.com/SorenKyhl/TICG-chromatin (master branch) in order to run the simulation engine and for some utility functions.

You can run the following bash commands to install TICG-chromatin within the GNN conda environment:

conda activate python3.9_pytorch1.9

cd ~/TICG-chromatin

git checkout master

make all

## Getting started with pre-trained model
First, download the pre-trained GNN model and argparse file from: https://huggingface.co/ERSchultz/GNN_HiC_to_Structure. Save them to ~/GNN_HiC_to_Structure/.

Next, simply run example_GNN_simulation.py. It will use the GNN to estimate the U matrix and simulate the experimental contact map located in the example folder.

## Overview of github directory
- bin contains bash scripts for environment creation and training the GNN.
- example contains an example experimental contact map to be simulated with example_GNN_simulation.py.
- scripts contains python scripts.
- scripts.data_generation includes scripts for generating synthetic training data.
- scripts.import_hic includes scripts for importing .hic files.
- scripts.neural_nets includes scripts for building and training PyTorch models.
- scripts.argparse_utils.py includes functions for command line interfacing via the argpase package.
- scripts.clean_directories.py includes functions for cleaning unnecessary files after training the GNN.
- scripts.plotting_utils includes a variety of plotting utility functions.
- core_test_train.py includes the core train/test loop for the GNN.


## To retrain the GNN
### Import experimental data
Run scripts.import_hic.import_contactmap_straw.py in order to import experimental contact maps at 50kb resolution for 9 cell lines.

### Generate synthetic training data
Run scripts/data_generation.data_generation.py to complete data generation procedure described in manuscript. This is a wrapper script that will call:
1) scripts/data_generation/fit_max_ent.py
2) scripts/data_generation/preprocess_max_ent.py
3) scripts/data_generation/generate_synthetic_parameters.py
3) scripts/data_generation/run_simulations.py

### Train GNN
Run bin/train_GNN.sh to train the GNN on the synthetic dataset. This bash script will call scripts/core_test_train.py
