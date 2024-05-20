Contains scripts for constructing pytorch (and pytorch geometric) datasets and neural network models.

base_networks.py contains basic pytorch utilities. For example, we have implemented the Pearson Correlation in pytorch. We also have several neural network building blocks in this script (e.g. convolutional blocks).

losses.py contains all of the loss functions we have explored. Our final model is trained using mse_log().

networks.py contains the GNN pytorch geometric model.

pyg_dataset_classes.py contains the pytorch geometric dataset class for convertign hic matrices to graphs.

pyg_fns.py contains basic pytorch geometric utilities. Notably, WeightedGATv2Conv is our implementation of GATv2Conv that accounts for edge attributes during the message passing computation. We also implement a number of hic-specific pytorch geometric BaseTransform classes.

utils.py contains utilty functions for loading models/datasets/data loaders. 
