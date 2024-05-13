import math
import os.path as osp

import numpy as np
import torch
import torch_geometric
from torch.utils.data import DataLoader, Subset

from .dataset_classes import DiagFunctions, Sequences, SequencesContacts
from .networks import get_model
from .pyg_dataset_classes import ContactsGraph


## model functions ##
def load_saved_model(opt, verbose=True, throw=True):
    model = get_model(opt, verbose)
    model.to(opt.device)
    model_name = osp.join(opt.ofile_folder, 'model.pt')
    if osp.exists(model_name):
        save_dict = torch.load(model_name, map_location=torch.device('cpu'))
        train_loss_arr = save_dict['train_loss']
        val_loss_arr = save_dict['val_loss']
        try:
            state_dict = save_dict['model_state_dict']
            # for key in list(state_dict.keys()):
                # if key.startswith('model.head_2'):
                    # state_dict[key.replace('head_2', 'head_D2')] = state_dict.pop(key)

                # starts = [1, 8, 15, 22]
                # for module, start in zip([1, 3, 5, 7], starts):
                #     for i in [0, 1, 2]:
                #         j = start + 2*i
                #         if key == f'model.module_{j}.weight':
                #             state_dict[f'model.module_{module}.model.{i}.model.0.weight'] = state_dict.pop(key)
                #         elif key == f'model.module_{j}.bias':
                #             state_dict[f'model.module_{module}.model.{i}.model.0.bias'] = state_dict.pop(key)
                #         elif key == f'model.module_{j+1}.weight':
                #             state_dict[f'model.module_{module}.model.{i}.model.1.weight'] = state_dict.pop(key)
                # if key.startswith('head_2.'):
                #     state_dict[key.replace('head_2.', 'head_2.0.model.')]= state_dict.pop(key)

                # if key.startswith('model.module_7'):
                #     state_dict[key.replace('model.module_7', 'model.module_2')] = state_dict.pop(key)
                # if key.startswith('model.module_14'):
                #     state_dict[key.replace('model.module_14', 'model.module_4')] = state_dict.pop(key)
                # if key.startswith('model.module_21'):
                #     state_dict[key.replace('model.module_21', 'model.module_6')] = state_dict.pop(key)
            # torch.save(save_dict, model_name)
            model.load_state_dict(state_dict)
            model.eval()
            if verbose:
                print('Model is loaded: {}'.format(model_name), file = opt.log_file)
        except Exception as e:
            print(e)
            print(state_dict.keys())
            if throw:
                raise
            else:
                return None, train_loss_arr, val_loss_arr
    else:
        raise Exception('Model does not exist: {}'.format(model_name))

    return model, train_loss_arr, val_loss_arr

## dataset functions ##
def get_dataset(opt, names=False, minmax=False, verbose=True,
                samples=None, sub_dir='samples'):
    opt.root = None
    if opt.GNN_mode:
        if opt.split_sizes is not None and -1 not in opt.split_sizes:
            max_sample = np.sum(opt.split_sizes)
        elif opt.max_sample is not None:
            max_sample = opt.max_sample
        else:
            max_sample = float('inf')

        dataset = ContactsGraph(opt.data_folder, opt.scratch, opt.root_name,
                                opt.input_m, opt.y_preprocessing,
                                opt.log_preprocessing, opt.kr, opt.rescale, opt.mean_filt,
                                opt.preprocessing_norm, opt.min_subtraction,
                                opt.use_node_features, opt.mlp_model_id,
                                opt.sparsify_threshold, opt.sparsify_threshold_upper,
                                opt.split_neg_pos_edges, opt.max_diagonal,
                                opt.transforms_processed, opt.pre_transforms_processed,
                                opt.output_mode, opt.crop, opt.log_file, verbose,
                                max_sample, samples, sub_dir,
                                opt.plaid_score_cutoff, opt.sweep_choices,
                                opt.diag, opt.corr, opt.eig,
                                opt.keep_zero_edges, opt.output_preprocesing,
                                opt.bonded_path)
        opt.root = dataset.root
        print('\n'*3)
    elif opt.autoencoder_mode and opt.output_mode == 'sequence':
        dataset = Sequences(opt.data_folder, opt.crop, opt.x_reshape, names)
    elif opt.model_type.upper() == 'MLP':
        dataset = DiagFunctions(opt.data_folder, opt.crop, opt.preprocessing_norm,
                                opt.y_preprocessing,
                                opt.log_preprocessing, opt.y_zero_diag_count,
                                opt.output_mode,
                                names = names, samples = samples)
    else:
        dataset = SequencesContacts(opt.data_folder, opt.toxx, opt.toxx_mode,
                                    opt.y_preprocessing, opt.preprocessing_norm,
                                    opt.x_reshape, opt.ydtype, opt.y_reshape,
                                    opt.crop, opt.min_subtraction, names, minmax)
    return dataset

def get_data_loaders(dataset, opt):
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, opt)
    if opt.verbose:
        print('dataset lengths: ', len(train_dataset), len(val_dataset), len(test_dataset))

    if opt.GNN_mode:
        dataloader_fn = torch_geometric.loader.DataLoader
    else:
        dataloader_fn = DataLoader
    train_dataloader = dataloader_fn(train_dataset, batch_size = opt.batch_size,
                                    shuffle = opt.shuffle, num_workers = opt.num_workers)
    if len(val_dataset) > 0:
        val_dataloader = dataloader_fn(val_dataset, batch_size = opt.batch_size,
                                        shuffle = opt.shuffle, num_workers = opt.num_workers)
    else:
        val_dataloader = None

    if len(test_dataset) > 0:
        test_dataloader = dataloader_fn(test_dataset, batch_size = opt.batch_size,
                                        shuffle = opt.shuffle, num_workers = opt.num_workers)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader

def split_dataset(dataset, opt):
    """Splits input dataset into proportions specified by split."""
    opt.N = len(dataset)
    if opt.split_percents is not None:
        assert abs(sum(opt.split_percents) - 1) < 1e-5, f"split doesn't sum to 1: {opt.split_percents}"
        opt.testN = math.floor(opt.N * opt.split_percents[2])
        opt.valN = math.floor(opt.N * opt.split_percents[1])
        opt.trainN = opt.N - opt.testN - opt.valN
    else:
        assert opt.split_sizes is not None
        assert opt.split_sizes.count(-1) < 2, "can be at most 1 entry set to -1"

        opt.trainN, opt.valN, opt.testN = opt.split_sizes
        if opt.trainN == -1:
            opt.trainN = opt.N - opt.testN - opt.valN
        elif opt.valN == -1:
            opt.valN = opt.N - opt.trainN - opt.testN
        elif opt.testN == -1:
            opt.testN = opt.N - opt.trainN - opt.valN

    print(f'split sizes: train={opt.trainN}, val={opt.valN}, test={opt.testN}, N={opt.N}',
        file = opt.log_file)

    if opt.random_split:
        return torch.utils.data.random_split(dataset, [opt.trainN, opt.valN, opt.testN],
                                            torch.Generator().manual_seed(opt.seed))
    elif opt.GNN_mode:
        test_dataset = dataset[:opt.testN]
        val_dataset = dataset[opt.testN:opt.testN+opt.valN]
        train_dataset = dataset[opt.testN+opt.valN:opt.testN+opt.valN+opt.trainN]
    else:
        # can't slice pytorch dataset, need to use Subset
        test_dataset = Subset(dataset, range(opt.testN))
        val_dataset = Subset(dataset, range(opt.testN, opt.testN+opt.valN))
        train_dataset = Subset(dataset, range(opt.testN+opt.valN, opt.testN+opt.valN+opt.trainN))


    assert len(test_dataset) == opt.testN, f"{len(test_dataset)} != {opt.testN}"
    assert len(train_dataset) == opt.trainN, f"{len(train_dataset)} != {opt.trainN}"

    return train_dataset, val_dataset, test_dataset

# pytorch helper functions
def optimizer_to(optim, device = None):
    # https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            if device is not None:
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            else:
                print(param.data.get_device())
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    if device is not None:
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
                    else:
                        print(subparam.data.get_device())
