'''
Functions related to argparse package.
Used to set up parameter options for neural networks.
'''

import argparse
import csv
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms
from pylib.utils.ArgparseConverter import ArgparseConverter
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing

from .neural_nets.losses import *
from .neural_nets.pyg_fns import (AdjPCATransform, AdjPCs, AdjTransform,
                                  ContactDistance, Degree,
                                  DiagonalParameterDistance, GeneticDistance,
                                  GeneticPosition, GridSize,
                                  MeanContactDistance, OneHotGeneticPosition,
                                  WeightedLocalDegreeProfile)


def get_base_parser():
    '''Helper function that returns basic parser with all necessary arguments.'''
    parser = argparse.ArgumentParser(description='Base parser', fromfile_prefix_chars='@',
                                    allow_abbrev = False)
    AC = ArgparseConverter()

    # GNN pre-processing args
    parser.add_argument('--transforms', type=AC.str2list, default=[],
                        help='list of transforms to use for GNN')
    parser.add_argument('--pre_transforms', type=AC.str2list, default=[],
                        help='list of pre-transforms to use for GNN')
    parser.add_argument('--sparsify_threshold', type=AC.str2float,
                        help='remove all edges with weight < threshold (None to do nothing)')
    parser.add_argument('--sparsify_threshold_upper', type=AC.str2float,
                        help='remove all edges with weight < threshold (None to do nothing)')
    parser.add_argument('--use_node_features', type=AC.str2bool, default=False,
                        help='True to use node features for GNN models')
    parser.add_argument('--use_edge_weights', type=AC.str2bool, default=True,
                        help='True to use edge weights in GNN')
    parser.add_argument('--use_edge_attr', type=AC.str2bool, default=False,
                        help='True to use edge attr in GNN')
    parser.add_argument('--keep_zero_edges', type=AC.str2bool, default=False,
                        help='True to keep edges with zero weight')

    # pre-processing args
    parser.add_argument('--data_folder', type=AC.str2list, default='dataset_04_18_21',
                        help='Location of data')
    parser.add_argument('--root_name', type=AC.str2None,
                        help='name of file to save graph data'
                            '(leave as None to create root automatically)'
                            '(root is the directory path - defined later)')
    parser.add_argument('--delete_root', type=AC.str2bool, default=True,
                        help='True to delete root directory after runtime')
    parser.add_argument('--y_preprocessing', type=AC.str2None,
                        help='type of pre-processing for contact map')
    parser.add_argument('--output_preprocesing', type=AC.str2None,
                        help='type of preprocessing for output')
    parser.add_argument('--kr', type=AC.str2bool,
                        help='True to use KnightRuiz balancing algorithm')
    parser.add_argument('--mean_filt', type=AC.str2int,
                        help='mean_filt: apply mean filter of width <mean_filt> (None to skip)')
    parser.add_argument('--rescale', type=AC.str2int,
                        help='rescale contact map by factor of <rescale> (None to skip)')
    parser.add_argument('--preprocessing_norm', type=AC.str2None, default='batch',
                        help='type of [0,1] normalization for input data')
    parser.add_argument('--min_subtraction', type=AC.str2bool, default=True,
                        help='if min subtraction should be used for preprocessing_norm')
    parser.add_argument('--x_reshape', type=AC.str2bool, default=True,
                        help='True if x should be considered a 1D image')
    parser.add_argument('--ydtype', type=AC.str2dtype, default='float32',
                        help='torch data type for y')
    parser.add_argument('--y_reshape', type=AC.str2bool, default=True,
                        help='True if y should be considered a 2D image')
    parser.add_argument('--classes', type=int, default=10,
                        help='number of classes in percentile normalization')

    # dataloader args
    parser.add_argument('--split_percents', type=AC.str2list,
                        help='Train, val, test split for dataset (percents)')
    parser.add_argument('--split_sizes', type=AC.str2list, default=[-1, 200, 0],
                        help='Train, val, test split for dataset (counts), -1 for remainder')
    parser.add_argument('--random_split', type=AC.str2bool, default=False,
                        help='True to use random train, val, test split')
    parser.add_argument('--shuffle', type=AC.str2bool, default=True,
                        help='Whether or not to shuffle dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for data loader to use')
    parser.add_argument('--max_sample', type=AC.str2int,
                        help='max sample from dataset (None for all)')

    # train args
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Starting epoch')
    parser.add_argument('--n_epochs', type=int, default=2,
                        help='Number of epochs to train for')
    parser.add_argument('--save_mod', type=int, default=5,
                        help='How often to save')
    parser.add_argument('--print_mod', type=int, default=2,
                        help='How often to print')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate. Default=0.001')
    parser.add_argument('--min_lr', type=AC.str2float, default=1e-7,
                        help='minium lr (model stops training at min_lr, None to ignore)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight Decay. Default=0.0')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of gpus')
    parser.add_argument('--scheduler', type=AC.str2None,
                        help='choice of LR scheduler')
    parser.add_argument('--milestones', type=AC.str2list, default=None,
                        help='Milestones for lr decay - format: <milestone1-milestone2>')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for lr scheduler')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for lr scheduler')
    parser.add_argument('--loss', type=str, default='mse',
                        help='Type of loss to use: options: {"mse", "cross_entropy"}')
    parser.add_argument('--loss_k', type=int,
                        help='k for loss functions')
    parser.add_argument('--lambda1', type=float, default=1,
                        help='weight for loss function')
    parser.add_argument('--lambda2', type=float, default=1,
                        help='weight for loss function')
    parser.add_argument('--lambda3', type=float, default=1,
                        help='weight for loss function')
    parser.add_argument('--grad_clip', type=AC.str2float,
                        help='Gradient clipping max norm')
    parser.add_argument('--w_reg', type=AC.str2None,
                        help='Type of regularization to use for W, options: {"l1", "l2"}')
    parser.add_argument('--reg_lambda', type=float, default=1e-4,
                        help='regularization strength for w_reg')
    parser.add_argument('--autoencoder_mode', type=AC.str2bool, default=False,
                        help='True to use input as target output (i.e. autoencoder)')
    parser.add_argument('--verbose', type=AC.str2bool, default=False,
                        help='True to print')
    parser.add_argument('--print_params', type=AC.str2bool, default=True,
                        help='True to print parameters after training')
    parser.add_argument('--output_mode', type=str, default='contact',
                        help='data structure of output {"contact", "sequence", "energy"}')
    parser.add_argument('--save_early_stop', type=AC.str2bool, default=False,
                        help='True to save model at first lr milestone')

    # model args
    parser.add_argument('--model_type', type=str, default='test',
                        help='Type of model')
    parser.add_argument('--id', type=AC.str2int,
                        help='id of model')
    parser.add_argument('--pretrain_id', type=AC.str2int,
                        help='ID for using a pretrained model')
    parser.add_argument('--resume_training', type=AC.str2bool, default=False,
                        help='True if resuming training of a partially trained model')
    parser.add_argument('--k', type=AC.str2int,
                        help='Number of input epigenetic marks')
    parser.add_argument('--m', type=int, default=1024,
                        help='Number of particles')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to use')
    parser.add_argument('--act', type=AC.str2None, default='relu',
                        help='default activation') # TODO impelement throughout
    parser.add_argument('--inner_act', type=AC.str2None,
                        help='default activation (not used for all networks)')
    parser.add_argument('--out_act', type=AC.str2None,
                        help='activation of final layer')
    parser.add_argument('--gated', type=AC.str2bool, default=False,
                        help='True to use gated connection')
    parser.add_argument('--training_norm', type=AC.str2None,
                        help='norm during training (batch, instance, or None)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout probability')
    parser.add_argument('--parameter_sharing', type=AC.str2bool, default=False,
                        help='true to use parameter sharing in autoencoder blocks')
    parser.add_argument('--use_bias', type=AC.str2bool, default=True,
                        help='true to use bias (only implemented in ContactGNN and MLP)')

    # GNN model args
    parser.add_argument('--use_sign_net', type=AC.str2bool, default=False,
                        help='True to use sign net architecture')
    parser.add_argument('--use_sign_plus', type=AC.str2bool, default=True,
                        help='True to use sign plus architecture')
    parser.add_argument('--message_passing', type=str, default='GCN',
                        help='type of message passing algorithm')
    parser.add_argument('--head_architecture', type=AC.str2None,
                        help='type of head architecture')
    parser.add_argument('--head_architecture_2', type=AC.str2None,
                        help='2nd type of head architecture')
    parser.add_argument('--head_hidden_sizes_list', type=AC.str2list,
                        help='List of hidden sizes for convolutional layers')
    parser.add_argument('--encoder_hidden_sizes_list', type=AC.str2list,
                        help='hidden sizes for node encoder')
    parser.add_argument('--inner_hidden_sizes_list', type=AC.str2list,
                        help='hidden sizes for inner architecture')
    parser.add_argument('--edge_encoder_hidden_sizes_list', type=AC.str2list,
                        help='hidden sizes for edge encoder')
    parser.add_argument('--update_hidden_sizes_list', type=AC.str2list,
                        help='hidden sizes for update step of MPGNN')
    parser.add_argument('--head_act', type=AC.str2None, default='relu',
                        help='activation function for head network')
    parser.add_argument('--num_heads', type=AC.str2int, default=1,
                        help='number of attention heads for relevant MPGNN')
    parser.add_argument('--concat_heads', type=AC.str2bool, default=True,
                        help='False to average instead of concat attention heads')
    parser.add_argument('--bonded_path', type=AC.str2None,
                        help='argument in GridSize transform')

    # SimpleEpiNet args
    parser.add_argument('--kernel_w_list', type=AC.str2list,
                        help='List of kernel widths of convolutional layers')
    parser.add_argument('--hidden_sizes_list', type=AC.str2list,
                        help='List of hidden sizes for convolutional layers')

    # post-processing args
    parser.add_argument('--plot', type=AC.str2bool, default=True,
                        help='True to plot result figures')
    parser.add_argument('--plot_predictions', type=AC.str2bool, default=True,
                        help='True to plot predictions')

    return parser

def finalize_opt(opt, parser, windows = False, debug = False):
    '''
    Helper function to processes command line arguments.

    Inputs:
        opt (argparse.ArgumentParser): parsed command line arguments from parser.parse_args()
        parser: instance of argparse.ArgumentParser() - used to re-parse if needed
        windows: True for windows file path
        debug: True for debug mode (won't throw warning for resume_training)

    Outputs:
        opt (argparse.ArgumentParser): processed object
    '''
    # set up output folders/files
    model_type_folder = osp.join('results', opt.model_type)
    os.makedirs(model_type_folder, exist_ok=True)

    if opt.resume_training:
        assert opt.id is not None

    if opt.id is None:
        if not osp.exists(model_type_folder):
            os.mkdir(model_type_folder, mode = 0o755)
            opt.id = 1
        else:
            max_id = 0
            for filename in os.listdir(model_type_folder):
                if filename.isnumeric():
                    id = int(filename)
                    if id > max_id:
                        max_id = id
            opt.id = max_id + 1
    else:
        txt_file = osp.join(model_type_folder, str(opt.id), 'argparse.txt')
        if osp.exists(txt_file):
            assert opt.resume_training or debug, f"issue with id={opt.id}"
            id_copy = opt.id
            args = sys.argv.copy() # need to copy if running finalize_opt multiple times
            args.insert(1, f'@{txt_file}')
            args.pop(0) # remove program name
            opt = parser.parse_args(args) # parse again
            # by inserting at position 1, the original arguments will override the txt file
            opt.id = id_copy


    opt.ofile_folder = osp.join(model_type_folder, str(opt.id))
    if not osp.exists(opt.ofile_folder):
        os.mkdir(opt.ofile_folder, mode = 0o755)
    opt.log_file_path = osp.join(opt.ofile_folder, 'out.log')
    opt.log_file = open(opt.log_file_path, 'a')

    param_file_path = osp.join(opt.ofile_folder, 'params.log')
    opt.param_file = open(param_file_path, 'a')

    # configure other model params
    assert (opt.split_percents is None) ^ (opt.split_sizes is None)


    if opt.message_passing.lower() == 'gat':
        assert not opt.use_edge_weights

    # configure GNN transforms
    opt.node_feature_size = 0
    if opt.use_node_features:
        assert opt.k is not None
        opt.node_feature_size += opt.k
    else:
        msg = f"need feature augmentation for id={opt.id}"
        assert (len(opt.transforms) + len(opt.pre_transforms)) > 0, msg

    if opt.rescale is not None:
        assert opt.rescale != 0, f'{opt.id}'
        opt.input_m = int(opt.m / opt.rescale)
    else:
        opt.input_m = opt.m

    # transforms
    process_transforms(opt)

    # configure loss
    process_loss(opt)

    # configure cuda
    if opt.gpus > 1:
        opt.cuda = True
        opt.use_parallel = True
        opt.gpu_ids = []
        for ii in range(6):
            try:
                torch.cuda.get_device_properties(ii)
                print(str(ii), file = opt.log_file)
                opt.gpu_ids.append(ii)
            except AssertionError:
                print('Not ' + str(ii) + "!", file = opt.log_file)
    elif opt.gpus == 1:
        opt.cuda = True
        opt.use_parallel = False
    else:
        opt.cuda = False
        opt.use_parallel = False

    if opt.cuda and not torch.cuda.is_available():
        print('Warning: falling back to cpu', file = opt.log_file)
        opt.cuda = False
        opt.use_parallel = False

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    # Set random seeds
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    opt.log_file.close() # save any writes so far
    opt.log_file = open(opt.log_file_path, 'a')
    return opt

def process_transforms(opt):
    '''
    Format pytorch geometric transformations based on opt.transforms and
    opt.pre_transforms.
    '''
    # collect these for printing purposes (see opt2list)
    opt.edge_transforms = []
    opt.node_transforms = []
    opt.edge_dim = 0 # keep track of dimension of edge attributes

    # transforms
    transforms_processed = []
    for t_str in opt.transforms:
        t_str = t_str.lower().split('_')
        if t_str[0] == 'constant':
            opt.node_transforms.append(torch_geometric.transforms.Constant())
            transforms_processed.append(torch_geometric.transforms.Constant())
            opt.node_feature_size += 1
        elif t_str[0] == 'sparse':
            opt.node_transforms.append(torch_geometric.transforms.ToSparseTensor())
            transforms_processed.append(torch_geometric.transforms.ToSparseTensor())
        else:
            raise Exception("Invalid transform {}".format(t_str))
    if len(transforms_processed) > 0:
        opt.transforms_processed = torch_geometric.transforms.Compose(transforms_processed)
    else:
        opt.transforms_processed = None

    # pre-transforms
    processed = []
    opt.diag = False # True if y_diag is needed for transform
    opt.corr = False # True if correlation matrix is needed for transform
    for t_str in opt.pre_transforms:
        t_str = t_str.lower().split('_')
        if t_str[0] == 'constant':
            transform = torch_geometric.transforms.Constant()
            opt.node_transforms.append(transform)
            opt.node_feature_size += 1
        elif t_str[0] == 'weightedldp':
            transform = WeightedLocalDegreeProfile()
            opt.node_transforms.append(transform)
            if (opt.sparsify_threshold is None and
                opt.sparsify_threshold_upper is None):
                print('Warning: using LDP without any sparsification')
            opt.node_feature_size += 5
        elif t_str[0] == 'degree' or t_str[0] == 'weighteddegree':
            if t_str[0] == 'weighteddegree':
                weighted = True
            else:
                weighted = False
            opt.node_feature_size += 1
            max_value = None
            for mode_str in t_str[1:]:
                if mode_str == 'diag':
                    opt.diag = True
                if mode_str.startswith('max'):
                    assert mode_str[3:].isnumeric()
                    max_value = float(mode_str[3:])
            transform = Degree(diag = opt.diag, max_val = max_value,
                            weighted = weighted) # normalized by default
            opt.node_transforms.append(transform)
        elif t_str[0] == 'onehotdegree':
            opt.node_feature_size += opt.m + 1
            transform = torch_geometric.transforms.OneHotDegree(opt.m)
            opt.node_transforms.append(transform)
        elif t_str[0] == 'adj':
            opt.node_feature_size += opt.m
            processed.append(AdjTransform())
            opt.node_transforms.append(AdjTransform())
        elif t_str[0] == 'adjpca':
            k = 8
            diag = False
            for mode_str in t_str[1:]:
                if mode_str == 'diag':
                    diag = True
                    opt.diag = True
                elif mode_str.isdigit():
                    k = int(mode_str)
            opt.node_feature_size += k
            transform = AdjPCATransform(k, diag)
            opt.node_transforms.append(transform)
        elif t_str[0] == 'adjpcs':
            opt.diag = True
            norm = False
            k = 8
            for mode_str in t_str[1:]:
                if mode_str == 'norm':
                    norm = True
                elif mode_str.isdigit():
                    k = int(mode_str)
            if not opt.use_sign_net and not opt.use_sign_plus:
                opt.node_feature_size += k
            transform = AdjPCs(k, norm, opt.use_sign_net or opt.use_sign_plus)
            opt.node_transforms.append(transform)
        elif t_str[0] == 'contactdistance':
            assert opt.use_edge_attr or opt.use_edge_weights
            if opt.use_edge_attr:
                opt.edge_dim += 1
            norm = False
            bonded = False
            rank = None
            diag_norm = False
            corr = False
            for mode_str in t_str[1:]:
                if mode_str == 'norm':
                    norm = True
                if mode_str == 'bonded':
                    bonded = True
                if mode_str.startswith('rank'):
                    rank = mode_str[4:]
                    assert rank.isnumeric()
                    rank = int(rank)
                if mode_str == 'diagnorm':
                    opt.diag = True
                    diag_norm = True
                if mode_str == 'corr':
                    opt.diag = True
                    opt.corr = True
                    corr = True
            transform = ContactDistance(norm = norm,
                                        convert_to_attr = opt.use_edge_attr,
                                        bonded = bonded, diag_normalize = diag_norm,
                                        corr = corr, rank = rank)
            opt.edge_transforms.append(transform)
        elif t_str[0] == 'meancontactdistance':
            assert opt.use_edge_attr or opt.use_edge_weights
            if opt.use_edge_attr:
                opt.edge_dim += 1
            norm = False
            bonded = False
            for mode_str in t_str[1:]:
                if mode_str == 'norm':
                    norm = True
                if mode_str == 'bonded':
                    bonded = True

            transform = MeanContactDistance(norm = norm,
                                        convert_to_attr = opt.use_edge_attr,
                                        bonded = bonded)
            opt.edge_transforms.append(transform)
        elif t_str[0] == 'geneticdistance':
            assert opt.use_edge_attr or opt.use_edge_weights
            if opt.use_edge_attr:
                opt.edge_dim += 1
            log = False
            log10 = False
            norm = False
            pos = False
            d = 0
            for mode_str in t_str[1:]:
                if mode_str == 'log':
                    log = True
                if mode_str == 'log10':
                    log10 = True
                if mode_str == 'norm':
                    norm = True
                if mode_str.startswith('pos'):
                    pos = True
                    d = int(mode_str[3:])
                    opt.edge_dim += d-1

            transform = GeneticDistance(convert_to_attr = opt.use_edge_attr,
                                        log = log, log10 = log10, norm = norm,
                                        positional_encoding = pos,
                                        positional_encoding_d = d)
            opt.edge_transforms.append(transform)
        elif t_str[0] == 'geneticposition':
            center = False
            norm = False
            for mode_str in t_str[1:]:
                if mode_str == 'center':
                    center = True
                elif mode_str == 'norm':
                    norm = True

            opt.node_feature_size += 1
            transform = GeneticPosition(center = center, norm = norm)
            opt.node_transforms.append(transform)
        elif t_str[0] == 'onehotgeneticposition':
            transform = OneHotGeneticPosition()
            opt.node_feature_size += opt.m
            opt.node_transforms.append(transform)
        elif t_str[0] == 'gridsize':
            transform = GridSize(opt.bonded_path)
            opt.node_transforms.append(transform)
            opt.node_feature_size += 1
        elif t_str[0] == 'diagonalparameterdistance':
            assert opt.use_edge_attr or opt.use_edge_weights

            if len(t_str) > 1 and t_str[1].isdigit():
                mlp_id = int(t_str[1])
            else:
                mlp_id = None
            if opt.use_edge_attr:
                opt.edge_dim += 1

            transform = DiagonalParameterDistance(convert_to_attr = opt.use_edge_attr,
                                                id = mlp_id)
            opt.edge_transforms.append(transform)
        else:
            raise Exception(f'Unrecognized transform: {t_str} for id={opt.id}')
        processed.append(transform)

    if len(processed) > 0:
        opt.pre_transforms_processed = torch_geometric.transforms.Compose(processed)
    else:
        opt.pre_transforms_processed = None

    # these are used in opt2list for making results table
    opt.edge_transforms = sorted([repr(i) for i in opt.edge_transforms])
    opt.node_transforms = sorted([repr(i) for i in opt.node_transforms])
    # see pre_transforms_processed for more complete description of transforms

def process_loss(opt):
    '''Format pytorch loss function based on opt.loss.'''
    opt.loss = opt.loss.lower()
    loss_list = opt.loss.split('_and_')
    criterion_list = []
    arg_list = []
    for loss in loss_list:
        arg = None
        if loss == 'mse':
            criterion = F.mse_loss
            opt.channels = 1
        elif loss == 'huber':
            criterion = F.huber_loss
            opt.channels = 1
        elif loss == 'BCE':
            assert opt.out_act is None, "Cannot use output activation with BCE"
            if opt.output_mode == 'contact':
                msg = 'must use some sort of preprocessing_norm'
                assert opt.preprocessing_norm is not None, msg
            criterion = F.binary_cross_entropy_with_logits
        elif loss == 'mse_center':
            criterion = mse_center
        elif loss == 'mse_log':
            criterion = mse_log
        elif loss == 'mse_exp_norm':
            criterion = MSE_EXP_NORM()
        elif loss == 'mse_log_scc':
            criterion = MSE_log_scc(opt.m)
        elif loss == 'mse_plaid':
            criterion = MSE_plaid()
        elif loss == 'mse_log_plaid':
            criterion = MSE_plaid(log=True)
        elif loss == 'mse_diag':
            criterion = MSE_diag()
        elif loss == 'mse_log_diag':
            criterion = MSE_diag(log=True)
        elif loss == 'scc':
            criterion = SCC_loss(opt.m)
        elif loss.startswith('scc_exp'):
            loss_split = loss.split('_')
            K=100; clip=None; norm=False
            for loss_str in loss_split:
                if loss_str[0] == 'K':
                    K = int(loss_str[1:])
                elif loss_str.startswith('clip'):
                    clip = int(loss_str[4:])
                elif loss_str == 'norm':
                    norm = True
            criterion = SCC_loss(opt.m, True, K=K, clip_val=clip, norm=norm)
        else:
            raise Exception(f'Invalid loss: {repr(loss)}')
        criterion_list.append(criterion)
        arg_list.append(arg)
    if len(criterion_list) == 1:
        opt.criterion = criterion_list.pop()
    else:
        assert len(criterion_list) < 4, "not supported"
        opt.criterion = Combined_Loss(criterion_list,
                                    [opt.lambda1, opt.lambda2, opt.lambda3],
                                    arg_list)

def argparse_setup():
    """Helper function set up parser."""
    parser = get_base_parser()
    opt = parser.parse_args()
    return finalize_opt(opt, parser)

def save_args(opt):
    with open(osp.join(opt.ofile_folder, 'argparse.txt'), 'w') as f:
        for arg in sys.argv[1:]: # skip the program file
            f.write(arg + '\n')

def opt2list(opt):
    '''
    Convert argparse.Namespace object to list.
    Note: does not exhaustively consider all otions.
    '''
    data_folder = '-'.join([osp.split(d)[1] for d in opt.data_folder])
    opt_list = [opt.model_type, opt.id, data_folder, opt.pretrain_id,
                opt.preprocessing_norm, opt.y_preprocessing, opt.output_preprocesing,
                opt.mean_filt, opt.rescale, opt.kr, opt.min_subtraction]
    opt_list.append(opt.split_percents if opt.split_percents is not None else opt.split_sizes)
    opt_list.extend([opt.shuffle, opt.batch_size, opt.num_workers, opt.n_epochs, opt.lr,
        opt.weight_decay, opt.w_reg,
        opt.milestones, opt.gamma, opt.loss,
        opt.k, opt.m, opt.seed, opt.act, opt.inner_act,
        opt.head_act, opt.out_act, opt.training_norm])
    # GNN options
    opt_list.extend([opt.use_node_features, opt.use_edge_weights, opt.use_edge_attr,
                    opt.node_transforms, opt.edge_transforms,
                    opt.sparsify_threshold, opt.sparsify_threshold_upper,
                    opt.encoder_hidden_sizes_list,
                    opt.edge_encoder_hidden_sizes_list,
                    opt.hidden_sizes_list, opt.message_passing,
                    opt.update_hidden_sizes_list,
                    f'{opt.head_architecture}+{opt.head_architecture_2}',
                    opt.head_hidden_sizes_list])
    if opt.use_sign_net:
        opt_list.append('sign_net')
    elif opt.use_sign_plus:
        opt_list.append('sign_plus')
    else:
        opt_list.append(None)

    opt_list.append(opt.output_mode)

    return opt_list

def save_opt(opt, ofile):
    if not osp.exists(ofile):
        with open(ofile, 'w', newline = '') as f:
            wr = csv.writer(f)
            opt_list = get_opt_header(opt.model_type)
            wr.writerow(opt_list)
    with open(ofile, 'a') as f:
        wr = csv.writer(f)
        opt_list = opt2list(opt)
        wr.writerow(opt_list)

def get_opt_header(model_type):
    '''Return list of strings corresponding to variables in argparse.Namespace object.'''
    opt_list = ['model_type', 'id',  'dataset', 'pretrain_id', 'preprocessing_norm',
        'y_preprocessing', 'output_preprocesing', 'mean_filt', 'rescale',
        'kr', 'min_subtraction', 'split', 'shuffle',
        'batch_size', 'num_workers', 'n_epochs', 'lr', 'weight_decay', 'w reg', 'milestones',
        'gamma', 'loss', 'k', 'm',
        'seed', 'act', 'inner_act', 'head_act', 'out_act',
        'training_norm']
    # GNN params
    opt_list.extend(['use_node_features','use_edge_weights', 'use_edge_attr',
                    'node_transforms', 'edge_transforms',
                    'sparsify_threshold', 'sparsify_threshold_upper',
                    'encoder_hidden_sizes', 'edge_encoder_hidden_sizes',
                    'hidden_sizes', 'message_passing', 'update_hidden_sizes',
                    'head_architecture', 'head_hidden_sizes', 'sign_net'])

    opt_list.append('output_mode')

    return opt_list
