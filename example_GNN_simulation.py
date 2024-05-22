import argparse
import json
import math
import os
import os.path as osp
import re
import shutil
import subprocess as sp
import sys
from time import sleep, time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pylib.analysis as analysis
import scipy.sparse as ss
import torch
from pylib.Pysim import Pysim
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import load_json, print_time

from scripts.argparse_utils import finalize_opt, get_base_parser
from scripts.data_generation.utils.setup_configs import setup_config
from scripts.neural_nets.utils import get_dataset, load_saved_model


def setup_example():
    hic = ss.load_npz('example/hic.npz')
    hic_arr = hic.toarray()
    np.save('example/hic.npy', hic_arr)
    plot_matrix(hic_arr, 'example/hic.png', vmax='mean')
        # maximum of color bar = mean(hic)

def setup_simulation():
    dir = 'example'
    root, config = setup_config(dir)

    hic_file = osp.join(dir, 'hic.npy')
    if not osp.exists(hic_file):
        raise Exception(f'files does not exist: {hic_file}')
    hic = np.load(hic_file).astype(np.float64)
    m = len(hic)

    config['nspecies'] = 0
    config['load_bead_types'] = False
    config['lmatrix_on'] = False
    config['dmatrix_on'] = False
    config['dump_frequency'] = 1000
    config['nSweeps'] = 3000
    config['nSweeps_eq'] = 100
    config['nbeads'] = m
    config["umatrix_filename"] = "umatrix.txt"

    gnn_root = f'{root}-GNN'

    return dir, root, gnn_root, config, hic

def run_GNN(model_path, argparse_path, m, dir, root, gnn_root, use_GPU=True, verbose=True):
    '''
    Use GNN to predit script U matrix.

    Inputs:
        model_path: path to .pt file
        argparse_path: path to argparse .txt file
        m: number of rows/cols in contact map (and number of particles in simulation)
        dir: directory where experimental contact map is located
        root: directory for bonded simulation
        gnn_root: directory for gnn simuation
        use_GPU: True to run GNN forward pass on GPU
        verbose: True for verbose print output
    '''
    log_file = osp.join(gnn_root, 'energy.log')
    ofile = osp.join(gnn_root, 'U.npy')

    t0 = time()
    stdout = sys.stdout
    with open(log_file, 'w') as sys.stdout:
        # set up argparse options
        parser = get_base_parser()
        sys.argv = [sys.argv[0]] # delete args from get_params, otherwise gnn opt will try and use them
        opt = parser.parse_args(['@{}'.format(argparse_path)])
        opt = finalize_opt(opt, parser, debug = True)
        opt.m = m
        opt.output_mode = None # don't need output, since only predicting
        opt.root_name = f'GNN{opt.id}' # need this to be unique if running in parallel
        opt.log_file = sys.stdout
        opt.cuda = False # default to use cpu (will change later if use_gpu=True)
        opt.device = torch.device('cpu')
        opt.scratch = ''
        if verbose:
            print(opt)

        # get model
        model, _, _ = load_saved_model(opt, model_path, verbose=verbose)

        # get dataset
        dataset = get_dataset(opt, verbose = verbose, file_paths = ['example'])
        if verbose:
            print('Dataset: ', dataset, len(dataset))
            print()

        # get prediction
        data = dataset[0]
        data.batch = torch.zeros(data.num_nodes, dtype=torch.int64)

        if verbose:
            print(data)

        if use_GPU:
            model = model.to('cuda:0')
            data = data.to('cuda:0')
            with torch.no_grad():
                prediction = model(data, verbose=verbose)
        else:
            with torch.no_grad():
                prediction = model(data, verbose=verbose)
        if torch.isnan(prediction).any():
            shutil.rmtree(opt.root)
            raise Exception(f'nan in prediction: {opt.root}')

        U = prediction.cpu().detach().numpy().reshape((opt.m,opt.m))
        if verbose:
            print('energy', U)

        if opt.output_preprocesing == 'log':
            U = np.multiply(np.sign(U), np.exp(np.abs(U)) - 1)

        if verbose:
            print('energy processed', U)

        ## cleanup
        # opt.root is set in get_dataset
        model = model.cpu()
        del model; del data; del dataset; del prediction
        torch.cuda.empty_cache()
        shutil.rmtree(opt.root)

        np.save(ofile, U)
    sys.stdout = stdout
    tf = time()
    print_time(t0, tf, 'gnn')

    return U

def main():
    '''
    Run simulation using GNN.
    '''
    setup_example()

    dir, root, gnn_root, config, hic = setup_simulation()
    m = len(hic)

    GNN_model = 'gnn_model.pt' # model parameters
    argparse_file = 'argparse.txt' # model architecture options
    assert osp.exists(GNN_model), f'GNN model not found at {GNN_model}'
    assert osp.exists(argparse_file), f'Argparse filee not found at {argparse_file}'


    if osp.exists(gnn_root):
        shutil.rmtree(gnn_root)
        # print('GNN root already exists - exiting')
        # return
    os.mkdir(gnn_root, mode=0o755)

    U = run_GNN(GNN_model, argparse_file, m, dir, root, gnn_root, use_GPU=False)
    if U is None:
        return

    stdout = sys.stdout
    with open(osp.join(gnn_root, 'log.log'), 'w') as sys.stdout:
        sim = Pysim(gnn_root, config, None, hic, randomize_seed = True,
                    mkdir = False, umatrix = U)
        t = sim.run_eq(config['nSweeps_eq'], config['nSweeps'], 1)
        print(f'Simulation took {np.round(t, 2)} seconds')

        analysis.main_no_maxent(dir=sim.root)
    sys.stdout = stdout

if __name__ == '__main__':
    main()
