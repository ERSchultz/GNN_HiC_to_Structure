import copy
import json
import math
import multiprocessing
import os
import os.path as osp
import sys
import tarfile
from shutil import rmtree

import imageio
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_geometric
from pylib.utils.ArgparseConverter import ArgparseConverter
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import *
from sklearn.metrics import mean_squared_error

from .argparse_utils import (finalize_opt, get_base_parser, get_opt_header,
                             opt2list)
from .neural_nets.utils import get_data_loaders, get_dataset, load_saved_model


#### Functions for plotting loss ####
def plot_models_from_disk(ids, modelType='ContactGNNEnergy', use_id_for_label=False):
    '''
    Compare multiple models by loading from disk and plotting loss curve.

    Inputs:
        ids: list of model ids to plot
        modelType: type of model being plotted
        use_id_for_label: True to use model id as plot label
                        (if False, label will be inferred from differences in
                        model param options)
    '''
    path = osp.join('results', modelType) # models should be saved here

    dirs = []
    opts = []
    parser = get_base_parser()
    for id in ids:
        # get argparse.ArgumentParser object (opt) for each model id
        id_path = osp.join(path, str(id))
        dirs.append(osp.join(id_path, 'model.pt'))
        txt_file = osp.join(id_path, 'argparse.txt')
        opt = parser.parse_args(['@{}'.format(txt_file)])
        opt = finalize_opt(opt, parser, local = True, debug = True)
        opts.append(opt)

    imagePath = osp.join(path, f'{ArgparseConverter.list2str(ids)} combined')
    os.makedirs(imagePath, exist_ok = True)

    for log in [True, False]:
        plot_models_from_disk_inner(dirs, imagePath, opts, log_y = log, use_id_for_label = use_id_for_label)

def plot_models_from_disk_inner(dirs, imagePath, opts, log_y=False, use_id_for_label=False):
    # check that only one param is different
    opt_header = get_opt_header(opts[0].model_type)
    opt_lists = []
    for opt in opts:
        opt_lists.append(opt2list(opt))

    differences_names = []
    differences = []
    ids = []
    for pos in range(len(opt_lists[0])):
        first = True
        for model in range(len(opt_lists)):
            if first:
                ref = opt_lists[model][pos]
                first = False
            else:
                if opt_lists[model][pos] != ref:
                    param = opt_header[pos]
                    if param not in {'id', 'resume_training'}:
                        differences_names.append(param)
                        differences.append((ref, opt_lists[model][pos]))
                    if param == 'id':
                        ids = (ref, opt_lists[model][pos])


    print('Differences:')
    if len(differences) == 1:
        diff_name = differences_names.pop()
        diff = differences.pop()
        if diff_name == 'edge_transforms':
            diff_a, diff_b = diff
            diff_a = set(diff_a); diff_b = set(diff_b)
            intersect = diff_a.intersection(diff_b)
            diff = (diff_a.difference(intersect), diff_b.difference(intersect))
        print(diff)
    else:
        for name, (a, b) in zip(differences_names, differences):
            print(f'{name}:\n\t{a}\n\t{b}')
        diff_name = 'id'
        diff = ids
        print(diff)

    if use_id_for_label:
        diff_name = 'id'
        diff = ids

    fig, ax = plt.subplots()
    colors = ['b', 'r', 'g', 'c']
    styles = ['-', '--']
    colors = colors[:len(dirs)]
    types = ['training', 'validation']
    labels = []
    for dir, opt, c in zip(dirs, opts, colors):
        saveDict = torch.load(dir, map_location=torch.device('cpu'))
        train_loss_arr = saveDict['train_loss']
        val_loss_arr = saveDict['val_loss']
        l1 = ax.plot(np.arange(1, len(train_loss_arr)+1), train_loss_arr,
                    ls = styles[0], color = c)
        if log_y:
            ax.set_yscale('log')
        l2 = ax.plot(np.arange(1, len(val_loss_arr)+1), val_loss_arr,
                    ls = styles[1], color = c)

    for c, label_i in zip(colors, diff):
        ax.plot(np.NaN, np.NaN, color = c, label = label_i)

    ax2 = ax.twinx()
    for type, style in zip(types, styles):
        ax2.plot(np.NaN, np.NaN, ls = style, label = type, c = 'k')
    ax2.get_yaxis().set_visible(False)

    ax.legend(loc = 1, title = diff_name)
    ax2.legend(loc = 3)

    ax.set_xlabel('Epoch', fontsize = 16)
    if opts[0].loss != opts[1].loss:
        ylabel = 'Loss'
    else:
        opt = opts[0]
        if opt.loss == 'mse':
            ylabel = 'MSE Loss'
        elif opt.loss == 'cross_entropy':
            ylabel = 'Cross Entropy Loss'
        elif opt.loss == 'BCE':
            ylabel = 'Binary Cross Entropy Loss'
        elif opt.loss == 'huber':
            ylabel = 'Huber Loss'
        else:
            ylabel = 'Loss'
    if log_y:
        ylabel = f'{ylabel} (log-scale)'
        ax.set_ylim(None, np.nanpercentile(train_loss_arr, 99))
    else:
        ax.set_ylim(0, np.nanpercentile(train_loss_arr, 99))
    ax.set_ylabel(ylabel, fontsize = 16)

    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.preprocessing_norm is not None:
        preprocessing_norm = opt.preprocessing_norm.capitalize()
    else:
         preprocessing_norm = 'None'
    plt.title(f'Y Preprocessing: {preprocessing}, Norm: {preprocessing_norm}',
                fontsize = 16)

    plt.tight_layout()
    if log_y:
        plt.savefig(osp.join(imagePath, 'train_val_loss_log.png'))
    else:
        plt.savefig(osp.join(imagePath, 'train_val_loss.png'))
    plt.close()

def plot_loss(train_loss_arr, val_loss_arr, imagePath, opt=None,
                        log_y=False):
    """
    Plots loss as function of epoch.

    Inputs:
        train_loss_arr: arry of training loss as function of epoch
        val_loss_arr: array of validation loss as function of epoch
        imagePath: path to save images to
        opt (argparse.ArgumentParser object): parameter options for model
        log_y: true to log transform y-axis
    """
    plt.plot(np.arange(1, len(train_loss_arr)+1), train_loss_arr, label = 'Training')
    plt.plot(np.arange(1, len(val_loss_arr)+1), val_loss_arr, label = 'Validation')

    ylabel = 'Loss'
    if opt is not None:
        if opt.loss == 'mse':
            ylabel = 'MSE Loss'
        elif opt.loss == 'cross_entropy':
            ylabel = 'Cross Entropy Loss'
        elif opt.loss == 'BCE':
            ylabel = 'Binary Cross Entropy Loss'
        elif opt.loss == 'huber':
            ylabel = 'Huber Loss'
        else:
            ylabel = "Loss"

        if opt.y_preprocessing is not None:
            preprocessing = opt.y_preprocessing.capitalize()
        else:
            preprocessing = 'None'
        if opt.preprocessing_norm is not None:
            preprocessing_norm = opt.preprocessing_norm.capitalize()
        else:
             preprocessing_norm = 'None'
        upper_title = f'Y Preprocessing: {preprocessing}, Norm: {preprocessing_norm}'
        train_title = f'Final Training Loss: {np.round(train_loss_arr[-1], 3)}'
        val_title = f'Final Validation Loss: {np.round(val_loss_arr[-1], 3)}'
        plt.title(f'{upper_title}\n{train_title}\n{val_title}', fontsize = 16)


        if opt.milestones is not None:
            lr = float(opt.lr)
            max_y = np.max(np.maximum(train_loss_arr, val_loss_arr))
            min_y = np.min(np.minimum(train_loss_arr, val_loss_arr))
            new_max_y = max_y + (max_y - min_y) * 0.1
            annotate_y = max_y + (max_y - min_y) * 0.05
            x_offset = (opt.milestones[0] - 1) * 0.05
            if not log_y:
                plt.ylim(top = new_max_y)
            plt.axvline(1, linestyle = 'dashed', color = 'green')
            plt.annotate(f'lr: {lr}', (1 + x_offset, annotate_y))
            for m in opt.milestones:
                lr = lr * opt.gamma
                plt.axvline(m, linestyle = 'dashed', color = 'green')
                plt.annotate('lr: {:.1e}'.format(lr), (m + x_offset, annotate_y))


    plt.xlabel('Epoch', fontsize = 16)
    if log_y:
        plt.ylabel(f'{ylabel} (log-scale)', fontsize = 16)
        plt.yscale('log')
    else:
        plt.ylabel(ylabel, fontsize = 16)

    plt.legend()
    plt.tight_layout()
    if log_y:
        plt.savefig(osp.join(imagePath, 'train_val_loss_log.png'))
    else:
        plt.savefig(osp.join(imagePath, 'train_val_loss.png'))
    plt.close()

### Functions for analyzing model performance ###
def plotEnergyPredictions(val_dataloader, model, opt, count=5):
    print('Prediction Results:', file = opt.log_file)
    assert opt.output_mode.startswith('energy')
    if opt.y_preprocessing is not None:
        preprocessing = opt.y_preprocessing.capitalize()
    else:
        preprocessing = 'None'
    if opt.preprocessing_norm is not None:
        preprocessing_norm = opt.preprocessing_norm.capitalize()
    else:
         preprocessing_norm = 'None'
    upper_title = f'Y Preprocessing: {preprocessing}, Norm: {preprocessing_norm}'

    loss_dim = 1
    if opt.loss == 'mse':
        loss_title = 'MSE Loss'
    elif opt.loss == 'huber':
        loss_title = 'Huber Loss'
    elif opt.loss == 'mse_and_mse_center':
        loss_title = 'MSE+MSE_center'
        loss_dim = 2
    elif opt.loss == 'mse_log':
        loss_title = 'MSE_log'
    elif opt.loss == 'mse_log_and_mse_center_log':
        loss_title = 'MSE_log+MSE_center_log'
        loss_dim = 2
    elif opt.loss == 'mse_and_mse_log':
        loss_title = 'MSE+MSE_log'
        loss_dim = 2
    elif opt.loss == 'mse_log_and_mse_kth_diagonal':
        loss_title = 'MSE_log+MSE_k_diag'
        loss_dim = 2
    elif opt.loss == 'mse_log_and_mse_top_k_diagonals':
        loss_title = 'MSE_log+MSE_top_k_diag'
        loss_dim = 2
    else:
        loss_title = f'{opt.loss} loss'

    loss_arr = np.zeros((loss_dim, min(count, opt.valN)))
    for i, data in enumerate(val_dataloader):
        if i == count:
            break
        data = data.to(opt.device)
        y = data.energy
        y = torch.reshape(y, (-1, opt.m, opt.m))
        yhat = model(data)
        path = data.path[0]

        if 'seqs' in data._mapping:
            seqs = torch.reshape(data.seqs, (-1, 10, opt.m)) # TODO hard-coded 10
        else:
            seqs = None

        if loss_dim > 1:
            loss1, loss2 = opt.criterion(yhat, y, seqs, split_loss=True)
            loss1 = loss1.item()
            loss2 = loss2.item()
            loss = loss1 + loss2
            loss_arr[0, i] = loss1
            loss_arr[1, i] = loss2
        else:
            loss = opt.criterion(yhat, y, seqs).item()
            loss_arr[0, i] = loss
        y = y.cpu().numpy().reshape((opt.m, opt.m))
        yhat = yhat.cpu().detach().numpy().reshape((opt.m,opt.m))

        left_path, sample = osp.split(path)
        dataset = left_path.split(osp.sep)[-2]
        subpath = osp.join(opt.ofile_folder, sample)
        print(f'{dataset} {sample}: {loss}', file = opt.log_file)
        os.makedirs(subpath, exist_ok = True)

        yhat_title = '{}\n{} ({}: {})'.format(upper_title, r'$\hat{S}$',
                                                loss_title, np.round(loss, 3))

        if opt.verbose:
            print('y', y, np.max(y))
            print('yhat', yhat, np.max(yhat))

        v_max = np.max(y)
        v_min = np.min(y)

        plot_matrix(yhat, osp.join(subpath, 'energy_hat.png'), vmin = v_min,
                        vmax = v_max, cmap = 'blue-red', title = yhat_title)
        np.savetxt(osp.join(subpath, 'energy_hat.txt'), yhat, fmt = '%.3f')

        plot_matrix(y, osp.join(subpath, 'energy.png'), vmin = v_min,
                        vmax = v_max, cmap = 'blue-red', title = r'$S$')
        np.savetxt(osp.join(subpath, 'energy.txt'), y, fmt = '%.3f')

        # plot dif
        dif = y - yhat
        plot_matrix(dif, osp.join(subpath, 'edif.png'), vmin = -1 * v_max,
                        vmax = v_max, title = r'S - $\hat{S}$', cmap = 'blue-red')

        # plot meanDist
        for arr, label in zip([y, yhat],['Ground Truth', 'GNN']):
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'freq')
            plt.plot(meanDist, label = label)
        plt.legend()
        plt.xscale('log')
        plt.ylabel('Mean', fontsize=16)
        plt.xlabel('Off-diagonal Index', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(subpath, 'meanDist_S.png'))
        plt.close()

        # plot plaid contribution
        latent = model.latent(data, None)
        if len(latent.shape) == 2:
            latent = torch.unsqueeze(latent, 0)

        for i, latent_i in enumerate(latent):
            plaid_hat = model.plaid_component(latent_i)
            if plaid_hat is not None:
                plaid_hat = plaid_hat.cpu().detach().numpy().reshape((opt.m,opt.m))
                plot_matrix(plaid_hat, osp.join(subpath, f'plaid_hat_{i}.png'),
                                vmin = -1 * v_max, vmax = v_max,
                                title = 'plaid portion', cmap = 'blue-red')

            # plot diag contribution
            diagonal_hat = model.diagonal_component(latent_i)
            if diagonal_hat is not None:
                diagonal_hat = diagonal_hat.cpu().detach().numpy().reshape((opt.m,opt.m))
                plot_matrix(diagonal_hat, osp.join(subpath, f'diagonal_hat_{i}.png'),
                                vmin = -1 * v_max, vmax = v_max,
                                title = 'diagonal portion', cmap = 'blue-red')


        # tar subpath
        os.chdir(opt.ofile_folder)
        with tarfile.open(f'{dataset}_{sample}.tar.gz', 'w:gz') as f:
            f.add(sample)
        rmtree(sample)

    if loss_dim > 1:
        print(loss_arr.shape)
        sum_loss_arr = np.sum(loss_arr, 0)
        mean_loss = np.round(np.mean(sum_loss_arr), 3)
        std_loss = np.round(np.std(sum_loss_arr), 3)
        mean_loss1 = np.round(np.mean(loss_arr[0]), 3)
        mean_loss2 = np.round(np.mean(loss_arr[1]), 3)
        print(f'Loss1: {mean_loss1}, Loss2: {mean_loss2}',
            file = opt.log_file)
    else:
        mean_loss = np.round(np.mean(loss_arr), 3)
        std_loss = np.round(np.std(loss_arr), 3)
    print(f'{loss_title}: {mean_loss} +- {std_loss}\n',
        file = opt.log_file)

    return mean_loss

#### Functions for plotting xyz files ####
def plot_xyz(xyz, L, x=None, ofile=None, show=True, title=None, legend=True,
            colors = None):
    '''
    Plots particles in xyz as 3D scatter plot.
    Only supports mutually exclusive bead types for coloring. # TODO
    Inputs:
        xyz: shape (N,3) array of all particle positions
        L: side of LxLxL box (nm), if None the plot will be fit to the input
        x: bead types to color
        LJ: True if Lennard Jones particles
        ofile: location to save image
        show: True to show
        title: title of plot
    '''
    fig = plt.figure(figsize=[12.8, 9.6])
    ax = plt.axes(projection = '3d')

    # connect particles if polymer
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], color= '0.8')

    if x is None:
        ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
    elif len(x.shape) == 2:
        m, k = x.shape
        # color unique types if x is not None
        for t in range(k):
            condition = x[:, t] == 1
            # print(condition)
            if colors is not None:
                ax.scatter(xyz[condition,0], xyz[condition,1], xyz[condition,2],
                            label = t, color = colors[t], s=[100]*len(xyz[condition,0]), marker = 'o')
            else:
                ax.scatter(xyz[condition,0], xyz[condition,1], xyz[condition,2],
                            label = t)
        if legend:
            plt.legend()
    elif len(x.shape) == 1:
        im = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=x, cmap='jet')
        plt.colorbar(im, location='bottom')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if L is not None:
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_zlim(0, L)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if ofile is not None:
        plt.savefig(ofile)
    if show:
        plt.show()
    plt.close()

def plot_xyz_gif(xyz, x, dir, ofile = 'xyz.gif', order = None, colors = None,
                fps = 2):
    filenames = []
    if order is None:
        order = range(len(xyz))
    for i in order:
        fname = osp.join(dir, f'{i}.png')
        filenames.append(fname)
        plot_xyz(xyz[i, :, :], None, x = x, ofile = fname, show = False,
                    title = None, legend = False, colors = colors)

    # build gif
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    imageio.mimsave(osp.join(dir, ofile), frames, format='GIF', fps=fps)

    # remove files
    for filename in set(filenames):
        os.remove(filename)

### Primary scripts ###
def plot_matrix_gif(arr, dir, ofile = None, title = None, vmin = 0, vmax = 1,
                    size_in = 6, minVal = None, maxVal = None, prcnt = False,
                    cmap = None, x_ticks = None, y_ticks = None):
    filenames = []
    for i in range(len(arr)):
        fname=osp.join(dir, f'{i}.png')
        filenames.append(fname)
        plot_matrix(arr[i,], fname, ofile, title, vmin, vmax, size_in, minVal,
                    maxVal, prcnt, cmap, x_ticks, y_ticks)

    # build gif
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    imageio.mimsave(ofile, frames, format='GIF', fps=1)

    # remove files
    for filename in set(filenames):
        os.remove(filename)

def plotting_script(model, opt, train_loss_arr = None, val_loss_arr = None,
                    dataset = None, samples = None):
    '''Core plotting script for trained model.'''
    if model is None:
        model, train_loss_arr, val_loss_arr = load_saved_model(opt, verbose = False,
                                                                throw = False)
    if model is not None and dataset is None:
        dataset = get_dataset(opt, names = True, minmax = True, samples = samples)
        opt.valN = len(dataset)
        dataloader_fn = torch_geometric.loader.DataLoader
        val_dataloader = dataloader_fn(dataset, batch_size = 1, shuffle = False,
                                        num_workers = opt.num_workers)
    else:
        opt.batch_size = 1 # batch size must be 1
        opt.shuffle = False # for reproducibility
        _, val_dataloader, _ = get_data_loaders(dataset, opt)


    imagePath = opt.ofile_folder
    print('#### Plotting Script ####', file = opt.log_file)
    plot_loss(train_loss_arr, val_loss_arr, imagePath, opt)
    plot_loss(train_loss_arr, val_loss_arr, imagePath, opt, True)

    loss_dict = {}
    if opt.plot_predictions:
        if opt.output_mode.startswith('energy'):
            loss = plotEnergyPredictions(val_dataloader, model, opt)
