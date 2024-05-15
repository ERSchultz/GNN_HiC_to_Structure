'''
Run simulations using synthetic parameters to generate simulated training data.
This script can be called from the command line using argparse if desired.
'''
import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import imageio.v2 as imageio
import numpy as np
import pylib.analysis as analysis
import utils.get_params as get_params
from pylib.Pysim import Pysim
from pylib.utils import default, utils
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import (calculate_D, calculate_diag_chi_step,
                                      calculate_U)
from pylib.utils.load_utils import load_L, load_max_ent_L
from pylib.utils.plotting_utils import (plot_diag_chi, plot_matrix,
                                        plot_mean_dist,
                                        plot_mean_vs_genomic_distance)
from pylib.utils.utils import crop
from sklearn.metrics import mean_squared_error
from utils.utils import get_config


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser',
                                    fromfile_prefix_chars='@',
                                    allow_abbrev = False)
    parser.add_argument('--start', type=int,
                        help='first sample')
    parser.add_argument('--end', type=int,
                        help='last sample')
    parser.add_argument('--jobs', type=int,
                        help='number of jobs')
    parser.add_argument('--odir_start', type=str, default='',
                        help='''prefix for odir (useful if running multiple
                            variations of data generation procedure)''')
    parser.add_argument('--scratch', type=str,
                        help='absolute path to scratch')
    parser.add_argument('--data_folder', type=str,
                        help='absolute path to dataset')
    parser.add_argument('--overwrite', action='store_true',
                        help='True to overwrite')
    parser.add_argument('--m', type=int,
                        help='num particles')

    args, _ = parser.parse_known_args()
    return args

def plot_all(args):
    if args.random_mode:
        y_path = osp.join(args.sample_folder, 'production_out')
        if not osp.exists(y_path):
            print(f'{y_path} does not exist')
            y_path = args.sample_folder

    else:
        y_path = osp.join(args.final_folder, "production_out")
        if not osp.exists(y_path):
            y_path = args.final_folder
    assert osp.exists(y_path), f'{y_path} does not exist'

    # get y
    y_file = osp.join(y_path, 'contacts.txt')
    if osp.exists(y_file):
        y = crop(np.loadtxt(y_file), args.m)
        args.m = len(y)
    else:
        max_sweep = -1
        # look for contacts{sweep}.txt
        for file in os.listdir(y_path):
            if file.startswith('contacts') and file.endswith('.txt'):
                sweep = int(file[8:-4])
                if sweep > max_sweep:
                    max_sweep = sweep
        if max_sweep > 0:
            y = crop(np.loadtxt(osp.join(y_path, f'contacts{max_sweep}.txt')), args.m)
        else:
            raise Exception(f"y path does not exist: {y_path}")

    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    y_diag = DiagonalPreprocessing.process(y, meanDist)

    # get L
    if args.random_mode:
        L = load_L(args.sample_folder, throw_exception = False)
    else:
        L = load_max_ent_L(args.replicate_folder)

    # get config
    if args.random_mode:
        with open(osp.join(args.sample_folder, 'config.json'), 'r') as f:
            config = json.load(f)
    else:
        with open(osp.join(args.final_folder, 'config.json'), 'r') as f:
            config = json.load(f)

    # diag chi
    diag_chi = None
    all_diag_chis = None
    diag_chi_step = None
    D = None
    if args.random_mode:
        file = osp.join(args.sample_folder, 'diag_chis.npy')
        if osp.exists(file):
            diag_chi = np.load(file)
            if len(diag_chi.shape) > 1:
                diag_chi = diag_chi[-1]
            file = osp.join(args.sample_folder, 'diag_chis_continuous.npy')
            if osp.exists(file):
                diag_chi_ref = np.load(file)
            else:
                diag_chi_ref = None


    elif args.replicate_folder is not None:
        file = osp.join(args.replicate_folder, 'chis_diag.txt')
        if osp.exists(file):
            all_diag_chis = np.loadtxt(file)
            diag_chi = np.atleast_2d(all_diag_chis)[-1]
            diag_chi_ref = None
    if diag_chi is not None:
        diag_chi_step = calculate_diag_chi_step(config, diag_chi)
        D = calculate_D(diag_chi_step)

    # get S
    U = None
    if L is not None:
        U = calculate_U(L, D)
    else:
        U_file = osp.join(args.sample_folder, 'U.npy')
        if osp.exists(U_file):
            U = np.load(U_file)

    if args.plot:
        plot_matrix(y, ofile = osp.join(args.save_folder, 'y.png'), vmax = 'mean')
        p = y / np.mean(np.diagonal(y))
        plot_matrix(p, ofile = osp.join(args.save_folder, 'p.png'), vmax = 'mean')

        if diag_chi is not None:
            plot_diag_chi(config, args.save_folder,
                            ref = diag_chi_ref, ref_label = 'continuous')
            plot_diag_chi(config, args.save_folder,
                            ref = diag_chi_ref, ref_label = 'continuous',
                            logx = True)

        # plot gif of diag chis
        if all_diag_chis is not None:
            files = []
            ylim = (np.min(all_diag_chis), np.max(all_diag_chis))
            for i in range(1, len(all_diag_chis)):
                diag_chi_i = all_diag_chis[i]
                file = f'{i}.png'
                diag_chi_step_i = calculate_diag_chi_step(config, diag_chi_i)
                plot_diag_chi(None, args.save_folder,
                                logx = True, ofile = file,
                                diag_chis_step = diag_chi_step_i, ylim = ylim,
                                title = f'Iteration {i}')
                files.append(osp.join(args.save_folder, file))

            frames = []
            for filename in files:
                frames.append(imageio.imread(filename))

            imageio.mimsave(osp.join(args.save_folder, 'pchis_diag_step.gif'), frames, format='GIF', fps=2)

            # remove files
            for filename in files:
                os.remove(filename)

        # plot energy matrices
        if args.m < 5000:
            # takes a long time for large m and not really necessary
            plot_matrix(y_diag, ofile = osp.join(args.save_folder, 'y_diag.png'),
                        vmin = 'center1', cmap='blue-red')

            if U is not None:
                plot_matrix(U, ofile = osp.join(args.save_folder, 'U.png'), title = 'S',
                            vmax = 'max', vmin = 'min', cmap = 'blue-red')

            if L is not None:
                plot_matrix(L, ofile = osp.join(args.save_folder, 'L.png'), title = 'L',
                            vmax = 'max', vmin = 'min', cmap = 'blue-red')

        # meanDist
        if args.random_mode:
            y_gt_file = osp.join(args.sample_folder, 'resources', 'y_gt.npy')
            if osp.exists(y_gt_file):
                y_gt = np.load(y_gt_file)
                meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(y_gt, 'prob')
                meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
                print('meanDist_gt', meanDist_gt.shape)
                rmse = mean_squared_error(meanDist, meanDist_gt, squared = False)
                title = f'RMSE: {np.round(rmse, 9)}'
                plot_mean_dist(meanDist, args.save_folder, 'meanDist_log_ref.png',
                                diag_chi_step, True, meanDist_gt, 'Reference', 'Sim',
                                'blue', title)
                plot_mean_dist(meanDist, args.save_folder, 'meanDist_ref.png',
                                diag_chi_step, False, meanDist_gt, 'Reference', 'Sim',
                                'blue', title)

            plot_mean_vs_genomic_distance(y, args.save_folder, 'meanDist.png',
                                            diag_chi_step)
            plot_mean_vs_genomic_distance(y, args.save_folder, 'meanDist_log.png',
                                            diag_chi_step, logx = True)

        else:
            meanDist_max_ent = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
            print('meanDist_max_ent', meanDist_max_ent[:10])
            if args.replicate_folder is not None:
                y_gt_file = osp.join(args.replicate_folder, 'resources', 'y_gt.npy')
            else:
                y_gt_file = osp.join(osp.split(args.save_folder)[0],  'y.npy')
            if osp.exists(y_gt_file):
                y_gt = np.load(y_gt_file)
                meanDist_gt = DiagonalPreprocessing.genomic_distance_statistics(y_gt, 'prob')
                print('meanDist_gt', meanDist_gt[:10])
                rmse = mean_squared_error(meanDist_max_ent, meanDist_gt, squared = False)
                title = f'RMSE: {np.round(rmse, 9)}'
            else:
                meanDist_gt = None
                title = None

            if args.final_it == 1:
                sim_label = 'GNN'
                color = 'green'
            else:
                sim_label = 'Max Ent'
                color = 'blue'
            plot_mean_dist(meanDist_max_ent, args.save_folder, 'meanDist.png',
                            diag_chi_step, False, ref = meanDist_gt,
                            ref_label = 'Reference',  label = sim_label,
                            color = color, title = title)
            plot_mean_dist(meanDist_max_ent, args.save_folder, 'meanDist_log.png',
                            diag_chi_step, True, ref = meanDist_gt,
                            ref_label = 'Reference',  label = sim_label,
                            color = color, title = title)

    if args.save_npy:
        np.save(osp.join(args.save_folder, 'y.npy'), y.astype(np.int16))
        np.save(osp.join(args.save_folder, 'y_diag.npy'), y_diag)
        if U is not None:
            np.save(osp.join(args.save_folder, 'U.npy'), U)
        if L is not None:
            np.save(osp.join(args.save_folder, 'L.npy'), L)

def check_dir(odir, overwrite):
    if osp.exists(odir):
        if overwrite:
            print(f'Overwriting {odir}')
            shutil.rmtree(odir)
        else:
            return True

    return False

def run(args, config_args, i):
    odir = osp.join(args.data_folder, f'samples/sample{i}')
    abort = check_dir(odir, args.overwrite)
    if abort:
        return

    odir_scratch = osp.join(args.scratch, f'{args.odir_start}{i}')
    if osp.exists(odir_scratch):
        shutil.rmtree(odir_scratch)
    os.mkdir(odir_scratch, mode=0o755)
    os.chdir(odir_scratch)

    defaults = '/home/erschultz/TICG-chromatin/defaults'
    shutil.copyfile(osp.join(defaults, 'config_erschultz.json'),
                    osp.join(odir_scratch, 'default_config.json'))

    args_file = osp.join(args.data_folder, f'setup/sample_{i}.txt')


    stdout = sys.stdout
    with open(osp.join(odir_scratch, 'params.log'), 'w') as sys.stdout:
        get_params.main(args_file)
    sys.stdout = stdout

    stdout = sys.stdout
    with open(osp.join(odir_scratch, 'config.log'), 'w') as sys.stdout:
        get_config(args_file, config_args)
    sys.stdout = stdout

    stdout = sys.stdout
    with open(osp.join(odir_scratch, 'log.log'), 'w') as sys.stdout:
        config = utils.load_json('config.json')

        # get sequences
        if osp.exists('x.npy'):
            seqs = np.load('x.npy')
        else:
            seqs = None

        sim = Pysim('', config, seqs, randomize_seed = False, mkdir = False)

        print('Running Simulation')
        sim.run_eq(10000, config['nSweeps'], 1)
    sys.stdout = stdout

    stdout = sys.stdout
    with open(osp.join(odir_scratch, 'contact_map.log'), 'w') as sys.stdout:
        args.save_npy = True
        args.random_mode = True
        args.plot = True
        args.save_folder = ''
        args.sample_folder = ''
        plot_all(args)
    sys.stdout = stdout

    os.remove('default_config.json')
    shutil.move(odir_scratch, odir)

def main(args=None, config_args=None):
    if args is None:
        args = getArgs()
    mapping = []
    for i in range(args.start, args.end+1):
        mapping.append((args, config_args, i))

    with mp.Pool(args.jobs) as p:
        p.starmap(run, mapping)

if __name__ == '__main__':
    main()
