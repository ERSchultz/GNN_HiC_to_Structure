'''Functions for setting up simulations with different parameters/parametrizations.'''

import argparse
import json
import math
import os
import os.path as osp
import re
import subprocess as sp
import sys
from collections import defaultdict
from time import sleep

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.linalg
import scipy.ndimage as ndimage
import torch
import torch_geometric
from pylib.utils.ArgparseConverter import ArgparseConverter
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.load_utils import load_L, load_psi, load_Y, load_Y_diag
from pylib.utils.plotting_utils import (plot_matrix, plot_seq_binary,
                                        plot_seq_continuous)
from pylib.utils.utils import LETTERS, crop, load_json, triu_to_full
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA, KernelPCA

from .InteractionConverter import InteractionConverter
from .knightRuiz import knightRuiz


def getArgs(args_file=None):
    parser = argparse.ArgumentParser(description='Base parser',
                                fromfile_prefix_chars='@',
                                allow_abbrev = False)
    AC = ArgparseConverter()

    # input data args
    parser.add_argument('--data_folder', type=str,
                        help='location of input data')
    parser.add_argument('--sample', type=str, default='40',
                        help='sample id')
    parser.add_argument('--sample_folder', type=str,
                        help='location of input data')
    parser.add_argument('--args_file', type=AC.str2None,
                        help='file with more args to load')

    # standard args
    parser.add_argument('--m', type=int, default=1024,
                        help='number of particles (will crop contact map) (-1 to infer)')
    parser.add_argument('--k', type=int,
                        help='sequences to generate')
    parser.add_argument('--plot', action='store_true',
                        help='true to plot seq as .png')

    # config args
    parser.add_argument('--config_ifile', type=str, default='default_config.json',
                        help='path to default config file')
    parser.add_argument('--config_ofile', type=str, default='config.json',
                            help='path to output config file')


    args, unknown = parser.parse_known_args()
    if args.args_file is None and args_file is not None:
        args.args_file = args_file
    if args.args_file is not None:
        assert osp.exists(args.args_file), f'{args.args_file} does not exist'
        print(f'parsing {args.args_file}')
        argv = sys.argv.copy()
        argv.append(f'@{args.args_file}') # appending means args_file will override other args
        argv.pop(0) # remove program name
        args, unknown = parser.parse_known_args(argv)

    if args.sample_folder is None and args.data_folder is not None:
        args.sample_folder = osp.join(args.data_folder, 'samples', f'sample{args.sample}')

    if args.m == -1:
        # infer m
        x = load_psi(args.sample_folder, throw_exception = False)
        y, _ = load_Y(args.sample_folder, throw_exception = False)
        if x is not None:
            args.m, _ = x.shape
        elif y is not None:
            args.m, _ = y.shape
        print(f'Inferrred m = {args.m}')

    if osp.exists(args.config_ifile):
        with open(args.config_ifile, 'rb') as f:
            args.config = json.load(f)
    else:
        print(f'{args.config_ifile} does not exist')
        args.config = None

    return args, unknown

class GetSeq():
    def __init__(self, args=None, unknown_args=None, verbose=True, config=None):
        if args is None:
            assert unknown_args is None and config is not None
            self.m = config['nbeads']
            self.k = config['nspecies']
        else:
            self.m = args.m
            self.k = args.k
            self.sample_folder = args.sample_folder
            self.sample = args.sample
            self.plot = args.plot
            self.method_recognized = True # default value
            self.args_file = args.args_file
            self.verbose = verbose
            self._get_args(unknown_args)
            if not self.method_recognized:
                # method is None
                return

            self.set_up_seq()

    def _get_args(self, unknown_args):
        AC = ArgparseConverter()
        parser = argparse.ArgumentParser(description='Seq parser', fromfile_prefix_chars='@',
                                    allow_abbrev = False)
        parser.add_argument('--method', type=AC.str2None,
                            help='method for assigning particle types')
        parser.add_argument('--seq_seed', type=AC.str2int,
                            help='random seed for numpy (None for random)')
        parser.add_argument('--exclusive', type=AC.str2bool, default=False,
                            help='True to use mutually exusive label (for random method)')
        parser.add_argument('--cell_line', type=str, default='HCT116',
                            help='cell line (only used with method = epigenetic)')
        parser.add_argument('--resolution', type=int, default=50000,
                            help='contact map resolution for method = epigenetic')
        parser.add_argument('--chromosome', type=AC.str2None,
                            help='chromoome for method = epigenetic')
        parser.add_argument('--start', type=AC.str2int,
                            help='contact map start in basepair for method = epigenetic')
        parser.add_argument('--end', type=AC.str2int,
                            help='contact map end in basepair for method = epigenetic')

        parser.add_argument('--p_switch', type=AC.str2float, default=None,
                            help='probability to switch bead assignment (for method=random)')
        parser.add_argument('--lmbda', type=AC.str2float, default=0.8,
                            help='lambda for Markov matrix of method = random')
        parser.add_argument('--f', type=AC.str2float, default=0.5,
                            help='mark frequency for method = random')
        parser.add_argument('--kernel', type=str, default='poly',
                            help='kernel for kernel PCA')
        parser.add_argument('--scale_resolution', type=AC.str2int, default=1,
                            help="generate seq at higher resolution, "
                                "find average frequency at lower resolution")
                            # TODO rename and document better

        if self.verbose:
            print("\nSeq args:")
        if self.args_file is not None:
            assert osp.exists(self.args_file)
            print(f'parsing {self.args_file}')
            unknown_args.append(f'@{self.args_file}') # appending args_file will override other args
            args, _ = parser.parse_known_args(unknown_args)

        else:
            args, _ = parser.parse_known_args(unknown_args)
        if self.verbose:
            print(args)
        if args.method is None:
            self.method_recognized = False
            return
        self._process_method(args)

        args.labels = None
        args.X = None # X for silhouette_score

        self._check_args(args)
        self.args = args

        if self.args.start is None and self.sample_folder is not None:
            import_log_file = osp.join(self.sample_folder, 'import.log')
            if osp.exists(import_log_file):
                with open(import_log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        line = line.split('=')
                        if line[0] == 'chrom':
                            self.args.chromosome = line[1]
                        elif line[0] == 'start':
                            self.args.start = int(line[1])
                        elif line[0] == 'end':
                            self.args.end = int(line[1])
                        elif line[0] == 'resolution':
                            self.args.resolution = int(line[1])
        if self.verbose:
            print(self.args)

    def _check_args(self, args):
        if args.binarize:
            if args.method_base in {'k_means'}:
                print(f"{args.method} is binarized by default")
            if args.method_base in {'pca', 'rpca', 'pca_split'}:
                pass
            elif args.method_base in {'nmf'} or args.method_base.startswith('block'):
                args.exclusive = True # will also be exclusive
            else:
                raise Exception(f'binarize not yet supported for {args.method_base}')
        if args.normalize:
            if args.method_base in {'pca', 'rpca', 'pca_split'}:
                pass
            else:
                raise Exception(f'normalize not yet supported for {args.method_base}')
        if args.exclusive:
            if args.method_base in {'nmf'} or args.method_base.startswith('block'):
                args.binarize = True # will also be binary
            elif args.method_base in {'random'}:
                pass
            else:
                raise Exception(f'exclusive not yet supported for {args.method_base}')
        if args.scale_resolution != 1:
            assert args.method_base == 'random', f"{args.method_base} not supported yet"

    def _process_method(self, args):
        # default values
        args.input = None # input in {y, x, psi}
        args.binarize = False # True to binarize labels (not implemented for all methods)')
        args.binarize_threshold = None
        args.normalize = False # True to normalize labels to [0,1] (or [-1, 1] for some methods)
            # (not implemented for all methods)')
        args.scale = False # True to scale variance for PCA
        args.use_energy = False
        args.append_random = False # True to append random seq
        args.exp = False # (for RPCA) convert from log space back to original space
        args.diag = False # (for RPCA) apply diagonal processing
        args.add_constant = False # add constant seq (all ones)
        args.epi_data_type = None # Type of data for experimental ChIP-seq
        args.log_inf = False # True to use log_inf before diag (only implemented for PCA)

        if osp.exists(args.method):
            return


        method_split = re.split(r'[-+]', args.method.lower())
        args.method_base = method_split.pop(0)

        for mode in set(method_split):
            if mode == 'x':
                args.input = 'x'
            elif mode == 'y':
                args.input = 'y'
            elif mode == 'psi':
                args.input = 'psi'
            elif mode.startswith('binarize'):
                if mode == 'binarize':
                    pass
                else:
                    args.binarize_threshold = mode[8:]
                    if args.binarize_threshold == 'mean':
                        pass
                    else:
                        args.binarize_threshold = float(args.binarize_threshold)
                args.binarize = True
            elif mode == 'normalize':
                args.normalize = True
            elif mode == 'scale':
                args.scale = True
            elif mode in {'s', 'l'}:
                args.use_energy = True
            elif mode == 'random':
                args.append_random = True
            elif mode == 'exp':
                args.exp = True
            elif mode == 'diag':
                args.diag = True
            elif mode == 'constant':
                args.add_constant = True
                self.k -= 1
            elif mode == 'fold_change_control':
                args.epi_data_type = mode
            elif mode == 'signal_p_value':
                args.epi_data_type = mode
            elif mode == 'log_inf':
                args.log_inf = True

    def get_random_seq(self, lmbda=None, f=0.5, p_switch=None,
                        seed=None, exclusive=False, scale_resolution=1):
        rng = np.random.default_rng(seed)
        assert (p_switch is not None) ^ (lmbda is not None)

        m = self.m * scale_resolution
        seq = np.zeros((m, self.k))

        if p_switch is not None:
            assert not scale_resolution, 'not supported'

            if exclusive:
                transition_probs = [1 - p_switch] # keep label with p = 1-p_switch
                transition_probs.extend([p_switch/(self.k-1)]*(self.k-1))
                # remaining transitions have to sum to p_switch

                ind = np.empty(m)
                ind[0] = rng.choice(range(self.k), size = 1)
                for i in range(1, m):
                    prev_label = ind[i-1]
                    other_labels = list(range(self.k))
                    other_labels.remove(prev_label)

                    choices = [prev_label]
                    choices.extend(other_labels)
                    ind[i] = rng.choice(choices, p=transition_probs)
                for row, col in enumerate(ind):
                    seq[row, int(col)] = 1
            else:
                seq[0, :] = rng.choice([1,0], size = self.k)
                for j in range(self.k):
                    for i in range(1, m):
                        if seq[i-1, j] == 1:
                            seq[i, j] = rng.choice([1,0], p=[1 - p_switch, p_switch])
                        else:
                            seq[i, j] = rng.choice([1,0], p=[p_switch, 1 - p_switch])
        else:
            # lmda is not None
            seq[0, :] = rng.choice([1,0], size = self.k)
            p11 = f*(1-lmbda)+lmbda
            assert p11 >= 0, f'p11={p11} for f={f}, lambda={lmbda}'
            p00 = f*(lmbda-1) + 1
            assert p00 >= 0, f'p00={p00} for f={f}, lambda={lmbda}'
            p01 = 1 - p11
            assert p01 >= 0, f'p01={p01} for f={f}, lambda={lmbda}'
            p10 = 1 - p00
            assert p10 >= 0, f'p10={p10} for f={f}, lambda={lmbda}'

            # p10 = f*(1-lmbda)
            # p00 = 1 - p10
            # p11 = 1 + lmbda - p00
            # p01 = 1 - p11
            # p_ij is prob j -> i

            for j in range(self.k):
                for i in range(1, m):
                    if seq[i-1, j] == 1:
                        # equals 1, need p11 or p01
                        seq[i, j] = rng.choice([1,0], p = [p11, p01])
                    else:
                        # equals 0, need p00 or p10
                        seq[i, j] = rng.choice([1,0], p = [p10, p00])

            if scale_resolution > 1:
                seq_high_resolution = seq.copy()
                seq = np.zeros((self.m, self.k))
                for i in range(self.m):
                    lower = i * scale_resolution
                    upper = lower + scale_resolution
                    slice = seq_high_resolution[lower:upper, :]
                    seq[i, :] = np.mean(slice, axis = 0)

        return seq

    def get_block_seq(self, method):
        '''
        Method should be formatted like "block-A100-B100"
        '''
        seq = np.zeros((self.m, self.k))
        method_split = re.split(r'[-+]', method)
        method_split.pop(0)
        lower_bead = 0
        letters = set()
        for s in method_split:
            letter = s[0].upper()
            letters.add(letter)
            label = LETTERS.find(letter)

            upper_bead = int(s[1:]) + lower_bead
            assert upper_bead <= self.m, f"too many beads: {upper_bead}"
            print(letter, lower_bead, upper_bead)

            seq[lower_bead:upper_bead, label] = 1
            lower_bead = upper_bead

        assert upper_bead == self.m, f"not enough beads: {upper_bead}"
        print(np.sum(seq, axis = 0))



        assert len(letters) == self.k, f"not enough letters ({letters}) for k = {self.k}"
        return seq

    def get_PCA_split_seq(self, input, normalize=False, binarize=False,
                        binarize_threshold=0.5, scale=False):
        '''
        Defines seq based on PCs of input.

        Inputs:
            input: matrix to perform PCA on
            normalize: True to normalize particle types / principal components to [0, 1]
            binarize: True to binarize PC vectors (will normalize and then set any value > binarize_threshold as 1)
            binarize_threshold: threshold for binarize, float or 'mean'
            scale: True to scale input before PCA

        Outputs:
            seq: array of particle types
        '''
        if binarize:
            normalize = True # reusing normalize code in binarize

        input = crop(input, self.m)
        pca = PCA()

        if scale:
            pca.fit(input/np.std(input, axis = 0))
        else:
            pca.fit(input)

        seq = np.zeros((self.m, self.k))
        j = 0
        PC_count = self.k // 2 # 2 seqs per PC
        for pc_i in range(PC_count):
            pc = pca.components_[pc_i]
            if normalize:
                min = np.min(pc)
                max = np.max(pc)
                if max > abs(min):
                    val = max
                else:
                    val = abs(min)

                # multiply by scale such that val x scale = 1
                scale = 1/val
                pc *= scale

            pcpos = pc.copy()
            pcpos[pc < 0] = 0 # set negative part to zero
            if binarize:
                # pc has already been normalized to [0, 1]ne, manual=False, soren=False):
                if isinstance(binarize_threshold, float):
                    val = binarize_threshold
                elif binarize_threshold == 'mean':
                    val = np.mean(pcpos)
                else:
                    raise Exception(f'issue with {binarize_threshold}')
                pcpos[pcpos <= val] = 0
                pcpos[pcpos > val] = 1
                print(pcpos)
            seq[:,j] = pcpos

            pcneg = pc.copy()
            pcneg[pc > 0] = 0 # set positive part to zero
            pcneg *= -1 # make positive
            if binarize:
                # pc has already been normalized to [0, 1]
                if isinstance(binarize_threshold, float):
                    val = binarize_threshold
                elif binarize_threshold == 'mean':
                    val = np.mean(pcneg)
                pcneg[pcneg <= val] = 0
                pcneg[pcneg > val] = 1
            seq[:,j+1] = pcneg


            j += 2
        return seq

    def get_PCA_seq(self, input, normalize=False, binarize=False, scale=False,
                    use_kernel=False, kernel=None, manual=False, soren=False,
                    randomized=False, smooth=False, h=3):
        '''
        Defines seq based on PCs of input.

        Inputs:
            input: matrix to perform PCA on
            normalize: True to normalize particle types / principal components to [-1, 1]
            binarize: True to binarize particle types (will normalize and then set any value > 0 as 1)
            use_kernel: True to use kernel PCA
            kernel: type of kernel to use

        Outputs:
            seq: array of particle types
        '''
        if binarize:
            normalize = True # reusing normalize code in binarize
        input = crop(input, self.m)
        if smooth:
            input = ndimage.gaussian_filter(input, (h, h))

        if use_kernel:
            assert kernel is not None
            pca = KernelPCA(kernel = kernel)
        else:
            pca = PCA()


        if manual:
            W, V = np.linalg.eig(np.corrcoef(input))
            V = V.T
        elif soren:
            U, S, V = scipy.linalg.svd(np.corrcoef(input))
        else:
            if scale:
                pca.fit(input/np.std(input, axis = 0))
            else:
                pca.fit(input)
            if use_kernel:
                V = pca.eigenvectors_.T
            else:
                V = pca.components_

        seq = np.zeros((self.m, self.k))
        for j in range(self.k):
            pc = V[j]
            if normalize:
                val = np.max(np.abs(pc))
                # multiply by scale such that val x scale = 1
                scale = 1/val
                pc *= scale

            if binarize:
                # pc has already been normalized to [-1, 1]
                pc[pc < 0] = 0
                pc[pc > 0] = 1

            seq[:,j] = pc

        return seq

    def get_k_means_seq(self, y, kr=True):
        y = crop(y, self.m)

        if kr:
            yKR = knightRuiz(y)
        kmeans = KMeans(n_clusters = self.k)
        try:
            kmeans.fit(yKR)
        except ValueError as e:
            print(e)
            print('Not using KR')
            kmeans.fit(y)
        seq = np.zeros((self.m, self.k))
        seq[np.arange(self.m), kmeans.labels_] = 1
        return seq, kmeans.labels_

    def get_nmf_seq(self, input, binarize):
        '''
        Applies NMF to input.

        Inputs:
            input: input matrix
            binarize: True to binarize NMF result (is exclusive as well)

        Outputs:
            seq: bead types
            labels: categorical labels for bead types (None if binarize is True)
        '''
        input = crop(input, self.m)

        nmf = NMF(n_components = self.k, max_iter = 1000, init=None)
        nmf.fit(input)
        H = nmf.components_

        print(f"NMF reconstruction error: {nmf.reconstruction_err_}")

        if binarize:
            nmf.labels_ = np.argmax(H, axis = 0)
            seq = np.zeros((self.m, self.k))
            seq[np.arange(self.m), nmf.labels_] = 1
            return seq, nmf.labels_
        else:
            seq = H.T
            return seq, None

    def get_epigenetic_seq(self, data_dir, start, end,
                            res, chr, sigmoid, min_coverage_prcnt=5):
        '''
        Loads experimental epigenetic data from data_folder to use as particle types.

        Inputs:
            data_dir: location of epigenetic data - file format: <chr>_*.npy
            start: start in base pairs
            end: end location in base pairs
            res: resolution of data/simulation
            chr: chromosome
            sigmoid: True to use sigmoid preprocessed ChIP-seq
            min_coverage_prcnt: minimum percent of particle of given particle type


        Outputs:
            seq: particle type np array of shape m x k
        '''
        if sigmoid:
            data_folder = osp.join(data_dir, 'processed_sigmoid')
        else:
            data_folder = osp.join(data_dir, 'processed')
        start = int(start / res)
        end = int(end / res)
        m = end - start # number of particles in simulation
        assert m == self.m, f'{m} != {self.m}'

        # store file names and coverage in list
        print(data_folder)
        file_list = [] # list of tuples (file_name, coverage)
        for file in os.listdir(data_folder):
            if file.startswith(f'chr{chr}_') and file.endswith('.npy'):
                seq_i = np.load(osp.join(data_folder, file))
                if sigmoid:
                    seq_i = seq_i[start:end+1]
                else:
                    seq_i = seq_i[start:end+1, 1] # crop to appropriate size
                coverage = np.sum(seq_i)
                file_list.append((file, coverage))
        print(file_list)

        if not sigmoid:
            # sort based on coverage
            file_list = sorted(file_list, key = lambda pair: pair[1], reverse = True)
            print(f'Top {self.k} marks with most coverage:\n\t{file_list[:self.k]}')

        # choose k marks with most coverage
        seq = np.zeros((self.m, self.k))
        marks = []
        for i, (file, coverage) in enumerate(file_list[:self.k]):
            mark = file.split('_')[1]
            marks.append(mark)
            if coverage < min_coverage_prcnt / 100 * m and not sigmoid:
                print(f"WARNING: mark {mark} has insufficient coverage: {coverage}")
            seq_i = np.load(osp.join(data_folder, file))
            if sigmoid:
                seq_i = seq_i[start:end]
            else:
                seq_i = seq_i[start:end, 1]
            seq[:, i] = seq_i
        i += 1
        if i < self.k:
            print(f"Warning: insufficient data - only {i} marks found")

        return seq, marks

    def get_ChromHMM_seq(self, ifile, start, end, res, chr,
                        min_coverage_prcnt=5):
        # check resolution
        split = ifile.split(osp.sep)
        dir = osp.join('/', *split[:-2])
        print(dir)
        with open(osp.join(dir, f'webpage_{self.k}.html')) as f:
            for line in f:
                if line.startswith('Full ChromHMM command'):
                    split = line.split(' ')
                    assert int(split[6]) == res, f'resolution mismatch {split[6]} != {res}'
                    break

        start = int(start / res)
        end = int(end / res)
        m = end - start # number of particles in simulation
        assert m == self.m, f"m != m, {m} != {self.m}"

        with open(ifile, 'r') as f:
            f.readline() # headler lines
            f.readline()
            states = np.array([int(state.strip()) - 1 for state in f.readlines()])
            # subtract 1 to convert to 0 based indexing
            states = states[start:end] # crop to size of contact map

        seq = np.zeros((self.m, self.k))
        seq[np.arange(self.m), states] = 1 # one hot encoding of states

        coverage_arr = np.sum(seq, axis = 0) # number of beads of each particle type

        # sort based on coverage
        insufficient_coverage = np.argwhere(coverage_arr < min_coverage_prcnt * m / 100).flatten()

        # exclude marks with no coverage
        for state in range(self.k):
            if state in insufficient_coverage:
                print(f"State {state} coverage: {coverage_arr[state]} (insufficient)")
            else:
                print(f"State {state} coverage: {coverage_arr[state]}")

        # get labels
        labels = np.where(seq == np.ones((m, 1)))[1]

        return seq, labels

    def get_seq_gnn(self, model_path, sample, normalize, scale):
        # deprecated
        '''
        Loads output from GNN model to use as particle types, seq

        Inputs:
            model_path: path to model results
            sample: sample id (int)
            normalize: True to normalize seq to [-1,1] (only for ContactGNNEnergy)

        Outputs:
            seq: particle types
        '''
        model_type = osp.split(osp.split(model_path)[0])[1]
        print(model_type)

        if model_type == 'ContactGNN':
            z_path = osp.join(model_path, f"sample{sample}/z.npy")
            if osp.exists(z_path):
                seq = np.load(z_path)
                assert seq.shape[1] == self.k
            else:
                raise Exception(f'z_path does not exist: {z_path}')
        elif model_type == 'ContactGNNEnergy':
            energy_hat_path = osp.join(model_path, f"sample{sample}/energy_hat.txt")
            if osp.exists(energy_hat_path):
                energy_hat = np.loadtxt(energy_hat_path)
            else:
                raise Exception(f's_path does not exist: {energy_hat_path}')

            seq = self.get_PCA_seq(energy_hat, normalize, binarize, scale)
        else:
            raise Exception(f"Unrecognized model_type: {model_type}")

        return seq

    def set_up_seq(self):
        args = self.args
        if self.k == 0:
            print('k is 0, returning')
            self.method_recognized = False
            return

        elif osp.exists(args.method):
            if args.method.endswith('.txt'):
                seq = np.loadtxt(args.method)
            elif args.method.endswith('.npy'):
                seq = np.load(args.method)
            else:
                raise Exception(f'Unrecognized file format {args.chi_method}')
            seq = seq[:self.m, :]
        else:
            args.method = args.method.lower()
            print(f'Method lowercase: {args.method}')
            if args.method.startswith('soren'):
                seq = np.load(osp.join(self.sample_folder, 'x_soren.npy'))
                if seq.shape == (self.k, self.m):
                    # Soren is sometimes transposed
                    seq = seq.T
            elif args.method.startswith('random'):
                seq = self.get_random_seq(args.lmbda, args.f, args.p_switch,
                                            args.seq_seed, args.exclusive,
                                            args.scale_resolution)
            elif args.method.startswith('block'):
                seq = self.get_block_seq(args.method)
            elif args.method.startswith('pca_split'):
                y_diag = load_Y_diag(self.sample_folder)
                seq = self.get_PCA_split_seq(y_diag, args.normalize, args.binarize,
                                            args.binarize_threshold, args.scale)
            elif args.method.startswith('pca'):
                y, y_diag = load_Y(self.sample_folder)
                if args.log_inf:
                    y = np.log(y)
                    y[np.isinf(y)] = np.nan
                    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
                    print(y, y.shape)
                    y_diag = DiagonalPreprocessing.process(y, meanDist, verbose = False)
                    np.nan_to_num(y_diag, copy = False, nan = 1.0)

                seq = self.get_PCA_seq(y_diag, args.normalize, args.binarize, args.scale)
            elif args.method.startswith('rpca'):
                L_file = osp.join(self.sample_folder, 'PCA_analysis', 'L_log.npy')
                if osp.exists(L_file):
                    L = np.load(L_file)
                    if args.exp:
                        L = np.exp(L)
                    if args.diag:
                        meanDist = DiagonalPreprocessing.genomic_distance_statistics(L)
                        L = DiagonalPreprocessing.process(L, meanDist)
                    seq = self.get_PCA_seq(L, args.normalize, args.binarize, args.scale)
                else:
                    y = np.load(osp.join(self.sample_folder, 'y.npy'))
                    seq = self.get_RPCA_seq(y, args.normalize, args.exp, args.diag)
            elif args.method.startswith('kpca'):
                if args.input == 'y':
                    input = load_Y_diag(self.sample_folder)
                elif args.input == 'x':
                    input = np.load(osp.join(self.sample_folder, 'x.npy'))
                elif args.input == 'psi':
                    input = np.load(osp.join(self.sample_folder, 'psi.npy'))
                seq = self.get_PCA_seq(input, args.normalize, args.binarize,
                                        args.scale, use_kernel = True, kernel = args.kernel)
            elif args.method.startswith('ground_truth'):
                x = load_psi(self.sample_folder, throw_exception = False)

                if args.input is None:
                    assert args.use_energy, 'missing input'
                elif args.input == 'x' or args.input == 'psi':
                    assert x is not None
                    seq = x
                    print(f'seq loaded with shape {seq.shape}')

                else:
                    raise Exception(f'Unrecognized input mode {args.input} for method {args.method} '
                                    f'for sample {self.sample_folder}')

                if args.append_random:
                    # TODO this is broken
                    assert not args.use_energy
                    _, k = seq.shape
                    assert args.k > k, f"{args.k} not > {k}"
                    seq_random = GetSeq(args.m, args.k - k).get_random_seq(args.lmbda,
                                                args.f, args.p_switch, args.seq_seed)
                    seq = np.concatenate((seq, seq_random), axis = 1)

                if args.use_energy:
                    L = load_L(self.sample_folder, psi)
            elif args.method.startswith('k_means') or args.method.startswith('k-means'):
                y_diag = np.load(osp.join(self.sample_folder, 'y_diag.npy'))
                seq, args.labels = self.get_k_means_seq(y_diag)
                args.X = y_diag
            elif args.method.startswith('nmf'):
                y_diag = np.load(osp.join(self.sample_folder, 'y_diag.npy'))
                seq, args.labels = self.get_nmf_seq(y_diag, args.binarize)
                args.X = y_diag
            else:
                print(f'Unkown method: {args.method}')
                self.method_recognized = False
                return


        m, k = seq.shape
        assert m == self.m, f"m mismatch: seq has {m} particles not {self.m}"
        if self.k > 0:
            assert k == self.k, f'''k mismatch: seq has {k} particle types not {self.k}
                                for method {args.method} for sample {self.sample_folder}'''

        if args.add_constant:
            seq = np.concatenate((seq, np.ones((m, 1))), axis = 1)

        np.save('x.npy', seq.astype(np.float64))
        print(f"Seq[0,:]: {seq[0,:]}")

        if self.plot:
            if args.method_base in {'k_means'}:
                # plot_seq_exclusive(seq, labels=args.labels, X=args.X)
                plot_seq_binary(seq)
            elif args.binarize:
                if args.method.startswith('pca_split'):
                    plot_seq_binary(seq, split = True)
                else:
                    plot_seq_binary(seq)
            else:
                plot_seq_continuous(seq)

class GetPlaidChi():
    def __init__(self, args, unknown_args):
        self.k = args.k
        self.m = args.m
        self.args_file = args.args_file
        self._get_args(unknown_args)
        self.set_up_plaid_chi()

    def _get_args(self, unknown_args):
        AC = ArgparseConverter()
        parser = argparse.ArgumentParser(description='Plaid chi parser', fromfile_prefix_chars='@',
                                    allow_abbrev = False)
        parser.add_argument('--chi', type=AC.str2list2D,
                            help='chi matrix using latex separator style'
                                '(None to generate chi with chi_method)')
        parser.add_argument('--chi_method', type=AC.str2None, default='random',
                            help='method for generating chi if not given')
        parser.add_argument('--min_chi', type=float, default=-1.,
                            help='minimum chi value for random generation')
        parser.add_argument('--max_chi', type=float, default=1.,
                            help='maximum chi value for random generation')
        parser.add_argument('--fill_diag', type=AC.str2float,
                            help='fill diag of chi with given value (None to skip)')
        parser.add_argument('--fill_offdiag', type=AC.str2float,
                            help='fill off diag of chi with given value (None to skip)')
        parser.add_argument('--ensure_distinguishable', action='store_true',
                            help='true to ensure that corresponding psi is distinguishable')
        parser.add_argument('--chi_seed', type=AC.str2int,
                            help='seed for generating chi (None for random)')
        parser.add_argument('--chi_constant', type=AC.str2float, default=0,
                            help='constant to add to chi')
        parser.add_argument('--chi_multiplier', type=AC.str2float, default=1,
                            help='multiplier to multiply by chi')

        print('\nChi args')
        if self.args_file is not None:
            assert osp.exists(self.args_file)
            print(f'parsing {self.args_file}')
            unknown_args.append(f'@{self.args_file}') # appending means args_file will override other args
            self.args, _ = parser.parse_known_args(unknown_args)

        else:
            self.args, _ = parser.parse_known_args(unknown_args)
        print(self.args)

    def set_up_plaid_chi(self):
        args = self.args
        if self.k == 0:
            print('k is 0, returning')
            self.method_recognized = False
            return
        elif self.k == -1:
            x = np.load('x.npy')
            _, self.k = x.shape
            print(f'k is -1, inferred {self.k}')

        if args.chi is not None:
            chi = np.triu(args.chi) # zero lower triangle
            rows, cols = chi.shape
            assert self.k == rows, f'number of particle types {self.k} does not match shape of chi {rows}'
            assert rows == cols, f"chi not square: {chi}"
            conv = InteractionConverter(self.k, chi)
            if not GetPlaidChi.unique_rows(conv.getS()):
                print('Warning: particles are not distinguishable')
        elif args.chi_method is None:
            chi = None
        elif osp.exists(args.chi_method):
            if args.chi_method.endswith('.txt'):
                chi = np.loadtxt(args.chi_method)
            elif args.chi_method.endswith('.npy'):
                chi = np.load(args.chi_method)
            else:
                raise Exception(f'Unrecognized file format {args.chi_method}')
            if len(chi.shape) == 1:
                chi = triu_to_full(chi)
                rows, cols = chi.shape
            else:
                rows, cols = chi.shape
                if rows > self.k and cols > self.k:
                    chi = triu_to_full(chi[-1])
                    rows, cols = chi.shape
            assert self.k == rows, f'number of particle types {self.k} does not match shape of chi {rows, cols}'
            assert rows == cols, f"chi not square: {chi}"
        else:
            args.chi_method = args.chi_method.lower()
            if args.chi_method in {'zero', 'zeros'}:
                chi = np.zeros((self.k, self.k))
            elif args.chi_method == 'random':
                chi = self.random_chi()
            elif args.chi_method.startswith('polynomial'):
                x = np.load('x.npy') # original particle types that interact nonlinearly
                ind = np.triu_indices(self.k)
                self.k = int(self.k*(self.k+1)/2)
                psi = np.zeros((self.m, self.k))
                for i in range(self.m):
                    psi[i] = np.outer(x[i], x[i])[ind]

                np.save('psi.npy', psi)
                chi = self.random_chi()
            else:
                raise Exception(f"Unrecognized chi_method: {args.chi_method}")

        # save chi
        if chi is not None:
            chi += args.chi_constant
            chi *= args.chi_multiplier
            print(f'Rank of chi: {np.linalg.matrix_rank(chi)}')
            np.save('chis.npy', chi)

    def _generate_random_chi(self, rng=np.random.default_rng(), decimals=1):
        '''Initializes random chi array.'''
        args = self.args

        # create array with random values in [minval, maxVal]
        rands = rng.random(size=(self.k, self.k)) * (args.max_chi - args.min_chi) + args.min_chi

        # zero lower triangle
        chi = np.triu(rands)

        if args.fill_offdiag is not None:
            # fills off diag chis with value of fill_offdiag
            diag_chi = np.diagonal(chi)
            chi = np.ones((args.k, args.k)) * args.fill_offdiag
            di = np.diag_indices(args.k)
            chi[di] = diag_chi
        if args.fill_diag is not None:
            # fills diag chis with value of fill_diag
            di = np.diag_indices(args.k)
            chi[di] = args.fill_diag

        return np.round(chi, decimals = decimals)

    def random_chi(self):
        args = self.args
        rng = np.random.default_rng(args.chi_seed)
        chi = self._generate_random_chi(rng)
        if args.ensure_distinguishable and self.k < 10: # if k is too large this is too RAM intensive
            conv = InteractionConverter(self.k, chi)
            max_it = 10
            it = 0
            while not GetPlaidChi.unique_rows(conv.getL()) and it < max_it: # defaults to False
                # generate random chi
                conv.chi = _generate_random_chi(rng)
                it += 1
            if it == max_it:
                print('Warning: maximum iteration reached')
                print('Warning: particles are not distinguishable')
            chi = conv.chi

        return chi

    @staticmethod
    def unique_rows(array):
        if array is None:
            return False

        if len(np.unique(array, axis=0)) / len(array) == 1.:
            return True
        else:
            return False

class GetDiagChi():
    def __init__(self, args, unknown_args):
        self.sample_folder = args.sample_folder
        self.m = args.m
        self.config = args.config
        self.args_file = args.args_file
        self._get_args(unknown_args)
        self.set_up_diag_chi()

        self.config['diag_start'] = self.args.diag_start

    def _get_args(self, unknown_args):
        AC = ArgparseConverter()
        parser = argparse.ArgumentParser(description='Diag chi parser', fromfile_prefix_chars='@',
                                    allow_abbrev = False)
        parser.add_argument('--diag_chi', type=AC.str2list,
                            help='diag chi (None to generate via diag_chi_method)')
        parser.add_argument('--diag_bins', type=AC.str2int, default=20,
                            help='number of diagonal bins for diag_chi_method')
        parser.add_argument('--diag_chi_method', type=AC.str2None, default='linear',
                            help='method for generating diag_chi if not given'
                                '(None for no diag_chi)')
        parser.add_argument('--diag_chi_slope', type=float,
                            help='slope (in thousandths) for diag_chi_method = log')
        parser.add_argument('--diag_chi_scale', type=AC.str2float,
                            help='scale (in thousandths) for diag_chi_method = log')
        parser.add_argument('--dense_diagonal_on', type=AC.str2bool, default=False,
                            help='True to use dense_diagonal')
        parser.add_argument('--logarithmic_diagonal_on', type=AC.str2bool, default=False,
                            help='True to use logarithmic diagonal')
        parser.add_argument('--dense_diagonal_cutoff', type=AC.str2float, default=1/16,
                            help='cutoff = nbeads * dense_diagonal_cutoff')
        parser.add_argument('--dense_diagonal_loading', type=AC.str2float, default=0.5,
                            help='proportion of beads to place left of cutoff')
        parser.add_argument('--small_binsize', type=int, default=0,
                            help='specify small_binsize instead of using dense_diagonal_cutoff')
        parser.add_argument('--big_binsize', type=int, default=-1,
                            help='specify big_binsize instead of using dense_diagonal_cutoff')
        parser.add_argument('--n_small_bins', type=int, default=0,
                            help='specify n_small_bins instead of using dense_diagonal_loading')
        parser.add_argument('--n_big_bins', type=int, default=-1,
                            help='specify n_big_bins instead of using dense_diagonal_loading')
        parser.add_argument('--max_diag_chi', type=AC.str2float,
                            help='maximum diag chi value for diag_chi_method')
        parser.add_argument('--min_diag_chi', type=float, default=0,
                            help='minimum diag chi value for diag_chi_method (currently only supported for logistic and linear)')
        parser.add_argument('--diag_chi_midpoint', type=float, default=0,
                            help='midpoint for logistic diag chi')
        parser.add_argument('--diag_chi_constant', type=AC.str2float, default=0,
                            help='constant to add to chi diag')
        parser.add_argument('--mlp_model_path', type=str,
                            help='path to MLP model')
        parser.add_argument('--m_continuous', type=AC.str2int,
                            help='Use m larger than self.m to define diag chis then crop')
        parser.add_argument('--diag_start', type=int, default=0,
                            help='minimum d to use diag chi')

        print('\nDiag chi args:')
        if self.args_file is not None:
            assert osp.exists(self.args_file)
            print(f'parsing {self.args_file}')
            unknown_args.append(f'@{self.args_file}') # appending means args_file will override other args
            self.args, _ = parser.parse_known_args(unknown_args)

        else:
            self.args, _ = parser.parse_known_args(unknown_args)
        if self.args.m_continuous is None:
            self.args.m_continuous = self.m

        print(self.args)

    def set_up_diag_chi(self):
        args = self.args
        diag_chis_continuous = None
        if args.diag_chi_method is not None:
            self.get_bin_sizes()
            self.d_arr = np.arange(args.m_continuous)
            if args.diag_chi_slope is not None:
                args.diag_chi_slope /= 1000


            if osp.exists(args.diag_chi_method):
                if args.diag_chi_method.endswith('txt'):
                    temp = np.loadtxt(args.diag_chi_method)
                else:
                    temp = np.load(args.diag_chi_method)
                if len(temp.shape) == 2:
                    # assume this is maxent diag chis and take last iteration
                    temp = temp[-1]
                if len(temp) == args.diag_bins:
                    diag_chis = temp
                elif len(temp) == args.m_continuous:
                    diag_chis_continuous = temp
                elif len(temp) > args.m_continuous:
                    # crop to m_continuous
                    diag_chis_continuous = temp[:args.m_continuous]
                else:
                    raise Exception(f'{len(temp)} != {args.diag_bins} != {args.m_continuous}')
            else:
                args.diag_chi_method = args.diag_chi_method.lower()
                if args.diag_chi_method in {'zero', 'zeros'}:
                    diag_chis = np.zeros(args.diag_bins)
                elif args.diag_chi_method == 'linear':
                    if args.max_diag_chi is not None:
                        diag_chis_continuous = np.linspace(args.min_diag_chi,
                                                            args.max_diag_chi,
                                                            args.m_continuous) + args.diag_chi_constant
                    elif args.diag_chi_slope is not None:
                        diag_chis_continuous = self.d_arr * args.diag_chi_slope + args.diag_chi_constant
                elif args.diag_chi_method == 'logistic':
                    num = (args.max_diag_chi - args.min_diag_chi)
                    denom = (1 + np.exp(-1*args.diag_chi_slope * (self.d_arr - args.diag_chi_midpoint)))
                    diag_chis_continuous = num / denom + args.min_diag_chi
                elif args.diag_chi_method.startswith('log'):
                    if args.diag_chi_scale is None:
                        args.diag_chi_scale = args.max_diag_chi / np.log(args.diag_chi_slope * (args.m_continuous - 1) + 1)
                    diag_chis_continuous = args.diag_chi_scale * np.log(args.diag_chi_slope * self.d_arr + 1)
                    diag_chis_continuous += args.diag_chi_constant
                    if args.diag_chi_method == 'logmax':
                        diag_chis_continuous[diag_chis_continuous < 0] = 0
                elif args.diag_chi_method == 'exp':
                    middle = 1.889 * np.exp(-args.diag_chi_slope * self.d_arr)
                    diag_chis_continuous = args.max_diag_chi - middle + args.diag_chi_constant
                elif args.diag_chi_method == 'mlp':
                    diag_chis_continuous, diag_chis = self.get_diag_chi_mlp(args.mlp_model_path, self.sample_folder)
                elif args.diag_chi_method == 'ground_truth':
                    assert self.sample_folder is not None, 'sample_folder is None'
                    sample_config_file = osp.join(self.sample_folder, 'config.json')
                    assert osp.exists(sample_config_file), f'ground truth config missing at {sample_config_file}'
                    with open(sample_config_file, 'rb') as f:
                        sample_config = json.load(f)

                    if not sample_config["diagonal_on"]:
                        print('WARNING: ground truth diagonal is off')
                        # assume that simulation was run with net energy matrix
                    diag_chis = np.array(sample_config["diag_chis"])

                    # override args
                    args.diag_bins = len(diag_chis)

                    # override any relevant config args
                    self.config["dense_diagonal_on"] = sample_config["dense_diagonal_on"]
                    self.config['n_small_bins'] = sample_config['n_small_bins']
                    self.config['n_big_bins'] = sample_config['n_big_bins']
                    self.config['small_binsize'] = sample_config['small_binsize']
                    self.config['big_binsize'] = sample_config['big_binsize']
                else:
                    raise Exception(f'Unrecognized chi diag method {args.diag_chi_method}')

            if diag_chis_continuous is not None:
                print('diag_chis_continuous:', diag_chis_continuous, diag_chis_continuous.shape)
                diag_chis = self.coarse_grain_diag_chi(diag_chis_continuous)
                np.save('diag_chis_continuous', diag_chis_continuous)
        elif args.diag_chi is not None:
            diag_chis = np.array(args.diag_chi)
        else:
            return

        assert len(diag_chis) == args.diag_bins, f"Shape mismatch: {len(diag_chis)} vs {args.diag_bins}"
        print('diag_chis: ', diag_chis, diag_chis.shape)
        np.save('diag_chis.npy', diag_chis)
        self.config["diag_chis"] = list(diag_chis) # ndarray not json serializable

    def get_bin_sizes(self):
        args = self.args
        print(f'm = {self.m}')
        if args.dense_diagonal_on:
            if args.dense_diagonal_loading is not None:
                self.n_small_bins = int(args.dense_diagonal_loading * args.diag_bins)
                assert args.diag_bins > self.n_small_bins, f"{args.diag_bins} < {self.n_small_bins}"
                self.n_big_bins = args.diag_bins - self.n_small_bins
            else:
                self.n_small_bins = args.n_small_bins
                self.n_big_bins = args.n_big_bins

            if args.dense_diagonal_cutoff is not None:
                self.dividing_line = self.m * args.dense_diagonal_cutoff
                self.small_binsize = int(self.dividing_line / (self.n_small_bins))
                self.big_binsize = int((self.m - self.dividing_line) / self.n_big_bins)
            else:
                self.small_binsize = args.small_binsize
                self.dividing_line = self.small_binsize * self.n_small_bins
                self.big_binsize = args.big_binsize

            if self.n_big_bins == -1:
                remainder = self.m - self.dividing_line
                print('remainder', remainder)
                self.n_big_bins = math.floor(args.diag_bins - self.n_small_bins)
                print('n_big_bins', self.n_big_bins)
                while remainder % self.n_big_bins != 0 and self.n_big_bins < remainder:
                    print(remainder % self.n_big_bins)
                    self.n_big_bins += 1

                self.big_binsize = remainder // self.n_big_bins
                if self.n_small_bins + self.n_big_bins != args.diag_bins:
                    print(f'args.diag_bins changed from {args.diag_bins} to {self.n_small_bins + self.n_big_bins}')
                    args.diag_bins = self.n_small_bins + self.n_big_bins

            result_string = f'{self.n_small_bins}x{self.small_binsize} + {self.n_big_bins}x{self.big_binsize} = {self.m}'
            assert self.n_small_bins * self.small_binsize + self.n_big_bins * self.big_binsize == self.m, result_string
            print(result_string)

            self.config['n_small_bins'] = self.n_small_bins
            self.config['n_big_bins'] = self.n_big_bins
            self.config['small_binsize'] = self.small_binsize
            self.config['big_binsize'] = self.big_binsize
        else:
            self.binsize = self.m / args.diag_bins
            print(f'binsize = {self.binsize}')
            assert self.m % args.diag_bins == 0, f'{self.m}%{args.diag_bins}!=0'
            self.config['n_small_bins'] = 0
            self.config['n_big_bins'] = args.diag_bins
            self.config['small_binsize'] = 0
            self.config['big_binsize'] = self.binsize

    def coarse_grain_diag_chi(self, diag_chis_continuous):
        args = self.args

        # get diag chis
        i = 0
        diag_chis = np.zeros(args.diag_bins)
        curr_bin_vals = []
        prev_bin = 0
        for d, val in enumerate(diag_chis_continuous):
            if args.dense_diagonal_on:
                if d > self.dividing_line:
                    bin = self.n_small_bins + math.floor((d - self.dividing_line) / self.big_binsize)
                else:
                    bin =  math.floor(d / self.small_binsize)
            else:
                bin = int(d / self.binsize)

            curr_bin = bin
            if curr_bin != prev_bin:
                prev_bin = curr_bin
                diag_chis[i] = np.mean(curr_bin_vals)
                curr_bin_vals = []
                i += 1
                if i >= len(diag_chis):
                    return diag_chis
            curr_bin_vals.append(val)
        diag_chis[i] = np.mean(curr_bin_vals)
        diag_chis = np.round(diag_chis, 5)

        return diag_chis

def main(args_file=None):
    args, unknown = getArgs(args_file)
    print(args)
    getSeq = GetSeq(args, unknown)
    assert getSeq.method_recognized
    GetPlaidChi(args, unknown)
    GetDiagChi(args, unknown)

    with open(args.config_ofile, 'w') as f:
        json.dump(args.config, f, indent = 2)
