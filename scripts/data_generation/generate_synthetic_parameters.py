'''
Generate synthetic parameters used for simulated training data. This script
can be called from the command line using argparse if desired.
'''

import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict

import numpy as np
from pylib.utils.energy_utils import calculate_diag_chi_step
from pylib.utils.utils import LETTERS, load_import_log, triu_to_full
from sklearn.neighbors import KernelDensity
from utils.utils import ROOT, get_samples


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser', allow_abbrev = False)

    # input data args
    parser.add_argument('--root', type=str,
                        help='root directory')
    parser.add_argument('--dataset', type=str,
                        help='output dataset')
    parser.add_argument('--k', type=int,
                        help='number of marks in max ent')
    parser.add_argument('--ar', type=float,
                        help='aspect ratio for spheroid boundary')
    parser.add_argument('--samples', type=int,
                        help='number of samples')
    parser.add_argument('--m', type=int,
                        help='number of beads')
    parser.add_argument('--data_dir', type=str,
                    help='where data will be found when running simulation')
    parser.add_argument('--exp_dataset', type=str, default='dataset_02_04_23',
                    help='dataset where experimental data is located')
    parser.add_argument('--cell_line', type=str,
                    help='cell_line to filter experimental data to')
    parser.add_argument('--b', type=int, default=140,
                        help='bond length')
    parser.add_argument('--phi', type=float,
                    help='phi chromatin')
    parser.add_argument('--v', type=int,
                    help='simulation volume')
    parser.add_argument('--conv_defn', type=str, default='loss')

    args = parser.parse_args()
    return args

class DatasetGenerator():
    def __init__(self, args):
        self.N = args.samples
        self.m = args.m
        self.k = args.k
        self.root = args.root
        self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.exp_dataset = args.exp_dataset
        self.cell_line = args.cell_line
        self.b = args.b
        self.phi = args.phi
        self.v = args.v
        self.conv_defn = args.conv_defn

        if self.phi is not None:
            assert self.v is None
            self.grid_root = f'optimize_grid_b_{self.b}_phi_{self.phi}'
            self.distributions_root = f'b_{self.b}_phi_{self.phi}'
        else:
            self.grid_root = f'optimize_grid_b_{self.b}_v_{self.v}'
            self.distributions_root = f'b_{self.b}_v_{self.v}'


        if args.ar != 1:
            self.grid_root += f'_spheroid_{args.ar}'
            self.distributions_root += f'_spheroid_{args.ar}'
        self.distributions_root += '_distributions'
        if self.cell_line is not None:
            self.distributions_root += f'_{self.cell_line}'
        print(f'Using {self.distributions_root}')

        self.max_ent_root = f'{self.grid_root}-max_ent{self.k}'

        self.get_exp_samples()

        odir = osp.join(self.root, self.dataset)
        os.makedirs(odir, exist_ok=True)
        self.odir = osp.join(odir, 'setup')
        os.makedirs(self.odir, exist_ok=True)

        self.exp_dir =  osp.join(self.root, self.exp_dataset, 'samples')

        # sample : dictionary of params
        self.sample_dict = defaultdict(dict)
        for i in range(self.N):
            self.sample_dict[i]['m'] = self.m
            if args.ar != 1:
                self.sample_dict[i]['boundary_type'] = 'spheroid'
                self.sample_dict[i]['aspect_ratio'] = args.ar

    def get_exp_samples(self):
        if self.cell_line is not None:
            samples, cell_lines = get_samples(self.exp_dataset, train=True,
                                                return_cell_lines=True,
                                                filter_cell_lines=set([self.cell_line]))
        else:
            samples, cell_lines = get_samples(self.exp_dataset, train=True, return_cell_lines=True)
        self.exp_samples = samples
        print(f'Using {len(self.exp_samples)} samples: {self.exp_samples}')
        print(f'Using cell lines: {set(cell_lines)}')

    def get_converged_samples(self):
        converged_samples = []
        for i in self.exp_samples:
            sample_folder = osp.join(self.exp_dir, f'sample{i}', f'{self.grid_root}-max_ent{self.k}')
            converged = False

            # check convergence
            if self.conv_defn == 'loss':
                convergence_file = osp.join(sample_folder, 'convergence.txt')
                eps = 1e-2
                if osp.exists(convergence_file):
                    conv = np.loadtxt(convergence_file)
                    for j in range(1, len(conv)):
                        diff = conv[j] - conv[j-1]
                        if np.abs(diff) < eps and conv[j] < conv[0]:
                            converged = True
                else:
                    print(f'Warning: {convergence_file} does not exist')
            elif self.conv_defn == 'param':
                all_chis = []
                all_diag_chis = []
                for j in range(31):
                    it_path = osp.join(sample_folder, f'iteration{j}')
                    if osp.exists(it_path):
                        config_file = osp.join(it_path, 'production_out/config.json')
                        with open(config_file) as f:
                            config = json.load(f)
                        chis = np.array(config['chis'])
                        chis = chis[np.triu_indices(len(chis))] # grab upper triangle
                        diag_chis = np.array(config['diag_chis'])

                        all_chis.append(chis)
                        all_diag_chis.append(diag_chis)

                params = np.concatenate((all_diag_chis, all_chis), axis = 1)

                convergence = []
                eps = 1e2
                for j in range(5, len(params)):
                    diff = params[j] - params[j-1]
                    diff = np.linalg.norm(diff, ord = 2)
                    if diff < eps:
                        converged = True


            if converged:
                converged_samples.append(i)
            else:
                print(f'sample{i} did not converge')

        print('converged %:', len(converged_samples) / len(self.exp_samples) * 100)

        return converged_samples

    def setup_seq_eig(self):
        x_dict = {} # id : x
        for j in self.exp_samples:
            sample_folder = osp.join(self.exp_dir,  f'sample{j}')
            assert osp.exists(sample_folder)
            max_ent_folder = osp.join(sample_folder, f'{self.max_ent_root}/resources')
            assert osp.exists(max_ent_folder), f'{max_ent_folder} does not exist'
            x = np.load(osp.join(max_ent_folder, 'x_eig_norm.npy'))
            x_dict[j] = x

        for i in range(self.N):
            j = np.random.choice(self.exp_samples)
            x = x_dict[j]
            x = x[:, :self.k]

            seq_file = osp.join(self.odir, f'x_{i+1}.npy')
            np.save(seq_file, x)

            seq_file = osp.join(self.data_dir, self.dataset, f'setup/x_{i+1}.npy')
            self.sample_dict[i]['method'] = seq_file

    def setup_chi(self):
        for i in range(self.N):
            self.sample_dict[i]['k'] = self.k
            if self.k == 0:
                self.sample_dict[i]['chi_method'] = 'none'
                continue

            # eignorm approach
            chi_ii = np.zeros(self.k)
            for j in range(self.k):
                l = LETTERS[j]
                with open(osp.join(self.root, self.exp_dataset, self.distributions_root,
                                    'plaid_param_distributions_eig_norm',
                                    f'k{self.k}_chi{l}{l}_KDE.pickle'), 'rb') as f:
                    kde = pickle.load(f) # KernelDensity object
                chi_ii[j] = kde.sample(1).reshape(-1)
            chi_ij = np.zeros(int(self.k*(self.k-1)/2))

            chi = np.zeros((self.k, self.k))
            np.fill_diagonal(chi, chi_ii)
            chi[np.triu_indices(self.k, 1)] = chi_ij
            chi = chi + np.triu(chi, 1).T

            chi_file = osp.join(self.odir, f'chi_{i+1}.npy')
            np.save(chi_file, chi)

            chi_file = osp.join(self.data_dir, self.dataset, f'setup/chi_{i+1}.npy')
            self.sample_dict[i]['chi_method'] = chi_file

    def setup_chi_dist(self):
        diag_dict = {} # id : diag_params
        grid_dict = {} # id : grid_size

        converged_samples = self.get_converged_samples()
        print(converged_samples, len(converged_samples))
        for j in converged_samples:
            sample_folder = osp.join(self.exp_dir, f'sample{j}', f'{self.grid_root}-max_ent{self.k}')
            diag_chis = np.loadtxt(osp.join(sample_folder, 'chis_diag.txt'))
            with open(osp.join(sample_folder, 'resources/config.json'), 'r') as f:
                config = json.load(f)
            diag_chi_step = calculate_diag_chi_step(config, diag_chis)
            diag_dict[j] = diag_chi_step

            # get grid_size
            grid_file = osp.join(self.exp_dir, f'sample{j}', f'{self.grid_root}/grid.txt')
            grid_dict[j] = np.loadtxt(grid_file)

        for i in range(self.N):
            j = np.random.choice(converged_samples)
            diag_chis = diag_dict[j]

            diag_file = osp.join(self.odir, f'diag_chis_{i+1}.npy')
            np.save(diag_file, diag_chis)

            diag_file = osp.join(self.data_dir, self.dataset, f'setup/diag_chis_{i+1}.npy')

            self.sample_dict[i]['diag_chi_experiment'] = osp.join(self.exp_dataset,
                                                                f'samples/sample{j}',
                                                                f'{self.grid_root}')
            self.sample_dict[i]['diag_chi_method'] = diag_file
            self.sample_dict[i]['diag_bins'] = self.m
            self.sample_dict[i]['grid_size'] = grid_dict[j]


    def get_dataset(self):
        self.setup_seq_eig()
        self.setup_chi()
        self.setup_chi_dist()

        # write to odir
        print(f'Writing to {self.odir}')
        for i in range(self.N):
            ofile = osp.join(self.odir, f'sample_{i+1}.txt')
            with open(ofile, 'w') as f:
                for key, val in self.sample_dict[i].items():
                    f.write(f'--{key}\n{val}\n')

def main(args=None):
    if args is None:
        args = getArgs()
        args.dataset = 'dataset_08_02_24_imr90_test'
        args.exp_dataset = 'dataset_12_06_23-small'
        args.samples = 10 # number of samples to generate
        args.k = 10
        args.m = 512
        args.b=200; args.v=8; args.ar=1.5
        args.cell_line='imr90'
        args.root = ROOT
        args.data_dir = "/home/erschultz/"

    generator = DatasetGenerator(args)
    generator.get_dataset()

if __name__ == '__main__':
    main()
