import json
import math
import os
import os.path as osp
import sys
import time
from shutil import rmtree

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import torch
import torch_geometric.data
import torch_geometric.transforms
import torch_geometric.utils
from pylib.utils import epilib
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_D, calculate_S
from scipy.ndimage import uniform_filter
from skimage.measure import block_reduce
from sklearn.cluster import KMeans
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_std

from ..argparse_utils import finalize_opt, get_base_parser
from ..knightRuiz import knightRuiz
from ..load_utils import load_L, load_Y
from ..utils import rescale_matrix
from .dataset_classes import DiagFunctions, make_dataset
from .networks import get_model


def plaid_score(y, y_diag):
    def c(y, a, b):
        return a@y@b

    def r(y, a, b):
        denom = c(y,a,b)**2
        num = c(y,a,a) * c(y,b,b)
        return num/denom

    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(y_diag)
    m = len(y)
    seq = np.zeros((m, 2))
    seq[np.arange(m), kmeans.labels_] = 1

    return r(y, seq[:, 1], seq[:, 0])

class ContactsGraph(torch_geometric.data.Dataset):
    # How to backprop through model after converting to GNN:
    # https://github.com/rusty1s/pytorch_geometric/issues/1511
    def __init__(self, dirname, scratch, root_name=None,
                m=1024, y_preprocessing='diag',
                y_log_transform=None, kr=False, rescale=None, mean_filt=None,
                y_norm='mean', min_subtraction=True,
                use_node_features=True, mlp_model_id=None,
                sparsify_threshold=None, sparsify_threshold_upper=None,
                split_neg_pos_edges=False, max_diagonal=None,
                transform=None, pre_transform=None, output='contact',
                crop=None, ofile=sys.stdout, verbose=True,
                max_sample=float('inf'), samples=None, sub_dir='samples',
                plaid_score_cutoff=None, sweep_choices=[2,3,4,5],
                diag=False, corr=False, eig=False,
                keep_zero_edges=False, output_preprocesing=None,
                bonded_root = None):
        '''
        Inputs:
            dirname: directory path to raw data (or list of paths)
            scratch: path to scratch (used for root)
            root_name: directory for loaded data
            m: number of particles/beads
            y_preprocessing: type of contact map preprocessing ('diag', None, etc)
            y_log_transform: type of log transform (int k for log_k, 'ln' for ln, None to skip)
            kr: True to balance with knightRuiz algorithm
            rescale: rescale contact map by factor of <rescale> (None to skip)
                    e.g. 2 will decrease size of contact mapy by 2
            mean_filt: apply mean filter of width <mean_filt> (None to skip)
            y_norm: type of normalization ('mean', 'max')
            min_subtraction: True to subtract min during normalization
            use_node_features: True to use bead labels as node features
            mlp_model_id: id for mlp diagonal parameters (can be used as edge attr)
            sparsify_threshold: lower threshold for sparsifying contact map (None to skip)
            sparsify_threshold_upper: upper threshold for sparsifying contact map (None to skip)
            split_neg_pos_edges: True to split negative and positive edges for training
            max_diagonal: maximum diagonal of adjacency matrix to consider
            transform: list of transforms
            pre_transform: list of transforms
            output: output mode ('contact', 'energy', 'energy_sym')
            crop: tuple of crop sizes
            ofile: where to print to if verbose == True
            verbose: True to print
            max_sample: max sample id to save
            samples: set of samples to include (None for all)
            plaid_score_cutoff: contact maps with plaid_score > cutoff are ignored
            sweep_choices: choices for num_sweeps for sweeprand y_preprocessing
            diag: True if y_diag should be calculated
            keep_zero_edges: True to keep edges with 0 weight
            output_preprocesing: Type of preprocessing for prediction target
        '''
        t0 = time.time()
        self.m = m
        self.dirname = dirname
        if isinstance(dirname, list):
            dirname = dirname[0]
        self.y_preprocessing = y_preprocessing
        self.y_log_transform = y_log_transform
        self.kr = kr
        self.rescale = rescale
        self.mean_filt = mean_filt
        self.y_norm = y_norm
        # self.min_subtraction = min_subtraction # deprecated
        self.use_node_features = use_node_features
        self.mlp_model_id = mlp_model_id
        self.sparsify_threshold = sparsify_threshold
        self.sparsify_threshold_upper = sparsify_threshold_upper
        self.split_neg_pos = split_neg_pos_edges
        self.max_diagonal = max_diagonal
        self.output = output
        self.crop = crop
        self.samples = None
        self.sweep_choices = sweep_choices
        self.num_edges_list = [] # list of number of edges per graph
        self.degree_list = [] # created in self.process()
        self.verbose = verbose
        self.diag = diag
        self.corr = corr
        self.eig = eig
        self.keep_zero_edges = keep_zero_edges
        self.output_preprocesing = output_preprocesing
        self.bonded_root = bonded_root


        self.file_paths = make_dataset(self.dirname, maxSample = max_sample,
                                        samples = samples, sub_dir = sub_dir)
        if plaid_score_cutoff is not None:
            for f in self.file_paths:
                y, y_diag = load_Y(f)
                y /= np.mean(np.diagonal(y))
                score = plaid_score(y, y_diag)
                if score > plaid_score_cutoff:
                    self.file_paths.remove(f)


        if root_name is None:
            # find any currently existing graph data folders
            # make new folder for this dataset
            max_val = -1
            for file in os.listdir(scratch):
                file_path = osp.join(scratch, file)
                if file.startswith('graphs') and osp.isdir(file_path):
                    # format is graphs<i> where i is integer
                    val = int(file[6:])
                    if val > max_val:
                        max_val = val
            self.root = osp.join(scratch, f'graphs{max_val+1}')
        else:
            # use exsting graph data folder
            self.root = osp.join(scratch, root_name)

        super(ContactsGraph, self).__init__(self.root, transform, pre_transform)
        # the super(...).__init__(...) calls self.process() and runs the pre_transforms

        if verbose:
            print('Dataset construction time: '
                    f'{np.round((time.time() - t0) / 60, 3)} minutes', file = ofile)
            print(f'Number of samples: {self.len()}', file = ofile)

            print('Average num edges per graph: ',
                    f'{np.mean(self.num_edges_list)}', file = ofile)
            # print('Num edges per graph: ',
            #         f'{self.num_edges_list}', file = ofile)
            print('Average num edges per graph: ',
                    f'{np.mean(self.num_edges_list)}')

            if self.degree_list:
                # self.degree_list will be None if loading already processed dataset
                self.degree_list = np.array(self.degree_list)
                mean_deg = np.round(np.mean(self.degree_list, axis = 1), 2)
                std_deg = np.round(np.std(self.degree_list, axis = 1), 2)
                print(f'Mean degree: {mean_deg} +- {std_deg}\n', file = ofile)

    @property
    def raw_file_names(self):
        return self.file_paths

    @property
    def processed_file_names(self):
        return [f'graph_{i}.pt' for i in range(self.len())]

    def process(self):
        for i, raw_folder in enumerate(self.raw_file_names):
            self.process_y(raw_folder)
            self.process_diag_params(raw_folder) # used for calculating S = L+D

            edge_index, pos_edge_index, neg_edge_index = self.generate_edge_index()

            if self.use_node_features:
                graph = torch_geometric.data.Data(x = x, edge_index = edge_index)
            else:
                graph = torch_geometric.data.Data(x = None, edge_index = edge_index)

            graph.path = raw_folder
            graph.num_nodes = self.m
            graph.pos_edge_index = pos_edge_index
            graph.neg_edge_index = neg_edge_index
            graph.mlp_model_id = self.mlp_model_id
            graph.sweep = self.sweep
            graph.seqs = self.seqs

            # copy these temporarily
            graph.weighted_degree = self.weighted_degree
            graph.contact_map = self.contact_map # created by process_y
            graph.contact_map_diag = self.contact_map_diag
            graph.contact_map_corr = self.contact_map_corr
            graph.contact_map_bonded = self.contact_map_bonded
            graph.diag_chi_continuous = self.diag_chis_continuous


            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            del graph.weighted_degree # no longer needed

            if self.output != 'contact':
                del graph.contact_map
            del graph.contact_map_diag
            del graph.contact_map_corr
            del graph.contact_map_bonded

            if self.output is None:
                pass
            elif self.output == 'diag_chi_continuous':
                graph.y = graph.diag_chi_continuous
                if self.crop is not None:
                    graph.y = graph.y[self.crop[0]:self.crop[1]]
            elif self.output.startswith('energy_diag'):
                D = calculate_D(graph.diag_chi_continuous)
                if self.crop is not None:
                    D = D[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]
                graph.energy = torch.tensor(D, dtype = torch.float32)
            elif self.output.startswith('energy_sym'):
                energy = load_L(raw_folder)
                energy = torch.tensor(energy, dtype = torch.float32)
                graph.energy = (energy + energy.t()) / 2 # symmetric

                if self.output.startswith('energy_sym_diag'):
                    D = calculate_D(graph.diag_chi_continuous)
                    if self.crop is not None:
                        D = D[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]
                    graph.energy += torch.tensor(D, dtype = torch.float32)
            else:
                raise Exception(f'Unrecognized output {self.output}')

            if self.output is not None and self.output_preprocesing is not None:
                if 'center' in self.output_preprocesing:
                    graph.energy -= torch.mean(graph.energy)
                if 'norm' in self.output_preprocesing:
                    graph.energy /= torch.max(torch.abs(graph.energy))
                if 'log' in self.output_preprocesing:
                    graph.energy = torch.sign(graph.energy) * torch.log(torch.abs(graph.energy)+1)

            del graph.diag_chi_continuous

            torch.save(graph, self.processed_paths[i])

            # record degree
            if self.verbose:
                deg = np.array(torch_geometric.utils.degree(graph.edge_index[0],
                                                            graph.num_nodes))
                self.degree_list.append(deg)

    def load_y(self, raw_folder):
        '''Helper function to load raw contact map and apply normalization.'''
        self.sweep = None
        if self.y_preprocessing.startswith('sweeprand'):
            _, *y_preprocessing = self.y_preprocessing.split('_')
            if isinstance(y_preprocessing, list):
                preprocessing = '_'.join(y_preprocessing)

            id = int(osp.split(raw_folder)[1][6:])
            rng = np.random.default_rng(seed = id)
            sweep = rng.choice(self.sweep_choices, 1)[0]
            self.sweep = int(sweep * 100000)

            y_path = osp.join(raw_folder, f'data_out/contacts{self.sweep}.txt')
            y_path2 = osp.join(raw_folder, f'production_out/contacts{self.sweep}.txt')
            if osp.exists(y_path):
                y = np.loadtxt(y_path).astype(np.float64)
            elif osp.exists(y_path2):
                y = np.loadtxt(y_path2).astype(np.float64)
            else:
                raise Exception(f"y_path missing: {y_path}, {y_path2}")

        elif self.y_preprocessing.startswith('sweep'):
            sweep, *y_preprocessing = self.y_preprocessing.split('_')
            self.sweep = int(sweep[5:])
            if isinstance(y_preprocessing, list):
                preprocessing = '_'.join(y_preprocessing)

            y_path = osp.join(raw_folder, f'data_out/contacts{self.sweep}.txt')
            y_path2 = osp.join(raw_folder, f'production_out/contacts{self.sweep}.txt')
            if osp.exists(y_path):
                y = np.loadtxt(y_path).astype(np.float64)
            elif osp.exists(y_path2):
                y = np.loadtxt(y_path2).astype(np.float64)
            else:
                raise Exception(f"y_path missing: {y_path}, {y_path2}")
        elif self.y_preprocessing.startswith('rescale'):
            rescale, *y_preprocessing = self.y_preprocessing.split('_')
            rescale = int(rescale[7:])
            if isinstance(y_preprocessing, list):
                preprocessing = '_'.join(y_preprocessing)

            y = np.load(osp.join(raw_folder, 'y.npy')).astype(np.float64)
            y = y * rescale
        else:
            y = np.load(osp.join(raw_folder, 'y.npy')).astype(np.float64)
            preprocessing = self.y_preprocessing

        return y, preprocessing

    def process_y(self, raw_folder):
        '''
        Helper function to load the appropriate contact map and apply any
        necessary preprocessing.
        '''
        y, preprocessing = self.load_y(raw_folder)

        # get bonded contact map
        split = raw_folder.split(os.sep)
        dir = '/' + osp.join(*split[:-3])
        dataset = split[-3]
        sample = split[-1][6:]
        setup_file = osp.join(dir, dataset, f'setup/sample_{sample}.txt')
        bonded_file1 = osp.join(raw_folder, self.bonded_root, 'y.npy')
        bonded_file2 = osp.join(raw_folder, osp.split(self.bonded_root)[0],'y.npy')

        y_bonded = None
        if osp.exists(setup_file):
            found = False
            with open(setup_file) as f:
                for line in f:
                    line = line.strip()
                    if line == '--diag_chi_experiment':
                        exp_subpath = f.readline().strip()
                        found = True
            if not found:
                raise Exception(f'--diag_chi_experiment missing from {setup_file}')
            y_bonded_file = osp.join(dir, exp_subpath, 'y.npy')
            assert osp.exists(y_bonded_file), y_bonded_file
            y_bonded = np.load(y_bonded_file).astype(np.float64)
        elif osp.exists(bonded_file1):
            y_bonded = np.load(bonded_file1).astype(np.float64)
        elif osp.exists(bonded_file2):
            y_bonded = np.load(bonded_file2).astype(np.float64)
        assert y_bonded is not None, f'{setup_file}, {bonded_file1}'

        if self.mean_filt is not None:
            y = uniform_filter(y, self.mean_filt)

        # get eigvecs from y before rescale
        if self.eig:
            seqs = epilib.get_pcs(epilib.get_oe(y), 10, normalize=True).T
            # TODO hard-coded 10
            self.seqs = torch.tensor(seqs, dtype=torch.float32)
        else:
            self.seqs = None

        if self.rescale is not None:
            y = rescale_matrix(y, self.rescale)
            y_bonded = rescale_matrix(y_bonded, self.rescale)

        for y_i in [y, y_bonded]:
            if y_i is None:
                continue
            if self.y_norm == 'max':
                y_i /= np.max(y_i)
            elif self.y_norm == 'mean':
                y_i /= np.mean(np.diagonal(y_i))
            elif self.y_norm == 'mean_fill':
                y_i /= np.mean(np.diagonal(y_i))
                np.fill_diagonal(y_i, 1)

        if self.kr:
            y = knightRuiz(y)
            y_bonded = knightRuiz(y_bonded)

        y_copy = np.copy(y)
        if preprocessing == 'log':
            y = np.log(y+1)
            if y_bonded is not None:
                y_bonded = np.log(y_bonded+1)
        elif preprocessing == 'log_inf':
            with np.errstate(divide = 'ignore'):
                # surpress warning, infs handled manually
                y = np.log(y)
            y[np.isinf(y)] = np.nan
            if y_bonded is not None:
                with np.errstate(divide = 'ignore'):
                    y_bonded = np.log(y_bonded)
                y_bonded[np.isinf(y_bonded)] = np.nan
        elif preprocessing is not None:
            raise Exception('Deprecated')
            # override y
            assert self.y_norm is None, f'y_norm={self.y_norm} not None, preprocessing={preprocessing}'
            y_path = osp.join(raw_folder, f'y_{preprocessing}.npy')
            if osp.exists(y_path):
                y = np.load(y_path)
            else:
                raise Exception(f"Unknown preprocessing: {preprocessing} or y_path missing: {y_path}")

        if self.diag:
            # y is now log(contact map)
            # use y_copy instead
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_copy)
            y_diag = DiagonalPreprocessing.process(y_copy, meanDist, verbose = False)
            # y_diag = np.nan_to_num(y_diag)
        else:
            y_diag = None

        if self.corr:
            y_corr = np.corrcoef(y_diag)
        else:
            y_corr = None

        if self.crop is not None:
            y = y[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]
            if y_diag is not None:
                y_diag = y_diag[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]
            if y_bonded is not None:
                y_bonded = y_bonded[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]
            if y_corr is not None:
                y_corr = y_corr[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]

        if self.max_diagonal is not None:
            y = np.tril(y, self.max_diagonal)
            y = np.triu(y, -self.max_diagonal)

        if self.y_log_transform is not None:
            raise Exception('Deprecated')
            assert not self.y_preprocessing.endswith('log'), "don't use log twice in a row"
            if self.y_log_transform == 'ln':
                y = np.log(y)
            elif self.y_log_transform.isdigit():
                val = int(self.y_log_transform)
                if val == 2:
                    y = np.log2(y)
                elif val == 10:
                    y = np.log10(y)
                else:
                    raise Exception(f'Unaccepted log base: {val}')
            else:
                raise Exception(f'Unrecognized log transform: {self.y_log_transform}')

            y[np.isinf(y)] = 0 # since we didn't add a constant to y before the log

        if self.sparsify_threshold is not None:
            y[np.abs(y) < self.sparsify_threshold] = np.nan
        if self.sparsify_threshold_upper is not None:
            y[np.abs(y) > self.sparsify_threshold_upper] = np.nan

        # self.plotDegreeProfile(y)
        self.contact_map = torch.tensor(y, dtype = torch.float32)
        if y_diag is not None:
            self.contact_map_diag = torch.tensor(y_diag, dtype = torch.float32)
        else:
            self.contact_map_diag = None
        if y_bonded is not None:
            self.contact_map_bonded = torch.tensor(y_bonded, dtype = torch.float32)
        else:
            self.contact_map_bonded = None

        if y_corr is not None:
            self.contact_map_corr = torch.tensor(y_corr, dtype = torch.float32)
        else:
            self.contact_map_corr = None

    def process_diag_params(self, raw_folder):
        path = osp.join(raw_folder, 'diag_chis_continuous.npy')
        path2 = osp.join(raw_folder, 'diag_chis.npy')
        if osp.exists(path):
            diag_chis_gt = torch.tensor(np.load(path), dtype = torch.float32)
        elif osp.exists(path2):
            diag_chis_gt = torch.tensor(np.load(path2), dtype = torch.float32)
            # assert len(diag_chis_gt) == self.m, f'incorrect size {len(diag_chis_gt)} != {self.m}'
        else:
            diag_chis_gt = None
            if self.output is not None:
                raise Exception(f'chi_diag not found for {raw_folder}')

        self.diag_chis_continuous = diag_chis_gt

    def get(self, index):
         data = torch.load(self.processed_paths[index])
         return data

    def len(self):
        return len(self.raw_file_names)

    @property
    def weighted_degree(self):
        return torch.sum(self.contact_map, axis = 1)

    def generate_edge_index(self):
        adj = torch.clone(self.contact_map)
        if self.keep_zero_edges:
            adj[adj == 0] = 1 # first set zero's to some nonzero value
            adj = torch.nan_to_num(adj) # then replace nans with zero
            edge_index = adj.nonzero().t() # ignore remaining zeros
        else:
            adj = torch.nan_to_num(adj) # replace nans with zero
            edge_index = adj.nonzero().t() # ignore all zeros
        self.num_edges_list.append(edge_index.shape[1])
        if self.split_neg_pos:
            if self.y_log_transform:
                split_val = 0
            else:
                split_val = 1
            pos_edge_index = (self.contact_map > split_val).nonzero().t()
            neg_edge_index = (self.contact_map < split_val).nonzero().t()
        else:
            pos_edge_index = None
            neg_edge_index = None

        return edge_index, pos_edge_index, neg_edge_index

    def plotDegreeProfile(self):
        # y is weighted adjacency matrix
        ycopy = self.contact_map.copy()
        ycopy[y > 0] = 1
        ycopy[y < 0] = 1

        ypos = y.copy()
        ypos[y > 0] = 1
        ypos[y < 0] = 0

        yneg = y.copy()
        yneg[y > 0] = 0
        yneg[y < 0] = -1

        deg = np.sum(ycopy, axis = 0)
        degpos = np.sum(ypos, axis = 0)
        degneg = np.sum(yneg, axis = 0)
        print('min: ', np.min(deg), 'max: ', np.max(deg), ss.mode(deg))
        print('min: ', np.min(degpos), 'max: ', np.max(degpos), ss.mode(degpos))
        print('min: ', np.min(degneg), 'max: ', np.max(degneg), ss.mode(degneg))
        plt.hist(deg, bins = 100, label = 'deg')
        plt.ylabel('count', fontsize=16)
        plt.xlabel('degree', fontsize=16)
        plt.legend()
        plt.show()
        plt.close()
