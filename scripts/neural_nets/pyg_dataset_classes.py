import json
import math
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch_geometric.data
import torch_geometric.transforms
import torch_geometric.utils
from pylib.utils import epilib
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_D
from pylib.utils.load_utils import load_U
from scipy.ndimage import uniform_filter
from skimage.measure import block_reduce

from ..data_generation.utils.knightRuiz import knightRuiz


def rescale_matrix(inp, factor, triu=True):
    '''
    Rescales input matrix by factor.
    if inp is 1024x1024 and factor=2, out is 512x512
    '''
    if inp is None:
        return None
    assert len(inp.shape) == 2, f'must be 2d array not {inp.shape}'
    m, _ = inp.shape
    assert m % factor == 0, f'factor must evenly divide m {m}%{factor}={m%factor}'

    if triu:
        inp = np.triu(inp) # need triu to not double count entries
    processed = block_reduce(inp, (factor, factor), np.sum) # sum-pool operation

    if triu:
        # need to make symmetric again
        processed = np.triu(processed)
        out = processed + np.triu(processed, 1).T
        return out
    else:
        return processed

class ContactsGraph(torch_geometric.data.Dataset):
    # How to backprop through model after converting to GNN:
    # https://github.com/rusty1s/pytorch_geometric/issues/1511
    def __init__(self, file_paths, scratch, root_name,
                m, y_preprocessing,
                kr=False, rescale=2, mean_filt=None,
                hic_normalization='mean_fill',
                use_node_features=True,
                sparsify_threshold=None, sparsify_threshold_upper=None,
                transform=None, pre_transform=None, output_mode='contact',
                ofile=sys.stdout, verbose=True,
                diag=False, corr=False,
                keep_zero_edges=False,
                output_preprocesing=None):
        '''
        Inputs:
            file_paths: list of file_paths where data each sample is located
            scratch: path to scratch (used for root)
            root_name: directory name for saving/loading processed graphs
            m: number of particles/beads
            y_preprocessing: type of contact map preprocessing ('diag', None, etc)
            kr: True to balance with knightRuiz algorithm
            rescale: rescale contact map by factor of <rescale> (None to skip)
                    e.g. 2 will decrease size of contact mapy by 2
            mean_filt: apply mean filter to contact map with width <mean_filt> (None to skip)
            hic_normalization: type of normalization for contact map ('mean', 'max', 'mean_fill')
            use_node_features: True to use bead labels as node features
            sparsify_threshold: lower threshold for sparsifying contact map (None to skip)
            sparsify_threshold_upper: upper threshold for sparsifying contact map (None to skip)
            transform: list of transforms
            pre_transform: list of transforms
            output_mode: output mode (None, 'contact_map', 'umatrix')
            ofile: where to print to if verbose == True
            verbose: True to print
            diag: True if y_diag should be calculated (used in some transforms)
            corr: True if y_corr should be calculated (used in some transforms)
            keep_zero_edges: True to keep edges with 0 weight
            output_preprocesing: Type of preprocessing for prediction target
        '''
        t0 = time.time()
        self.file_paths = file_paths
        self.m = m
        self.y_preprocessing = y_preprocessing
        self.kr = kr
        self.rescale = rescale
        self.mean_filt = mean_filt
        self.hic_normalization = hic_normalization
        self.use_node_features = use_node_features
        self.sparsify_threshold = sparsify_threshold
        self.sparsify_threshold_upper = sparsify_threshold_upper
        self.output_mode = output_mode
        self.num_edges_list = [] # list of number of edges per graph
        self.degree_list = [] # created in self.process()
        self.verbose = verbose
        self.diag = diag
        self.corr = corr
        self.keep_zero_edges = keep_zero_edges
        self.output_preprocesing = output_preprocesing


        # set self.root - where processed graphs will be saved to
        if root_name is None:
            # set root_name to graphs<max_val+1>

            # first find any currently existing graph data folders
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
            self.process_hic(raw_folder)

            edge_index = self.generate_edge_index() # convert hic to edge_index

            if self.use_node_features:
                graph = torch_geometric.data.Data(x = x, edge_index = edge_index)
            else:
                graph = torch_geometric.data.Data(x = None, edge_index = edge_index)

            graph.path = raw_folder
            graph.num_nodes = self.m

            # copy these temporarily for use in pre_transforms
            graph.weighted_degree = self.weighted_degree
            graph.contact_map = self.contact_map # created by process_hic
            graph.contact_map_diag = self.contact_map_diag
            graph.contact_map_corr = self.contact_map_corr
            graph.contact_map_bonded = self.contact_map_bonded


            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            del graph.weighted_degree # no longer needed

            if self.output_mode != 'contact_map':
                del graph.contact_map
            del graph.contact_map_diag
            del graph.contact_map_corr
            del graph.contact_map_bonded

            if self.output_mode is None or self.output_mode == 'contact_map':
                pass
            elif self.output == 'umatrix':
                U = load_U(raw_folder)
                graph.energy = torch.tensor(U, dtype = torch.float32)
                if self.output_preprocesing == 'log':
                    graph.energy = torch.sign(graph.energy) * torch.log(torch.abs(graph.energy)+1)
            else:
                raise Exception(f'Unrecognized output {self.output}')

            torch.save(graph, self.processed_paths[i])

            # record degree
            if self.verbose:
                deg = np.array(torch_geometric.utils.degree(graph.edge_index[0],
                                                            graph.num_nodes))
                self.degree_list.append(deg)

    def process_hic(self, raw_folder):
        '''
        Helper function to load the appropriate contact map and apply any
        necessary preprocessing.
        '''
        y = np.load(osp.join(raw_folder, 'hic.npy')).astype(np.float64)

        # contact_map_bonded is deprecated - used for some transforms
        self.contact_map_bonded = None

        if self.mean_filt is not None:
            y = uniform_filter(y, self.mean_filt)

        if self.rescale is not None:
            y = rescale_matrix(y, self.rescale)

        if self.hic_normalization == 'max':
            y /= np.max(y)
        elif self.hic_normalization == 'mean':
            y /= np.mean(np.diagonal(y))
        elif self.hic_normalization == 'mean_fill':
            y /= np.mean(np.diagonal(y))
            np.fill_diagonal(y, 1)

        if self.kr:
            y = knightRuiz(y)


        self.contact_map_diag = None
        if self.diag:
            # use y_copy from before preprocessing was applied
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
            y_diag = DiagonalPreprocessing.process(y, meanDist, verbose = False)
            self.contact_map_diag = torch.tensor(y_diag, dtype = torch.float32)

        self.contact_map_corr = None
        if self.corr:
            y_corr = np.corrcoef(y_diag)
            self.contact_map_corr = torch.tensor(y_corr, dtype = torch.float32)

        if self.y_preprocessing == 'log':
            y = np.log(y+1)
        elif self.y_preprocessing == 'log_inf':
            with np.errstate(divide = 'ignore'):
                # surpress warning, infs handled manually
                y = np.log(y)
            y[np.isinf(y)] = np.nan

        # sparsify contact map for computational efficiency
        if self.sparsify_threshold is not None:
            y[np.abs(y) < self.sparsify_threshold] = np.nan
        if self.sparsify_threshold_upper is not None:
            y[np.abs(y) > self.sparsify_threshold_upper] = np.nan

        self.contact_map = torch.tensor(y, dtype = torch.float32)

    def get(self, index):
         data = torch.load(self.processed_paths[index])
         return data

    def len(self):
        return len(self.raw_file_names)

    @property
    def weighted_degree(self):
        return torch.sum(self.contact_map, axis = 1)

    def generate_edge_index(self):
        adj = torch.clone(self.contact_map) # temporary variable
        if self.keep_zero_edges:
            adj[adj == 0] = 1 # first set zero's to some nonzero value
        adj = torch.nan_to_num(adj) # replace nans with zero
        edge_index = adj.nonzero().t() # use all non-zero entries as edges
        self.num_edges_list.append(edge_index.shape[1])

        return edge_index
