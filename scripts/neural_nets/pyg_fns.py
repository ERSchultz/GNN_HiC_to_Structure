import json
import os
import os.path as osp
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_D
from sklearn.decomposition import PCA
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.transforms import BaseTransform, Constant
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import (add_self_loops, degree, remove_self_loops,
                                   softmax)
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_std
from torch_sparse import SparseTensor, matmul, set_diag


# node transforms
class WeightedLocalDegreeProfile(BaseTransform):
    '''
    Weighted version of Local Degree Profile (LDP) from https://arxiv.org/abs/1811.03508
    Appends WLDP features to feature vector.

    Reference code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/
        torch_geometric/transforms/local_degree_profile.html#LocalDegreeProfile
    '''
    def __call__(self, data):
        row, col = data.edge_index
        N = data.num_nodes

        # weighted_degree must exist
        deg = data.weighted_degree
        deg_col = deg[col]

        min_deg, _ = scatter_min(deg_col, row, dim_size=N)
        min_deg[min_deg > 10000] = 0
        max_deg, _ = scatter_max(deg_col, row, dim_size=N)
        max_deg[max_deg < -10000] = 0
        mean_deg = scatter_mean(deg_col, row, dim_size=N)
        std_deg = scatter_std(deg_col, row, dim_size=N)

        x = torch.stack([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, x], dim=-1)
        else:
            data.x = x

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')

class Degree(BaseTransform):
    '''
    Appends degree features to node feature vector.

    Reference code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/
        torch_geometric/transforms/target_indegree.html#TargetIndegree
    '''
    def __init__(self, norm = True, max_val = None, weighted = False,
                split_edges = False, split_val = 0, diag = False):
        '''
        Inputs:
            norm: True to normalize degree by dividing by max_val
            max_val: value for norm, if None uses maximum degree
            weighted: True for weighted degree, False for count
            split_edges: True to divide edges based on split_val
            split_val: split value for split_edges
            diag: TODO
        '''
        self.norm = norm # bool
        self.max = max_val # float
        self.weighted = weighted # bool
        self.split_edges = split_edges # bool
        self.split_val = split_val # float
        self.diag = diag
        # values less than split_val are one type of edge
        # values greater than split_val are second type of edge

    def __call__(self, data):
        if self.weighted:
            if self.split_edges:
                if self.diag:
                    ypos = torch.clone(data.contact_map_diag)
                    yneg = torch.clone(data.contact_map_diag)
                else:
                    ypos = torch.clone(data.contact_map)
                    yneg = torch.clone(data.contact_map)

                ypos[ypos < self.split_val] = 0
                pos_deg = torch.sum(ypos, axis = 1)
                del ypos
                yneg = torch.clone(data.contact_map)
                yneg[yneg > self.split_val] = 0
                neg_deg = torch.sum(yneg, axis = 1)
                del yneg
            else:
                deg = data.weighted_degree
        else:
            if self.split_edges:
                if self.diag:
                    pos_deg = degree((data.contact_map_diag > self.split_val).nonzero().t()[0], data.num_nodes)
                    neg_deg = degree((data.contact_map_diag < self.split_val).nonzero().t()[0], data.num_nodes)
                else:
                    pos_deg = degree((data.contact_map > self.split_val).nonzero().t()[0], data.num_nodes)
                    neg_deg = degree((data.contact_map < self.split_val).nonzero().t()[0], data.num_nodes)
            else:
                deg = degree(data.edge_index[0], data.num_nodes)
                # deg = degree((data.contact_map).nonzero().t()[0], data.num_nodes)


        if self.norm:
            if self.split_edges:
                # print(pos_deg.max())
                # print(neg_deg.max())
                pos_deg /= (pos_deg.max() if self.max is None else self.max)
                neg_deg /= (neg_deg.max() if self.max is None else self.max)
            else:
                deg /= (deg.max() if self.max is None else self.max)

        if self.split_edges:
            deg = torch.stack([pos_deg, neg_deg], dim=1)
        else:
            deg = torch.stack([deg], dim=1)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, deg], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        repr = (f'{self.__class__.__name__}'
                f'(norm={self.norm}')

        if self.max is not None:
            repr += f', max={self.max}'

        if self.weighted:
            repr += f'weighted={self.weighted}'

        repr += f', split_edges={self.split_edges}'
        if self.split_edges:
            repr += f', split_val={self.split_val})'
        else:
            repr += ')'

        return repr

class AdjPCATransform(BaseTransform):
    '''Appends rank k PCA transformation of adjacency matrix to feature vector.'''
    def __init__(self, k = 5, diag = False):
        self.k = k
        self.diag = True

    def __call__(self, data,):
        if self.diag:
            input = torch.clone(data.contact_map_diag)
        else:
            input = torch.clone(data.contact_map)
        pca = PCA(n_components = self.k)
        y_trans = pca.fit_transform(input)

        y_trans = torch.tensor(y_trans, dtype = torch.float32)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, y_trans], dim=-1)
        else:
            data.x = y_trans

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k={self.k}, diag={self.diag})')

class AdjPCs(BaseTransform):
    '''Appends values from top k PCs of adjacency matrix to feature vector.'''
    def __init__(self, k = 5, normalize = False, sign_net = False):
        self.k = k
        self.normalize = normalize
        self.sign_net = sign_net

    def __call__(self, data):
        input = data.contact_map_diag
        # input = torch.nan_to_num(data.contact_map_diag, nan = 1.0)
        pca = PCA(n_components = self.k)
        pca.fit(input)

        m = len(input)
        topk_pcs = np.zeros((m, self.k))
        for j in range(self.k):
            pc = pca.components_[j]

            if self.normalize:
                min = np.min(pc)
                max = np.max(pc)
                if max > abs(min):
                    val = max
                else:
                    val = abs(min)

                # multiply by scale such that val x scale = 1
                scale = 1/val
                pc *= scale

            topk_pcs[:,j] = pc

        topk_pcs = torch.tensor(topk_pcs, dtype = torch.float32)

        if self.sign_net:
            data.eigen_vectors = topk_pcs.reshape(-1)
            data.eigen_values = torch.tensor(pca.singular_values_[:self.k], dtype = torch.float32)
        else:
            if data.x is not None:
                data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
                data.x = torch.cat([data.x, topk_pcs], dim=-1)
            else:
                data.x = topk_pcs

        return data

    def __repr__(self) -> str:
        repr = f'{self.__class__.__name__}(k={self.k}, normalize={self.normalize}'

        if self.sign_net:
            repr += f', sign_net={self.sign_net})'
        else:
            repr += ')'

        return repr

class AdjTransform(BaseTransform):
    '''Appends rows of adjacency matrix to feature vector.'''
    def __call__(self, data):
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, torch.clone(data.contact_map)], dim=-1)
        else:
            data.x = torch.clone(data.contact_map)

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')

class OneHotGeneticPosition(BaseTransform):
    '''Appends one hot encoded genetic position to feature vector.'''
    def __call__(self, data):
        pos = torch.arange(0, data.num_nodes, dtype=torch.long)
        pos = F.one_hot(pos, num_classes=data.num_nodes).to(torch.float)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, pos], dim=-1)
        else:
            data.x = pos

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')

class GeneticPosition(BaseTransform):
    '''Appends genetic position to feature vector.'''
    def __init__(self, center = False, norm = False):
        self.center = center # bool
        self.norm = norm # bool

    def __call__(self, data):
        pos = torch.arange(0, data.num_nodes, dtype=torch.float32).reshape(data.num_nodes, 1)

        if self.center:
            pos -= pos.mean(dim=-2, keepdim=True)
        if self.norm:
            pos /= pos.max()

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, pos], dim=-1)
        else:
            data.x = pos

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(center={self.center}, norm={self.norm})')

class GridSize(BaseTransform):
    '''Appends grid size as node feature.'''
    def __init__(self, grid_path=None):
        if grid_path is not None:
            self.grid_file = osp.join(grid_path, 'grid.txt')
        else:
            self.grid_file = None

    def __call__(self, data):
        if self.grid_file is not None:
            grid_file = osp.join(data.path, self.grid_file)
        else:
            grid_file = None

        config_file = osp.join(data.path, 'config.json')
        if osp.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
            grid_size = config['grid_size']
        elif grid_file is not None and osp.exists(grid_file):
            grid_size = np.atleast_1d(np.loadtxt(grid_file))[-1]
        else:
            raise Exception(f"Grid size files not found for {data.path}: {config_file}, {grid_file}")

        pos = torch.full((data.num_nodes, 1), grid_size, dtype=torch.float32)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, pos], dim=-1)
        else:
            data.x = pos

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}')


# edge transforms
class GeneticDistance(BaseTransform):
    '''
    Appends genetic distance to features to edge attr vector.

    Based off of https://pytorch-geometric.readthedocs.io/en/latest/
    _modules/torch_geometric/transforms/distance.html
    Note that GeneticDistance doesn't assume data.pos exists while Distance does
    '''
    def __init__(self, norm = False, max_val = None, cat = True,
                split_edges = False, convert_to_attr = False,
                log = False, log10 = False, positional_encoding = False,
                positional_encoding_d = 0):
        '''
        Inputs:
            norm: True to normalize distances by dividing by max_val
            max_val: value for norm, if None uses maximum distance
            cat: True to concatenate attr, False to overwrite
            split_edges: True if graph has 'positive' and 'negative' edges
            convert_to_attr: True for edge attr, False for edge weight
            log: ln transform
            log10: log10 transform
        '''
        self.norm = norm
        self.max = max_val
        self.cat = cat
        self.split_edges = split_edges # bool
        self.convert_to_attr = convert_to_attr # bool, converts to 2d array
        self.log = log # apply ln transform
        self.log10 = log10 # apply log10 transform
        self.positional_encoding = positional_encoding
        self.positional_encoding_d = positional_encoding_d
        # True to use positional encoding from Attention is All You Need
        assert not self.log or not self.log10, "only one can be True"
        if self.positional_encoding:
            assert not self.split_edges and not self.norm and not self.log and not self.log10
            assert self.convert_to_attr
            assert self.positional_encoding_d > 0 and self.positional_encoding_d % 2 == 0


    def __call__(self, data):
        pos = torch.arange(0, data.num_nodes, dtype=torch.float32).reshape(data.num_nodes, 1)

        if self.split_edges:
            # positive
            if 'pos_edge_attr' in data._mapping:
                pseudo = data.pos_edge_attr
            else:
                pseudo = None

            row, col = data.pos_edge_index
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1)
            if self.norm and dist.numel() > 0:
                dist = dist / (data.num_nodes if self.max is None else self.max)
            if self.log:
                dist = np.log(dist)
            elif self.log10:
                dist = np.log10(dist)
            if self.convert_to_attr:
                dist = dist.reshape(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.pos_edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
            else:
                data.pos_edge_attr = dist

            # negative
            if 'neg_edge_attr' in data._mapping:
                pseudo = data.neg_edge_attr
            else:
                pseudo = None
            row, col = data.neg_edge_index
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1)
            if self.norm and dist.numel() > 0:
                dist = dist / (dist.max() if self.max is None else self.max)
            if self.log:
                dist = np.log(dist)
            if self.convert_to_attr:
                dist = dist.reshape(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.neg_edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
            else:
                data.neg_edge_attr = dist

        else:
            (row, col), pseudo = data.edge_index, data.edge_attr
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1)

            if self.positional_encoding:
                d = self.positional_encoding_d
                new_dist = torch.zeros(len(dist), d)
                for i in range(d//2):
                    dist_rescale = dist/10000**(2*i/d)
                    new_dist[:, 2*i] = torch.sin(dist_rescale)
                    new_dist[:, 2*i+1] = torch.cos(dist_rescale)
                dist = new_dist
            else:
                if self.convert_to_attr:
                    dist = dist.reshape(-1, 1)
                if self.norm and dist.numel() > 0:
                    dist = dist / (data.num_nodes if self.max is None else self.max)
                if self.log:
                    dist = torch.log(dist)
                elif self.log10:
                    dist = torch.log10(dist)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = dist

        return data

    def __repr__(self) -> str:
        repr = f'{self.__class__.__name__}(norm={self.norm}'

        if self.max is not None:
            repr += f', max={self.max}'

        if self.log:
            repr += f', log={self.log}'


        if self.positional_encoding:
            repr += f', positional_encoding with d={self.positional_encoding_d})'
        else:
            repr += ')'

        return repr

class ContactDistance(BaseTransform):
    '''
    Appends contact map entries to edge attr vector.
    '''
    def __init__(self, norm = False, max_val = None, cat = True, split_edges = False,
                    convert_to_attr = False, bonded = False, diag_normalize=False,
                    corr = False, rank = None):
        '''
        Inputs:
            norm (bool): True for instant normalization
            max_val (None or int): Normalize by max_val if given, otherwise by true max
            cat (bool): True to concatenate attr, False to overwrite
            split_edges (bool): True if graph has 'positive' and 'negative' edges
            convert_to_attr (bool): True for edge attr, False for edge weight
            bonded (bool): True to use bonded contact map instead of experiment
        '''
        self.norm = norm
        self.max = max_val
        self.cat = cat
        self.split_edges = split_edges
        self.convert_to_attr = convert_to_attr # converts to 2d array,
                                               # else would be an edge weight
        self.bonded = bonded
        self.diag_norm = diag_normalize
        self.corr = corr
        self.rank = rank
        assert not (self.bonded and self.diag_norm), 'mutually exclusive options'
        assert not (self.bonded and self.corr), 'mutually exclusive options'
        assert not (self.corr and self.diag_norm), 'mutually exclusive options'

    def __call__(self, data):
        if self.split_edges:
            assert not self.bonded and not self.diag_norm, 'Not Implemented yet'
            # positive
            if 'pos_edge_attr' in data._mapping:
                pseudo = data.pos_edge_attr
            else:
                pseudo = None

            row, col = data.pos_edge_index
            pos_edge_attr = data.contact_map[row, col]
            if self.convert_to_attr:
                pos_edge_attr = pos_edge_attr.reshape(-1, 1)
            if self.norm:
                pos_edge_attr /= (pos_edge_attr.max() if self.max is None else self.max)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.pos_edge_attr = torch.cat([pseudo, pos_edge_attr.type_as(pseudo)], dim=-1)
            else:
                data.pos_edge_attr = pos_edge_attr

            # negative
            if 'neg_edge_attr' in data._mapping:
                pseudo = data.neg_edge_attr
            else:
                pseudo = None
            row, col = data.neg_edge_index
            neg_edge_attr = data.contact_map[row, col]
            if self.convert_to_attr:
                neg_edge_attr = neg_edge_attr.reshape(-1, 1)
            if self.norm:
                neg_edge_attr /= (neg_edge_attr.max() if self.max is None else self.max)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.neg_edge_attr = torch.cat([pseudo, neg_edge_attr.type_as(pseudo)], dim=-1)
            else:
                data.neg_edge_attr = neg_edge_attr

        else:
            (row, col), pseudo = data.edge_index, data.edge_attr
            if self.bonded:
                assert data.contact_map_bonded is not None, "Can't be None"
                graph = data.contact_map_bonded
            elif self.diag_norm:
                graph = data.contact_map_diag
            elif self.corr:
                graph = data.contact_map_corr
            else:
                graph = data.contact_map

            if self.rank is not None:
                pca = PCA(n_components = self.rank)
                transform = pca.fit_transform(graph)
                graph = pca.inverse_transform(transform)
                graph = torch.tensor(graph, dtype=torch.float32)

            edge_attr = graph[row, col]
            if self.convert_to_attr:
                edge_attr = edge_attr.reshape(-1, 1)

            if self.norm:
                edge_attr /= (edge_attr.max() if self.max is None else self.max)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, edge_attr.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = edge_attr

        return data


    def __repr__(self) -> str:
        repr = f'{self.__class__.__name__}(norm={self.norm}'

        if self.max is not None:
            repr += f', max={self.max}'

        if self.bonded:
            repr += ', bonded=True'

        if self.diag_norm:
            repr += ', diag_norm=True'

        if self.corr:
            repr += ', corr=True'

        if self.rank is not None:
            repr += f', rank={self.rank}'

        if self.split_edges:
            repr += f', split_edges={self.split_edges})'
        else:
            repr += ')'

        return repr

class MeanContactDistance(BaseTransform):
    '''
    Appends mean diagonal of contact map to edge attr vector.
    '''
    def __init__(self, norm=False, max_val=None, cat=True,
                    convert_to_attr=False, bonded=False,
                    bonded_path=None):
        '''
        Inputs:
            norm (bool): True for instant normalization
            max_val (None or int): Normalize by max_val if given, otherwise by true max
            cat (bool): True to concatenate attr, False to overwrite
            split_edges (bool): True if graph has 'positive' and 'negative' edges
            convert_to_attr (bool): True for edge attr, False for edge weight
            bonded (bool): True to use bonded contact map instead of experiment
        '''
        self.norm = norm
        self.max = max_val
        self.cat = cat
        self.convert_to_attr = convert_to_attr # bool, converts to 2d array
        # else would be an edge weight
        self.bonded = bonded

    def __call__(self, data):
        if self.bonded:
            assert data.contact_map_bonded is not None, "Can't be None"
            mean_per_diagonal = DiagonalPreprocessing.genomic_distance_statistics(data.contact_map_bonded, mode = 'freq')
            mean_per_diagonal[np.isnan(mean_per_diagonal)] = np.nanmin(mean_per_diagonal)
        else:
            mean_per_diagonal = DiagonalPreprocessing.genomic_distance_statistics(data.contact_map, mode = 'freq')
        contact_map = calculate_D(mean_per_diagonal)
        contact_map = torch.tensor(contact_map, dtype = torch.float32)

        (row, col), pseudo = data.edge_index, data.edge_attr
        edge_attr = contact_map[row, col]
        if self.convert_to_attr:
            edge_attr = edge_attr.reshape(-1, 1)

        if self.norm:
            edge_attr /= (edge_attr.max() if self.max is None else self.max)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, edge_attr.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = edge_attr

        return data


    def __repr__(self) -> str:
        repr = f'{self.__class__.__name__}(norm={self.norm}'

        if self.max is not None:
            repr += f', max={self.max}'

        if self.bonded:
            repr += f', bonded={self.bonded})'
        else:
            repr += ')'

        return repr

class DiagonalParameterDistance(BaseTransform):
    '''
    Appends diagonal parameter to features to edge attr vector.
    '''
    def __init__(self, cat = True, split_edges = False, convert_to_attr = False,
                id = None):
        '''
        Inputs:
            cat: True to concatenate attr, False to overwrite
            split_edges: consider 'positive' and 'negative' edges separately
            convert_to_attr: True for edge attr, False for edge weight
            id: ID of trained MLP model
        '''
        self.cat = cat
        self.split_edges = split_edges # bool
        self.convert_to_attr = convert_to_attr # bool, converts to 2d array
        self.mlp_id = id

    def __call__(self, data):
        # get D
        if self.mlp_id is None:
            D = calculate_D(data.diag_chi_continuous)
        else:
            assert data.mlp_model_id == self.mlp_id
            D = calculate_D(data.diag_chi_continuous_mlp)
        D = torch.tensor(D, dtype = torch.float32)

        if self.split_edges:
            # positive
            if 'pos_edge_attr' in data._mapping:
                pseudo = data.pos_edge_attr
            else:
                pseudo = None

            row, col = data.pos_edge_index
            pos_edge_attr = D[row, col]
            if self.convert_to_attr:
                pos_edge_attr = pos_edge_attr.reshape(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.pos_edge_attr = torch.cat([pseudo, pos_edge_attr.type_as(pseudo)], dim=-1)
            else:
                data.pos_edge_attr = pos_edge_attr

            # negative
            if 'neg_edge_attr' in data._mapping:
                pseudo = data.neg_edge_attr
            else:
                pseudo = None
            row, col = data.neg_edge_index
            neg_edge_attr = D[row, col]
            if self.convert_to_attr:
                neg_edge_attr = neg_edge_attr.reshape(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.neg_edge_attr = torch.cat([pseudo, neg_edge_attr.type_as(pseudo)], dim=-1)
            else:
                data.neg_edge_attr = neg_edge_attr

        else:
            (row, col), pseudo = data.edge_index, data.edge_attr
            edge_attr = D[row, col]
            if self.convert_to_attr:
                edge_attr = edge_attr.reshape(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, edge_attr.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = edge_attr

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'(mlp_id={self.mlp_id})')



# message passing
class WeightedSignedConv(MessagePassing):
    '''
    Variant of SignedConv that allows for edge weights or edge features.
    Adapted from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/
            torch_geometric/nn/conv/signed_conv.html#SignedConv

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        first_aggr (bool): Denotes which aggregation formula to use.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    '''
    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool,
                 bias: bool = True, edge_dim: int = 0, **kwargs):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr
        self.edge_dim = edge_dim

        if first_aggr:
            self.lin_pos_l = Linear(in_channels + self.edge_dim, out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(in_channels + self.edge_dim, out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)
        else:
            self.lin_pos_l = Linear(2 * (in_channels + self.edge_dim), out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(2 * (in_channels + self.edge_dim), out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_pos_l.reset_parameters()
        self.lin_pos_r.reset_parameters()
        self.lin_neg_l.reset_parameters()
        self.lin_neg_r.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj, pos_edge_attr: OptTensor = None,
                neg_edge_attr: OptTensor = None):
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_weight: OptTensor)
        if self.first_aggr:
            out_pos = self.propagate(pos_edge_index, x=x, size=None,
                                    edge_attr=pos_edge_attr)
            out_pos = self.lin_pos_l(out_pos)
            out_pos += self.lin_pos_r(x[1])

            out_neg = self.propagate(neg_edge_index, x=x, size=None,
                                    edge_attr=neg_edge_attr)
            out_neg = self.lin_neg_l(out_neg)
            out_neg += self.lin_neg_r(x[1])

            return torch.cat([out_pos, out_neg], dim=-1)

        else:
            F_in = self.in_channels

            out_pos1 = self.propagate(pos_edge_index, size=None,
                                    edge_attr=pos_edge_attr,
                                    x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_pos2 = self.propagate(neg_edge_index, size=None,
                                    edge_attr=neg_edge_attr,
                                    x=(x[0][..., F_in:], x[1][..., F_in:]))
            out_pos = torch.cat([out_pos1, out_pos2], dim=-1)
            out_pos = self.lin_pos_l(out_pos)
            out_pos += self.lin_pos_r(x[1][..., :F_in])

            out_neg1 = self.propagate(pos_edge_index, size=None,
                                    edge_attr=pos_edge_attr,
                                    x=(x[0][..., F_in:], x[1][..., F_in:]))
            out_neg2 = self.propagate(neg_edge_index, size=None,
                                    edge_attr=neg_edge_attr,
                                    x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_neg = torch.cat([out_neg1, out_neg2], dim=-1)
            out_neg = self.lin_neg_l(out_neg)
            out_neg += self.lin_neg_r(x[1][..., F_in:])

            return torch.cat([out_pos, out_neg], dim=-1)


    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        if self.edge_dim == 0:
            # treat edge_attr as a weight if it exists
            result = x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j
        else:
            result = x_j if edge_attr is None else torch.cat((x_j, edge_attr), axis = 1)
        return result


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, first_aggr={self.first_aggr},'
                f'edge_dim={self.edge_dim})')

class WeightedGATv2Conv(MessagePassing):
    """
    Variant of GATv2Conv that allows for edge features during
    message passing (instead of just as attention coefficients).

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        edge_dim_MP (bool, optional): True to use edge features in message passing
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)

        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        edge_dim_MP: bool = False,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.edge_dim_MP = edge_dim_MP
        self.fill_value = fill_value
        self.share_weights = share_weights


        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            # if edge_dim_MP:
            #     self.lin_edge_MP = Linear(edge_dim, heads * out_channels, bias=False,
            #                            weight_initializer='glorot')
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)

            # if self.edge_dim_MP:
            #     assert self.lin_edge_MP is not None

            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if self.edge_dim_MP and edge_attr is not None:
            return (x_j + edge_attr) * alpha.unsqueeze(-1)
        else:
            return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
