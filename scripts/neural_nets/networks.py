'''
Graph neural network implementations.
'''

import math
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from .base_networks import (MLP, AverageTo2d, Bilinear, ConvBlock,
                            FillDiagonalsFromArray, LinearBlock, Symmetrize2D,
                            act2module, torch_eig, torch_mean_dist,
                            torch_triu_to_full)
from .pyg_fns import WeightedGATv2Conv, WeightedSignedConv
from .sign_net.sign_net import SignNet
from .sign_net.transform import to_dense_list_EVD


## model functions ##
def get_model(opt, verbose = True):
    assert opt.model_type.startswith('ContactGNN'), f'Invalid model type: {opt.model_type}'
    GNNClass = ContactGNN

    GNN_model = GNNClass(opt.input_m, opt.node_feature_size,
                opt.hidden_sizes_list,
                opt.encoder_hidden_sizes_list, opt.edge_encoder_hidden_sizes_list,
                opt.update_hidden_sizes_list,
                opt.act, opt.inner_act, opt.out_act,
                opt.message_passing, opt.use_edge_weights or opt.use_edge_attr, opt.edge_dim,
                opt.head_architecture, opt.head_architecture_2, opt.head_hidden_sizes_list,
                opt.head_act, opt.use_bias, opt.rescale, opt.gated, opt.dropout,
                opt.input_L_to_D, opt.input_L_to_D_mode,
                opt.training_norm, opt.num_heads, opt.concat_heads, opt.output_clip,
                opt.log_file, opt.verbose or verbose,
                opt.use_sign_net, opt.use_sign_plus, opt.k)

    if opt.use_sign_net:
        model = SignNetGNN(opt.node_feature_size, opt.edge_dim, opt.update_hidden_sizes_list[-1],
                        8, 8, opt.k, False, GNN_model,
                        opt.log_file, opt.verbose or verbose)
    elif opt.use_sign_plus:
        model = SignPlus(GNN_model, opt.k)
    else:
        model = GNN_model

    return model


class ContactGNN(nn.Module):
    '''
    Graph neural network that maps contact map data to node embeddings of arbitrary length.
    '''
    def __init__(self, m, input_size, MP_hidden_sizes_list,
                node_encoder_hidden_sizes_list, edge_encoder_hidden_sizes_list,
                update_hidden_sizes_list,
                act, inner_act, out_act,
                message_passing, use_edge_attr, edge_dim,
                head_architecture_L, head_architecture_D, head_hidden_sizes_list,
                head_act, use_bias, rescale, gated, dropout,
                input_L_to_D, input_L_to_D_mode,
                training_norm, num_heads, concat_heads, output_clip,
                ofile = sys.stdout, verbose = True,
                sign_net = False, sign_plus = False, k = None):
        '''
        Inputs:
            m: number of nodes
            input_size: size of input node feature vector
            MP_hidden_sizes_list: list of node feature vector hidden sizes during message passing
            node_encoder_hidden_sizes_list: list of hidden sizes for MLP encoder
            edge_encoder_hidden_sizes_list: list of hidden sizes for MLP encoder
            update_hidden_sizes_list: list of hidden sizes for MLP for update during message passing
            inner_hidden_sizes_list: list of hidden sizes for MLP for final update during message passing
            act (str): activation function
            inner_act (str): inner activation function
            out_act (str): output activation
            message_passing (str): type of message passing algorithm to use
                                    {idendity, gcn, signedconv, z, gat, weighted_gat}
            use_edge_attr: True to use edge attributes/weights
            edge_dim: 0 for edge weights, 1+ for edge_attr
            head_architecture_L: type of head architecture {None, fc, inner, bilinear, AverageTo2d.mode_options}
            head_architecture_D: type of head architecture {None, fc, inner, bilinear, AverageTo2d.mode_options}
            head_hidden_sizes_list: hidden sizes of head architecture
            head_act: activation for head_architecture (and head_architecture_2)
            use_bias: True to use bias term - applies for message passing and head
            rescale: rescale by factor <rescale> (None to skip) uses F.interpolate
            gated: True to use gated residual connection (https://doi.org/10.3389/fmolb.2021.647915)
            dropout: Value for dropout (0.0 for no dropout)
            training_norm: Normalization layer
            sign_net: True if using additional_x from sign_net
            sign_plus: True if using additional_x from sign_plus
            k: size for sign_plus
        '''
        super(ContactGNN, self).__init__()

        self.m = m
        self.input_size = input_size
        self.message_passing = message_passing.lower()
        self.use_edge_attr = use_edge_attr
        self.edge_dim = edge_dim
        self.head_architecture_L = head_architecture_L
        self.head_architecture_D = head_architecture_D
        if self.head_architecture_L is not None:
            self.head_architecture_L = head_architecture_L.lower()
        if self.head_architecture_D is not None:
            self.head_architecture_D = head_architecture_D.lower()
        self.to2D = AverageTo2d(mode = None)

        # set up activations
        self.act = act2module(act)
        self.inner_act = act2module(inner_act)
        self.out_act = act2module(out_act)
        self.update_hidden_sizes_list = update_hidden_sizes_list
        self.MP_hidden_sizes_list = MP_hidden_sizes_list
        self.head_hidden_sizes_list = head_hidden_sizes_list
        if head_hidden_sizes_list is not None and len(head_hidden_sizes_list) > 1:
            self.head_act = act2module(head_act)
            # only want this to show up as a parameter if actually needed
        else:
            self.head_act = None
        self.use_bias = use_bias
        self.rescale = rescale
        self.gated = gated
        self.dropout = dropout
        self.input_L_to_D = input_L_to_D
        self.input_L_to_D_mode = input_L_to_D_mode.lower()
        self.training_norm = training_norm
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.output_clip = output_clip
        self.verbose = verbose

        ### Encoder Architecture ###
        self.node_encoder = None
        if node_encoder_hidden_sizes_list is not None:
            encoder = []
            input_size = self.input_size
            for output_size in node_encoder_hidden_sizes_list:
                module = gnn.Linear(input_size, output_size, bias = use_bias)
                encoder.extend([(module, 'x -> x'), self.act])
                input_size = output_size
            self.node_encoder = gnn.Sequential('x', encoder)

        if sign_net:
            output_size = update_hidden_sizes_list[-1]
            self.linear = nn.Linear(input_size+update_hidden_sizes_list[-1], output_size)
            input_size = output_size
        elif sign_plus:
            output_size = update_hidden_sizes_list[-1]
            self.linear = nn.Linear(input_size+k, output_size)
            input_size = output_size
        else:
            self.linear = None

        self.edge_encoder = None
        if edge_encoder_hidden_sizes_list is not None:
            encoder = []
            input_edge_size = output_size * 2 + edge_dim
            for output_size in edge_encoder_hidden_sizes_list:
                encoder.append(LinearBlock(input_edge_size, output_size, activation = self.act,
                                        bias = use_bias))
                input_edge_size = output_size
            self.edge_encoder = nn.Sequential(*encoder)
            self.edge_dim = output_size


        ### Trunk Architecture ###
        self.process_trunk(input_size)

        ### Head Architecture ###
        self.head_L, self.head_L2 = self.process_L()
        self.head_D, self.head_D2 = self.process_D()

        if verbose:
            print("#### ARCHITECTURE ####", file = ofile)
            print('Node Encoder:\n', self.node_encoder, '\n', file = ofile)
            print('Linear:\n', self.linear, '\n', file = ofile)
            print('Edge Encoder:\n', self.edge_encoder, '\n', file = ofile)
            print('Model:\n', self.model, '\n', file = ofile)
            print('Head L:\n', self.head_L, '\n', self.head_L2, '\n', file = ofile)
            print('Head D:\n', self.head_D, '\n', self.head_D2, '\n', file = ofile)

    def process_trunk(self, input_size):
        model = []
        if self.message_passing == 'identity':
            # debugging option to skip message passing
            self.model = None
        elif self.message_passing in {'gcn', 'transformer', 'gat', 'weighted_gat'}:
            if self.use_edge_attr:
                inputs = 'x, edge_index, edge_attr'
            else:
                inputs = 'x, edge_index'
            fn_header = f'{inputs} -> x'

            for i, output_size in enumerate(self.MP_hidden_sizes_list):
                if self.message_passing == 'gcn':
                    module = gnn.GCNConv(input_size, output_size, bias = self.use_bias)
                elif self.message_passing == 'transformer':
                    module = gnn.TransformerConv(input_size, self.output_size,
                                            heads = self.num_heads,
                                            edge_dim = self.edge_dim)
                elif self.message_passing == 'gat':
                    module = gnn.GATv2Conv(input_size, output_size,
                                            heads = self.num_heads, concat = self.concat_heads,
                                            edge_dim = self.edge_dim,
                                            bias = self.use_bias)
                elif self.message_passing == 'weighted_gat':
                    module = WeightedGATv2Conv(input_size, output_size,
                                            heads = self.num_heads, concat = self.concat_heads,
                                            dropout = self.dropout,
                                            edge_dim = self.edge_dim, edge_dim_MP = True,
                                            bias = self.use_bias)
                model.append((module, fn_header))
                if self.concat_heads:
                    input_size = output_size * self.num_heads
                else:
                    input_size = output_size

                if i == len(self.MP_hidden_sizes_list) - 1:
                    act = self.inner_act
                else:
                    act = self.act
                if self.update_hidden_sizes_list is not None:
                    model.append(MLP(input_size, self.update_hidden_sizes_list, self.use_bias,
                                    self.act, act, dropout = self.dropout,
                                    dropout_last_layer = True, gated = self.gated))
                    input_size = self.update_hidden_sizes_list[-1]
                else:
                    model.append(act)

            if self.training_norm == 'instance':
                model.append(gnn.InstanceNorm(input_size))
            elif self.training_norm is not None:
                raise Exception(f'Invalid training_norm: {training_norm}')

            self.model = gnn.Sequential('x, edge_index, edge_attr', model)
        elif self.message_passing == 'signedconv':
            if self.use_edge_attr:
                inputs = 'x, pos_edge_index, neg_edge_index, pos_edge_attr, neg_edge_attr'
            else:
                inputs = 'x, pos_edge_index, neg_edge_index'
            fn_header = f'{inputs} -> x'
            assert self.head_architecture is not None
            first_layer = True

            for output_size in self.MP_hidden_sizes_list:
                model.append((WeightedSignedConv(input_size, output_size,
                                first_aggr = first_layer, bias = self.use_bias,
                                edge_dim = self.edge_dim),
                                fn_header))
                input_size = output_size
                input_size *= 2
                # SignedConv convention that output_size is the size of the
                # negative representation and positive representation respectively,
                # so the total length is 2 * output_size

                if self.update_hidden_sizes_list is not None:
                    for update_output_size in self.update_hidden_sizes_list:
                        model.extend([gnn.Linear(input_size, update_output_size, bias = self.use_bias), self.act])
                        input_size = update_output_size
                else:
                    model.append(self.act)
                input_size //= 2
                first_layer = False

            # replace final act with inner_act
            model.pop()
            model.append(self.inner_act)

            input_size *= 2
            if training_norm == 'instance':
                model.append(gnn.InstanceNorm(input_size))
            elif training_norm is not None:
                raise Exception(f'Invalid training_norm: {training_norm}')

            self.model = gnn.Sequential(inputs, model)
        elif self.message_passing == 'z':
            # uses prior model to predict particle types
            # designed for debugging
            self.model = None
        else:
            raise Exception(f"Unkown message_passing {message_passing}")

        self.latent_size = input_size
        # save input_size to latent_size
        # this is the output_size of latent space
        # and the input size for head_architecture

    def process_L(self):
        head_architecture = self.head_architecture_L
        if head_architecture is None:
            return None

        split = head_architecture.split('-')
        head_list_a = []
        head_list_b = []
        input_size = self.latent_size
        if 'dconv' in split:
            dilations = [1, 2, 4, 8, 16]
            for dilation in dilations:
                if dilation == dilations[-1]:
                    act = self.out_act
                else:
                    act = self.head_act
                head_list_a.append(ConvBlock(input_size, input_size, 3, padding = dilation,
                                        activation = act,
                                        dilation = dilation, conv1d = True))
        elif 'conv' in split:
            head_list_a.append(ConvBlock(input_size, input_size, 3, padding = 1,
                                    activation = self.head_act, conv1d = True))

        if 'triu' in split:
            size = int(self.latent_size*(self.latent_size+1)/2)
            init = torch.zeros(size)
            self.sym = torch_triu_to_full
        else:
            init = torch.zeros((self.latent_size, self.latent_size))
            torch.nn.init.xavier_normal_(init)
            self.sym = Symmetrize2D()
        if 'chi' not in split:
            self.W = nn.Parameter(init)

        if 'bilinear' in split:
            head_b = 'Bilinear'
        elif 'inner' in split:
            head_b = 'Inner'
        else:
            head_b = None

        head_a = None
        if head_list_a:
            head_a = nn.Sequential(*head_list_a)
        if head_list_b:
            head_b = nn.Sequential(*head_list_b)

        return head_a, head_b

    def process_D(self):
        head_architecture = self.head_architecture_D
        if head_architecture is None:
            return None, None

        split = head_architecture.split('_')
        if head_architecture is None:
            return None

        if len(split) > 1:
            output_size = int(split[1])
            head_hidden_sizes_list = self.head_hidden_sizes_list + [output_size]
        else:
            head_hidden_sizes_list = self.head_hidden_sizes_list

        head_list_a = []
        head_list_b = []
        input_size = self.latent_size

        if 'dconv' in head_architecture:
            output_size = int(input_size / 2)
            for dilation in [1, 2, 4, 8, 16]:
                head_list_a.append(ConvBlock(input_size, output_size, 3, padding = dilation,
                                        activation = self.head_act,
                                        dilation = dilation, conv1d = True))
                input_size = output_size

        if 'stride' in head_architecture:
            # single strided conv layer to reduce
            head_list_a.append(ConvBlock(input_size, input_size, 3, stride = 2, padding = 1,
                                    activation = self.head_act, conv1d = True))
            # new_m = math.floor((self.m-3+2)/2)+1
            input_size = int(input_size * self.m/2)
        elif 'pool' in head_architecture:
            head_list_a.append(ConvBlock(input_size, input_size, 3,
                                    activation = self.head_act, pool = 'maxpool',
                                    conv1d = True))
            input_size = int(input_size * self.m / 2)
        else:
            input_size = input_size * self.m

        if self.input_L_to_D:
            if 'meandist' in self.input_L_to_D_mode:
                input_size += int(self.m * self.rescale)
            if 'eigval' in self.input_L_to_D_mode:
                input_size += 10
            if 'subtract' in self.input_L_to_D_mode:
                self.fill = FillDiagonalsFromArray()

        if 'fc' in head_architecture:
            head_list_b.append(MLP(input_size, head_hidden_sizes_list, self.use_bias,
                                self.head_act, self.out_act, dropout = self.dropout))
        if 'fill' in head_architecture:
            head_list_b.append(FillDiagonalsFromArray())

        head_a = None; head_b = None
        if head_list_a:
            head_a = nn.Sequential(*head_list_a)
        if head_list_b:
            head_b = nn.Sequential(*head_list_b)

        return head_a, head_b

    def forward(self, graph, additional_x=None, verbose=False):
        self.batch_size = int(graph.batch.max()) + 1
        latent = self.latent(graph, additional_x)

        if self.head_architecture_L is None and self.head_architecture_D is None:
            return latent

        L_out = self.plaid_component(latent)

        additional = None
        if self.input_L_to_D:
            # calculate mean along each diagonal of L_out
            if 'meandist' in self.input_L_to_D_mode and 'eigval' in self.input_L_to_D_mode:
                meanDist = torch_mean_dist(L_out)
                eig_vals = torch_eig(L_out, 10)
                additional = torch.cat((meanDist, eig_vals), 1)
            elif 'meandist' in self.input_L_to_D_mode:
                additional = torch_mean_dist(L_out)
            elif 'eigval' in self.input_L_to_D_mode:
                additional = torch_eig(L_out, 10)
            elif 'subtract' in self.input_L_to_D_mode:
                meanDist = torch_mean_dist(L_out)
                L_meanDist = self.fill(meanDist)
            else:
                raise Exception(f'{self.input_L_to_D_mode} not recognized')
        D_out = self.diagonal_component(latent, additional)
        if self.input_L_to_D and 'subtract' in self.input_L_to_D_mode:
            D_out -= L_meanDist

        if L_out is None:
            return D_out
        if D_out is None:
            return L_out

        del latent
        try:
            S_out = L_out + D_out
        except RuntimeError:
            print(L_out.shape, D_out.shape)
            raise

        if self.output_clip is not None:
            S_out[S_out > self.output_clip] = self.output_clip
            S_out[S_out < -self.output_clip] = -self.output_clip

        del L_out; del D_out
        torch.cuda.empty_cache()
        return S_out

    def latent(self, graph, additional_x):
        if self.node_encoder is not None:
            x = self.node_encoder(graph.x)
        else:
            x = graph.x

        if additional_x is not None:
            if x is None:
                x = self.linear(additional_x)
            else:
                x = self.linear(torch.cat([x, additional_x], dim=-1))

        if self.edge_encoder is not None:
            row, col = graph.edge_index
            concat = torch.cat((x[row], x[col], graph.edge_attr), dim = -1)
            edge_attr = self.edge_encoder(concat)
        else:
            edge_attr = graph.edge_attr


        if self.message_passing == 'identity':
            latent = x
        elif self.message_passing in {'gcn', 'transformer', 'gat', 'weighted_gat'}:
            # print('x', x)
            # print('graph.edge_index', graph.edge_index)
            # print('edge_attr', edge_attr)
            latent = self.model(x, graph.edge_index, edge_attr)
            # print('latent', latent)
        elif self.message_passing == 'signedconv':
            if self.use_edge_attr:
                latent = self.model(x, graph.pos_edge_index, graph.neg_edge_index,
                                    graph.pos_edge_attr, graph.neg_edge_attr)
            else:
                latent = self.model(x, graph.pos_edge_index, graph.neg_edge_index)

        return latent

    def diagonal_component(self, latent, additional=None):
        _, output_size = latent.shape
        if self.head_architecture_D is None:
            return none
        elif 'fc' in self.head_architecture_D:
            if self.head_D is not None:
                latent = latent.reshape(self.batch_size, self.m, output_size)
                latent = latent.permute(0, 2, 1) # permute to combine over m index
                D_out = self.head_D(latent)
                latent = latent.permute(0, 2, 1) # permute back
            else:
                D_out = torch.clone(latent)
            D_out = D_out.reshape(self.batch_size, -1)
            if additional is not None:
                # additional is shape Nxm
                D_out = self.head_D2(torch.cat((D_out, additional), 1))
            else:
                D_out = self.head_D2(D_out)
        elif self.head_architecture_D in self.to2D.mode_options:
            latent = latent.reshape(self.batch_size, self.m, output_size)
            latent = latent.permute(0, 2, 1) # permute to combine over m index
            D_out = self.to2D(latent)
            D_out = D_out.permute(0, 2, 3, 1) # permute back
            latent = latent.permute(0, 2, 1) # permute back
            D_out = self.head(D_out)
            if len(D_out.shape) > 3:
                D_out = torch.squeeze(D_out, 3)

            if self.rescale is not None:
                D_out = torch.unsqueeze(D_out, 1)
                m_new = int(self.m * self.rescale)
                D_out = F.interpolate(D_out, size = (m_new, m_new))
                D_out = torch.squeeze(D_out, 1)

        return D_out

    def plaid_component(self, latent):
        _, output_size = latent.shape
        if self.head_architecture_L is None:
            return None

        latent = latent.reshape(self.batch_size, self.m, output_size)
        if self.head_L is not None:
            latent = latent.permute(0, 2, 1) # permute to combine over m index
            L_out = self.head_L(latent)
            L_out = L_out.permute(0, 2, 1) # permute back
            latent = latent.permute(0, 2, 1) # permute back
        else:
            L_out = torch.clone(latent)

        if self.head_L2 == 'Inner':
            L_out = torch.einsum('nik, njk->nij', L_out, L_out)
        elif self.head_L2 == 'Bilinear':
            # if 'chi' in self.head_architecture_L:
            #     latent = latent.reshape(-1, self.m * output_size)
            #     self.W = self.head_L3(latent)
            #     L_out = latent.reshape(-1, self.m, output_size)

            if 'asym' in self.head_architecture_L:
                L_out = torch.einsum('nik,njk->nij', L_out @ self.W, L_out)
            else:
                W = self.sym(self.W)
                if len(W.shape) == 2:
                    left = torch.einsum('nij,jk->nik', L_out, W)
                elif len(W.shape) == 3:
                    left = torch.einsum('nij,njk->nik', L_out, W)
                L_out = torch.einsum('nik,njk->nij', left, L_out)


        if self.rescale is not None:
            L_out = torch.unsqueeze(L_out, 1)
            m_new = int(self.m * self.rescale)
            L_out = F.interpolate(L_out, size = (m_new, m_new))
            L_out = torch.squeeze(L_out, 1)

        return L_out

class SignNetGNN(nn.Module):
    # Modified from https://github.com/cptq/SignNet-BasisNet/blob/main/Alchemy/sign_net/sign_net.py
    def __init__(self, node_feat, edge_feat, n_hid, nl_signnet,
                    nl_rho, k, ignore_eigval, gnn_model, ofile, verbose):
        super().__init__()
        self.sign_net = SignNet(n_hid, nl_signnet, nl_rho, ignore_eigval, k)
        self.gnn = gnn_model

        if verbose:
            print("#### ARCHITECTURE ####", file = ofile)
            print('Sign Net:\n', self.sign_net, '\n', file = ofile)


    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data, verbose=False):
        pos = self.sign_net(data)
        return self.gnn(data, pos)

    def latent(self, data, *args):
        pos = self.sign_net(data)
        return self.gnn.latent(data, pos)

    def plaid_component(self, data):
        return None

    def diagonal_component(self, data):
        return None

class SignPlus(nn.Module):
    # Modified from https://github.com/cptq/SignNet-BasisNet/blob/main/LearningFilters/signbasisnet.py
    # negate v, do not negate x
    def __init__(self, model, k):
        super(SignPlus, self).__init__()
        self.model = model
        self.k = k

    def get_eig(self, data, verbose=False):
        eigS_dense, eigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch, self.k)
        if verbose:
            print('eigV_dense', eigV_dense, torch.min(eigV_dense), torch.max(eigV_dense))
        return eigV_dense

    def forward(self, data, verbose=False):
        x = self.get_eig(data, verbose)
        if verbose:
            print('x', data.x)
            print('edge_attr', data.edge_attr, torch.min(data.edge_attr), torch.max(data.edge_attr))
            print('edge_index', data.edge_index, torch.min(data.edge_index), torch.max(data.edge_index))
        return self.model(data, x) + self.model(data, -x)

    def latent(self, data, *args):
        x = self.get_eig(data)
        self.model.batch_size = int(data.batch.max()) + 1
        result = torch.stack((self.model.latent(data, x), self.model.latent(data, -x)))
        return result

    def plaid_component(self, latent):
        return self.model.plaid_component(latent)

    def diagonal_component(self, latent):
        return self.model.diagonal_component(latent)
