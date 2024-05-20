'''
Custom loss functions.
'''

import os.path as osp

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from .base_networks import TORCH_SCC, torch_mean_dist, torch_subtract_diag


def mse_center(input, target):
    input_center = input - torch.mean(input)
    target_center = target - torch.mean(target)
    return F.mse_loss(input_center, target_center)

def mse_log(input, target, *args):
    input_log = torch.sign(input) * torch.log(torch.abs(input) + 1)
    target_log = torch.sign(target) * torch.log(torch.abs(target) + 1)
    return F.mse_loss(input_log, target_log)

class MSE_EXP_NORM():
    def __init__(self):
        pass

    def normalize(arr):
        N, m, _ = arr.shape
        main_diagonals = torch.diagonal(arr, dim1=1, dim2=2)
        means = torch.mean(main_diagonals, dim=1)
        means = torch.broadcast_to(means, (m, m, N))
        means = torch.permute(means, (2, 0, 1))
        return arr / means

    def normalize2(arr):
        N, m, _ = arr.shape
        means = torch.mean(arr, dim=(1,2))
        means = torch.broadcast_to(means, (m, m, N))
        means = torch.permute(means, (2, 0, 1))

        arr = arr / means
        arr[arr > 1] = 1

        return arr

    def clip(arr, val):
        arr[arr > val] = val
        arr[arr < -val] = -val

        return arr

    def __call__(self, input, target):
        input_exp = torch.exp(-input)
        input_exp_norm = MSE_EXP_NORM.normalize2(input_exp)

        # print('input', torch.min(input), torch.max(input))
        # print('input_exp', torch.min(input_exp), torch.max(input_exp))
        # print('input_exp_norm', torch.min(input_exp_norm), torch.max(input_exp_norm))
        # print('---')

        target_exp = torch.exp(-target)
        target_exp_norm = MSE_EXP_NORM.normalize2(target_exp)

        return F.mse_loss(input_exp_norm, target_exp_norm)

class SCC_loss():
    '''
    Assume contact map is approximately equal to exp(-Umatrix).
    Then compute loss as SCC(exp(-\hat{Umatrix}), exp(-Umatrix)).
    '''
    def __init__(self, m, exp=False, h=5, K=100, clip_val=None, norm=False):
        self.tscc = TORCH_SCC(m, h, K)
        self.exp = exp
        self.clip_val = clip_val

    def clip(self, arr):
        arr[arr > self.clip_val] = self.clip_val
        arr[arr < -self.clip_val] = -self.clip_val

        return arr

    def __call__(self, input, target, *args):
        N = input.shape[0]

        if self.clip_val is not None:
            input = self.clip(input)
            target = self.clip(target)

        if self.exp:
            input_exp = torch.exp(-input)
            target_exp = torch.exp(-target)

            scc = self.tscc(input_exp, target_exp, distance = True)
        else:
            scc = self.tscc(input, target, distance = True)

        loss = torch.mean(scc) / N
        return loss

class MSE_plaid():
    def __init__(self, log=False):
        if log:
            self.loss_fn = mse_log
        else:
            self.loss_fn = F.mse_loss

    def __call__(self, input, target):
        input_plaid = torch_subtract_diag(input)
        target_plaid = torch_subtract_diag(target)

        return self.loss_fn(input_plaid, target_plaid)

class MSE_diag():
    def __init__(self, log=False):
        if log:
            self.loss_fn = mse_log
        else:
            self.loss_fn = F.mse_loss

    def __call__(self, input, target):
        input_diag = torch_mean_dist(input)
        target_diag = torch_mean_dist(target)

        return self.loss_fn(input_diag, target_diag)

class MSE_log_scc():
    def __init__(self, m):
        self.weights = np.zeros(m)
        for d in np.arange(0, m-1):
            n = m - d

            weight = n * np.var(np.arange(1, n+1)/n, ddof = 1)
            assert weight > 0, d
            self.weights[d] = weight

        self.weights /= np.max(self.weights)

        weights_toep = scipy.linalg.toeplitz(self.weights)
        self.weights_toep = torch.tensor(weights_toep, dtype=torch.float32)

    def __call__(self, input, target):
        input_log = torch.sign(input) * torch.log(torch.abs(input) + 1)
        target_log = torch.sign(target) * torch.log(torch.abs(target) + 1)
        diff = input_log - target_log
        if diff.is_cuda and not self.weights_toep.is_cuda:
            self.weights_toep = self.weights_toep.to(diff.get_device())
        error = torch.multiply(diff, self.weights_toep)
        return torch.mean(torch.square(error))

class Combined_Loss():
    '''
    Generic class for combining multiple loss functions.
    '''
    def __init__(self, criterions, lambdas, args):
        self.criterions = criterions
        self.lambdas = lambdas
        self.args = args

    def __repr__(self):
        return f'Combined_Loss of ({self.criterions}, {self.lambdas}, {self.args})'


    def __call__(self, input, target, arg2=None, split_loss=False):
        loss_list = []
        tot_loss = 0

        zipper = zip(self.criterions, self.lambdas, self.args)
        for criterion, loss_lambda, arg1 in zipper:
            if arg1 is None and arg2 is None:
                loss = loss_lambda * criterion(input, target)
            else:
                assert arg1 is None or arg2 is None, f"I don't handle this case yet {arg1},{arg2}"
                if arg1 is None:
                    arg = arg2
                else:
                    arg = arg1

                if isinstance(arg, list):
                    loss = loss_lambda * criterion(input, target, *arg)
                else:
                    loss = loss_lambda * criterion(input, target, arg)

            try:
                if len(loss.shape) == 1:
                    loss = loss.item()
                loss_list.append(loss)
                tot_loss += loss
            except RuntimeError:
                print(criterion, loss_lambda, arg1, arg2, loss)
                raise

        if split_loss:
            return loss_list
        else:
            return tot_loss

if __name__ == '__main__':
    test()
