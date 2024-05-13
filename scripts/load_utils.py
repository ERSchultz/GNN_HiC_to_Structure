'''
Utility functions for loading data from disk.
'''


import json
import os
import os.path as osp
import sys

import hicrep
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import (calculate_D, calculate_diag_chi_step,
                                      calculate_L, calculate_S)
from pylib.utils.utils import LETTERS, triu_to_full
from pylib.utils.xyz import xyz_load, xyz_to_contact_grid


## load data functions ##
def load_psi(sample_folder, throw_exception = True, verbose = False):
    x_files = ['x.npy', 'resources/x.npy', 'iteration0/x.npy']
    for f in x_files:
        f = osp.join(sample_folder, f)
        if osp.exists(f):
            x = np.load(f)
            if verbose:
                print(f'x loaded with shape {x.shape}')
            break
    else:
        if throw_exception:
            raise Exception(f'x not found for {sample_folder}')
        else:
            x = None

    assert not osp.exists(osp.join(sample_folder, 'psi.npy')), 'deprecated'

    if x is not None and x.shape[1] > x.shape[0]:
        x = x.T

    return x

def load_Y(sample_folder, throw_exception = True):
    y_file = osp.join(sample_folder, 'y.npy')
    y_file2 = osp.join(sample_folder, 'data_out/contacts.txt')
    y_file3 = osp.join(sample_folder, 'production_out/contacts.txt')
    y_file4 = osp.join(sample_folder, 'y.cool')
    y = None
    if osp.exists(y_file):
        y = np.load(y_file)
    elif osp.exists(y_file2):
        y = np.loadtxt(y_file2)
        np.save(y_file, y) # save in proper place
    elif osp.exists(y_file3):
        y = np.loadtxt(y_file3)
        np.save(y_file, y) # save in proper place
    elif osp.exists(y_file4):
        clr, binsize = hicrep.utils.readMcool(y_file4, -1)
        y = clr.matrix(balance=False).fetch('10')
        np.save(y_file, y) # save in proper place
    else:
        files = os.listdir(osp.join(sample_folder, 'production_out'))
        try:
            max_sweeps = -1
            for f in files:
                if f.startswith('contacts') and f.endswith('.txt'):
                    sweeps = int(f[8:-4])
                    if sweeps > max_sweeps:
                        max_sweeps = sweeps
            y = np.loadtxt(osp.join(sample_folder, 'production_out', f'contacts{max_sweeps}.txt'))
            np.save(y_file, y) # save in proper place
        except Exception as e:
            if throw_exception:
                raise e
            else:
                print(e)

    if y is None and throw_exception:
        raise Exception(f'y not found for {sample_folder}')
    else:
        y = y.astype(float)

    ydiag_file = osp.join(sample_folder, 'y_diag.npy')
    try:
        if osp.exists(ydiag_file):
            ydiag = np.load(ydiag_file)
        elif y is not None:
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
            ydiag = DiagonalPreprocessing.process(y, meanDist)
            np.save(ydiag_file, ydiag) # save in proper place
        else:
            ydiag = None
    except Exception:
        print(f'Exception when loading y_diag for {sample_folder}')
        raise

    return y, ydiag

def load_Y_diag(sample_folder, throw_exception = False):
    ydiag_file = osp.join(sample_folder, 'y_diag.npy')
    if osp.exists(ydiag_file):
        ydiag = np.load(ydiag_file)
    else:
        _, ydiag = load_Y(sample_folder)

    return ydiag

def load_L(sample_folder, psi = None, chi = None, save = False,
                throw_exception = True):
    '''
    Load L

    Inputs:
        sample_folder: path to sample
        psi: psi np array (None to load if needed)
        chi: chi np array (None to load if needed)
        save: True to save L.npy
        throw_exception: True to throw exception if L missing
    '''
    calc = False # TRUE if need to calculate L

    load_fns = [np.load, np.loadtxt]
    files = [osp.join(sample_folder, i) for i in ['L.npy', 'L_matrix.txt']]
    for f, load_fn in zip(files, load_fns):
        if osp.exists(f):
            L = load_fn(f)
            break
    else:
        L = None
        calc = True

    if calc:
        if psi is None:
            psi = load_psi(sample_folder, throw_exception=throw_exception)
        if chi is None:
            chi = load_chi(sample_folder, throw_exception=throw_exception)
        if psi is not None and chi is not None:
            L = calculate_L(psi, chi)

        if save and L is not None:
            np.save(osp.join(sample_folder, 'L.npy'), L)

    return L

def load_D(sample_folder, throw_exception = True):
    config_file = osp.join(sample_folder, 'config.json')
    with open(config_file, 'r') as f:
        config = json.load(f)
        diag_chi_step = calculate_diag_chi_step(config)

    D = calculate_D(diag_chi_step)
    return D

def load_S(sample_folder, throw_exception = True):
    L = load_L(sample_folder, throw_exception=throw_exception)
    D = load_D(sample_folder, throw_exception=throw_exception)
    S = calculate_S(L, D)

    return S

def load_all(sample_folder, plot = False, data_folder = None, log_file = None,
                save = False, experimental = False, throw_exception = True):
    '''Loads x, psi, chi, chi_diag, L, y, ydiag.'''
    y, ydiag = load_Y(sample_folder, throw_exception = throw_exception)

    if experimental:
        # everything else is None
        return None, None, None, None, None, None, y, ydiag

    x = load_psi(sample_folder, throw_exception = throw_exception)
    # x = x.astype(float)

    if plot and x is not None:
        m, k = x.shape
        for i in range(k):
            plt.plot(x[:, i])
            plt.title(r'$X$[:, {}]'.format(i))
            plt.savefig(osp.join(sample_folder, 'x_{}'.format(i)))
            plt.close()

    chi = load_chi(sample_folder, throw_exception)
    if chi is not None:
        chi = chi.astype(np.float64)
        if log_file is not None:
            print('Chi:\n', chi, file = log_file)

    chi_diag = None
    config_file = osp.join(sample_folder, 'config.json')
    if osp.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            if "diag_chis" in config:
                chi_diag = np.array(config["diag_chis"])

    L = load_L(sample_folder, x, chi, save = save,
                    throw_exception = throw_exception)

    return x, chi, chi_diag, L, y, ydiag

def load_chi(dir, throw_exception=True):
    if osp.exists(osp.join(dir, 'chis.txt')):
        chi = np.loadtxt(osp.join(dir, 'chis.txt'))
        chi = np.atleast_2d(chi)[-1]
        return triu_to_full(chi)
    elif osp.exists(osp.join(dir, 'chis.npy')):
        chi = np.load(osp.join(dir, 'chis.npy'))
        return chi
    elif osp.exists(osp.join(dir, 'config.json')):
        with open(osp.join(dir, 'config.json'), 'rb') as f:
            config = json.load(f)

        if 'chis' in config:
            chi = np.array(config['chis'])
            return chi


    if throw_exception:
        raise Exception(f'chi not found for {dir}')

    return None

def load_max_ent_chi(k, path, throw_exception = True):

    config_file = osp.join(path, 'config.json')
    if osp.exists(config_file):
        with open(config_file, 'rb') as f:
            config = json.load(f)
    else:
        return None

    try:
        chi = config['chis']
        chi = np.array(chi)
    except:
        chi = np.zeros((k,k))
        for i, bead_i in enumerate(LETTERS[:k]):
            for j in range(i,k):
                bead_j = LETTERS[j]
                try:
                    chi[i,j] = config[f'chi{bead_i}{bead_j}']
                except KeyError:
                    if throw_exception:
                        print(f'config_file: {config_file}')
                        print(config)
                        raise
                    else:
                        return None

    return chi

def get_final_max_ent_folder(replicate_folder, throw_exception = True, return_it = False):
    '''Find final max ent iteration within replicate folder.'''
    max_it = -1
    for file in os.listdir(replicate_folder):
        if osp.isdir(osp.join(replicate_folder, file)) and 'iteration' in file:
            start = file.find('iteration')+9
            it = int(file[start:])
            if it > max_it:
                max_it = it

    if max_it < 0:
        if throw_exception:
            raise Exception(f'max it not found for {replicate_folder}')
        else:
            return None

    final_folder = osp.join(replicate_folder, f'iteration{max_it}')
    if return_it:
        return final_folder, max_it
    return final_folder

def get_converged_max_ent_folder(replicate_folder, conv_defn, throw_exception=True,
                                return_it = False):
    if conv_defn == 'strict':
        eps = 1e-3
    elif conv_defn == 'normal':
        eps = 1e-2
    else:
        raise Exception(f'Unrecognized conv_defn: {conv_defn}')

    conv_file = osp.join(replicate_folder, 'convergence.txt')
    assert osp.exists(conv_file), f'conv_file does not exists: {conv_file}'
    conv = np.atleast_1d(np.loadtxt(conv_file))
    converged_it = None
    for j in range(1, len(conv)):
        diff = conv[j] - conv[j-1]
        if np.abs(diff) < eps and conv[j] < conv[0]:
            converged_it = j
            break

    if converged_it is not None:
        if return_it:
            return converged_it
        else:
            final = osp.join(replicate_folder, f'iteration{j}')
            return final
    elif throw_exception:
        raise Exception(f'{replicate_folder} did not converge')
    else:
        return None

def load_max_ent_D(path):
    if path is None:
        return None

    final = get_final_max_ent_folder(path)

    config_file = osp.join(final, 'config.json')
    if osp.exists(config_file):
        with open(config_file, 'rb') as f:
            config = json.load(f)
    else:
        return None

    diag_chis_step = calculate_diag_chi_step(config)

    D = calculate_D(diag_chis_step)
    return D

def load_max_ent_L(path, throw_exception=False):
    if path is None:
        return None
    # load x
    x = load_psi(path, throw_exception=throw_exception)
    if x is None:
        return

    _, k = x.shape
    # load chi
    chi = load_chi(path, throw_exception)

    if chi is None and throw_exception:
        raise Exception(f'chi not found: {path}')

    # calculate s
    L = calculate_L(x, chi)

    if L is None and throw_exception:
        raise Exception(f'L is None: {path}')

    return L

def load_max_ent_S(path, throw_exception=False):
    L = load_max_ent_L(path, throw_exception = throw_exception)
    D = load_max_ent_D(path)
    return calculate_S(L, D)
