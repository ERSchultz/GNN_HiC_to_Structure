import argparse
import csv
import json
import os
import os.path as osp
import shutil
import sys

import numpy as np
import pylib.utils.default as default
from pylib.utils.ArgparseConverter import ArgparseConverter
from pylib.utils.energy_utils import calculate_D, calculate_diag_chi_step
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import LETTERS, crop

from .optimize_grid import main as optimize_grid


def setup_config(dir, bl=200, phi=None, v=8,
                vb=None, aspect_ratio=1.5, bond_type='gaussian', k=None,
                contacts_distance=False, k_angle=0, theta_0=180, verbose=True):
    '''Set up generic config file for maximum entropy or GNN.'''
    if verbose:
        print('setup_config for', dir)
    assert osp.exists(dir), f'{dir} does not exist'

    bonded_config = default.bonded_config.copy()
    bonded_config['bond_length'] = bl
    assert phi is None or v is None
    if phi is not None:
        bonded_config['phi_chromatin'] = phi
    if v is not None:
        bonded_config['target_volume'] = v
    bonded_config['bond_type'] = bond_type
    if bond_type == 'SC':
        bonded_config['k_bond'] = 0.02
    bonded_config['update_contacts_distance'] = contacts_distance
    if k_angle != 0:
        bonded_config['angles_on'] = True
        bonded_config['k_angle'] = k_angle
        bonded_config['theta_0'] = theta_0
    if bonded_config['update_contacts_distance']:
        mode = 'distance'
    else:
        mode = 'grid'
    if vb is not None:
        bonded_config['beadvol'] = vb
    else:
        if bonded_config['bond_length'] <= 100:
            bonded_config['beadvol'] = 13000
        elif bonded_config['bond_length'] == 140:
            bonded_config['beadvol'] = 65000
        else:
            bonded_config['beadvol'] = 130000
    if aspect_ratio != 1.0:
        bonded_config['boundary_type'] = 'spheroid'
        bonded_config['aspect_ratio'] = aspect_ratio

    root = f"optimize_{mode}"
    if phi is not None:
        assert v is None
        root = f"{root}_b_{bl}_phi_{phi}"
    else:
        root = f"{root}_b_{bl}_v_{v}"
    if bonded_config['angles_on']:
        root += f"_angle_{bonded_config['k_angle']}_theta0_{bonded_config['theta_0']}"
    if bonded_config['boundary_type'] == 'spheroid':
        root += f'_spheroid_{aspect_ratio}'
    if bonded_config['bond_type'] != 'gaussian':
        root += f'_{bonded_config["bond_type"]}'

    root = osp.join(dir, root)
    optimum_file = osp.join(root, f'{mode}.txt')
    if osp.exists(optimum_file):
        if mode == 'grid':
            bonded_config['grid_size'] = np.loadtxt(optimum_file)
        elif mode == 'distance':
            bonded_config["distance_cutoff"] = np.loadtxt(optimum_file)
            bonded_config['grid_size'] = 200 # TODO
        angle_file = osp.join(root, 'angle.txt')
        if osp.exists(angle_file):
            bonded_config['k_angle'] = np.loadtxt(angle_file)
            bonded_config['angles_on'] = True
    else:
        if osp.exists(root):
            shutil.rmtree(root)
        root, bonded_config = optimize_grid(root, bonded_config, mode)

    config = default.config
    for key in ['beadvol', 'bond_length', 'phi_chromatin', 'target_volume',
                'grid_size', 'distance_cutoff', 'k_angle', 'angles_on', 'theta_0', 'boundary_type',
                'update_contacts_distance', 'aspect_ratio', 'bond_type']:
        if key in bonded_config:
            config[key] = bonded_config[key]

    return root, config


### config functions for synethetic data ###
# TODO clean this, should be redundant with setup_config above
def config_getArgs(args_file=None, args_tmp=None):
    parser = argparse.ArgumentParser(description='Base parser', fromfile_prefix_chars='@',
                                allow_abbrev = False)
    AC = ArgparseConverter()

    parser.add_argument('--config_ifile', type=str, default='config.json',
                        help='path to default config file')
    parser.add_argument('--config_ofile', type=str, default='config.json',
                            help='path to output config file')
    parser.add_argument('--args_file', type=AC.str2None,
                        help='file with more args to load')

    # config params
    parser.add_argument('--m', type=int, default=-1,
                        help='number of particles (-1 to infer)')
    parser.add_argument('--load_configuration_filename', type=AC.str2None,
                        help='file name of initial config (None to not load)')
    parser.add_argument('--dump_frequency', type=int,
                        help='set to change dump frequency')
    parser.add_argument('--dump_stats_frequency', type=int,
                        help='set to change dump stats frequency')
    parser.add_argument('--n_sweeps', type=int,
                        help='set to change nSweeps')
    parser.add_argument('--TICG_seed', type=AC.str2int,
                        help='set to change random seed for simulation (None for random)')
    parser.add_argument('--use_ground_truth_TICG_seed', type=AC.str2bool,
                        help='True to copy seed from config file in sample_folder')
    parser.add_argument('--sample_folder', type=str,
                        help='location of sample for ground truth chi')
    parser.add_argument('--bond_type', type=AC.str2None, default='gaussian',
                        help='type of bonded interaction')
    parser.add_argument('--parallel', type=AC.str2bool, default=False,
                        help='True to run simulation in parallel')
    parser.add_argument('--num_threads', type=int, default=2,
                        help='Number of threads if parallel is True')
    parser.add_argument('--phi_chromatin', type=AC.str2float,
                        help='chromatin volume fraction')
    parser.add_argument('--volume', type=int,
                        help='simulation volume')
    parser.add_argument('--bond_length', type=float, default=16.5,
                        help='bond length')
    parser.add_argument('--k_angle', type=float, default=0.0,
                        help='k for angle term')
    parser.add_argument('--boundary_type', type=str, default='spherical',
                        help='simulation boundary type {cubic, spherical}')
    parser.add_argument('--aspect_ratio', type=float, default=1.0,
                        help='for simulation boundary type = spheroid')
    parser.add_argument('--track_contactmap', type=AC.str2bool, default=False,
                        help='True to dump contact map every dump_frequency')
    parser.add_argument('--gridmove_on', type=AC.str2bool, default=True,
                        help='True to use grid MC move')
    parser.add_argument('--grid_size',
                        help='TICG grid size (None to load from optimize_grid)')
    parser.add_argument('--bead_vol', type=float, default=520,
                        help='bead volume')
    parser.add_argument('--update_contacts_distance', type=AC.str2bool, default=False,
                        help='True to use distance instead of grid')

    # chi config params
    parser.add_argument('--use_ground_truth_chi', type=AC.str2bool, default=False,
                        help='True to use ground truth chi and diag chi')

    # constant chi config params
    parser.add_argument('--constant_chi', type=float, default=0,
                        help='constant chi parameter between all beads')

    # energy config params
    parser.add_argument('--use_umatrix', type=AC.str2bool, default=True,
                        help='True to use U_matrix')
    parser.add_argument('--use_dmatrix', type=AC.str2bool, default=True,
                        help='True to use d_matrix')
    parser.add_argument('--use_lmatrix', type=AC.str2bool, default=True,
                        help='True to use l_matrix')

    parser.add_argument('--e_constant', type=float, default=0,
                        help='constant to add to e')
    parser.add_argument('--s_constant', type=float, default=0,
                        help='constant to add to s')

    # diagonal config params
    parser.add_argument('--diag_pseudobeads_on', type=AC.str2bool, default=True)
    parser.add_argument('--dense_diagonal_on', type=AC.str2bool, default=False,
                        help='True to place 1/2 of beads left of cutoff')


    # max_ent options
    parser.add_argument('--max_ent', action="store_true",
                        help='true to save chi to wd in format needed for max ent')
    parser.add_argument('--mode', type=str,
                        help='mode for max_ent')


    args, _ = parser.parse_known_args()
    if args.args_file is None and args_file is not None:
        args.args_file = args_file
    if args.args_file is not None:
        assert osp.exists(args.args_file), f'{args.args_file} does not exist'
        print(f'parsing {args.args_file}')
        argv = sys.argv.copy()
        print(f'argv: {argv}\n')
        argv.append(f'@{args.args_file}') # appending means args_file will override other args
        print(f'argv: {argv}\n')
        args, unknown = parser.parse_known_args(argv)
        print(f'unknown: {unknown}\n')

    if args_tmp is not None:
        # hacky solution
        args.n_sweeps = args_tmp.n_sweeps
        args.dump_frequency = args_tmp.dump_frequency
        args.TICG_seed = args_tmp.TICG_seed
        args.phi_chromatin = args_tmp.phi_chromatin
        args.volume = args_tmp.volume
        args.bead_vol = args_tmp.bead_vol
        args.bond_length = args_tmp.bond_length

    return args

def writeSeq(seq, format='%.8e'):
    m, k = seq.shape
    for j in range(k):
        np.savetxt(f'pcf{j+1}.txt', seq[:, j], fmt = format)

def get_config(args_file=None, args_tmp=None):
    args = config_getArgs(args_file, args_tmp)
    print(args)

    with open(args.config_ifile, 'rb') as f:
        config = json.load(f)

    if args.m == -1:
        # need to infer m
        if osp.exists('x.npy'):
            x = np.load('x.npy')
            args.m, _ = x.shape
        elif osp.exists('U.npy'):
            U = np.load('U.npy')
            args.m, _ = U.shape
        elif args.sample_folder is not None:
            y_file = osp.join(args.sample_folder, 'y.npy')
            if osp.exists(y_file):
                args.m = len(np.load(y_file))

        if args.m == -1:
            raise Exception('Could not infer m')
        else:
            print(f'inferred m = {args.m}')

    # save nbeads
    config['nbeads'] = args.m

    # infer k
    if osp.exists('psi.npy'):
        psi = np.load('psi.npy')
        _, args.k = psi.shape
    elif osp.exists('x.npy'):
        x = np.load('x.npy')
        _, args.k = x.shape
    else:
        args.k = 0
    print(f'inferred k = {args.k}')

    # load chi
    chi_file = 'chis.npy'
    if args.use_ground_truth_chi:
        assert args.sample_folder is not None
        args.chi = np.load(osp.join(args.sample_folder, 'chis.npy'))
        _, k = args.chi.shape
        np.save(chi_file, args.chi)
        assert k == args.k, f"cols of ground truth chi {args.k} doesn't match cols of seq {k}"
    elif osp.exists(chi_file):
        args.chi = np.load(chi_file)

        if args.max_ent:
            with open('chis.txt', 'w', newline='') as f:
                wr = csv.writer(f, delimiter = '\t')
                wr.writerow(args.chi[np.triu_indices(args.k)])
                wr.writerow(args.chi[np.triu_indices(args.k)])
    else:
        args.chi = None

    # save dense_diagonal
    config['dense_diagonal_on'] = args.dense_diagonal_on

    # load diag chi
    if osp.exists('diag_chis.npy'):
        config["diagonal_on"] = True
        if args.max_ent:
            diag_chis = np.load('diag_chis.npy')
            print(f'diag_chis loaded with shape {diag_chis.shape}')
            with open('chis_diag.txt', 'w', newline='') as f:
                wr = csv.writer(f, delimiter = '\t')
                wr.writerow(diag_chis)
                wr.writerow(diag_chis)
    else:
        config["diagonal_on"] = False

    if args.use_dmatrix:
        diag_chis = np.load('diag_chis.npy')
        if len(diag_chis) == args.m:
            D = calculate_D(diag_chis)
        else:
            diag_chis_step = calculate_diag_chi_step(config, diag_chis)
            D = calculate_D(diag_chis_step)

    # save diag_pseudobeads_on
    config['diag_pseudobeads_on'] = args.diag_pseudobeads_on

    # set up psi
    if osp.exists('psi.npy'):
        psi = np.load('psi.npy')
        print(f'psi loaded with shape {psi.shape}')
    elif osp.exists('x.npy'):
        psi = np.load('x.npy')
        print(f'psi (x) loaded with shape {psi.shape}')
    else:
        psi = None
        print('psi is None')

    if args.s_constant != 0 or args.e_constant != 0:
        raise Exception('deprecated')

    # set up e, s
    if psi is not None:
        writeSeq(psi)

        # save seq
        config['bead_type_files'] = [f'pcf{i}.txt' for i in range(1, args.k+1)]

        # save nspecies
        config["nspecies"] = args.k

        # save chi to config
        config['chis'] = args.chi.tolist()
    elif args.use_umatrix:
        config['bead_type_files'] = None
        config["nspecies"] = 0
    else:
        config['plaid_on'] = False
        config['bead_type_files'] = None
        config["nspecies"] = 0

    if args.use_dmatrix:
        config['dmatrix_on'] = True
    else:
        config['dmatrix_on'] = False
    if args.use_lmatrix:
        config['lmatrix_on'] = True
    else:
        config['lmatrix_on'] = False
    if args.use_umatrix:
        config['umatrix_on'] = True
    else:
        config['umatrix_on'] = False

    if not config['plaid_on'] and not config["diagonal_on"]:
        config['nonbonded_on'] = False

    # save bond type
    if args.bond_type is None:
        config['bond_type'] = 'none'
        config['bonded_on'] = False
        config["displacement_on"] = True
        config["translation_on"] = False
        config["crankshaft_on"] = False
        config["pivot_on"] = False
        config["rotate_on"] = False
    else:
        config['bond_type'] = args.bond_type
        if args.bond_type == 'gaussian':
            config["rotate_on"] = False

    # save nSweeps
    if args.n_sweeps is not None:
        config['nSweeps'] = args.n_sweeps

    # save phi_chromatin
    if args.phi_chromatin is None:
        assert args.volume is not None
        config['target_volume'] = args.volume
    else:
        config['phi_chromatin'] = args.phi_chromatin

    # save bond_length
    bond_length_file = 'bond_length.txt'
    if osp.exists(bond_length_file):
        args.bond_length = float(np.loadtxt(bond_length_file))
        print('Bond_length:', args.bond_length)
    config['bond_length'] = args.bond_length

    # save boundary_type
    config['boundary_type'] = args.boundary_type
    if args.aspect_ratio != 1:
        assert config['boundary_type'] == 'spheroid'
        config['aspect_ratio'] = args.aspect_ratio

    # save dump frequency
    if args.dump_frequency is not None:
        config['dump_frequency'] = args.dump_frequency
    if args.dump_stats_frequency is not None:
        config['dump_stats_frequency'] = args.dump_stats_frequency

    # save track_contactmap
    config['track_contactmap'] = args.track_contactmap

    # save gridmove_on
    config['gridmove_on'] = args.gridmove_on

    # save grid_size
    grid_size = None
    if osp.exists(args.grid_size):
        if args.grid_size.endswith('json'):
            with open(args.grid_size, 'r') as f:
                tmp = json.load(f)
                grid_size = tmp['grid_size']
        elif args.grid_size.endswith('txt'):
            arr = np.loadtxt(args.grid_size)
            arr = np.atleast_1d(arr)
            print(f'shape of args.grid_size = {arr.shape}')
            if len(arr) > 1:
                grid_size = arr[-1]
            else:
                grid_size = arr.item()
    elif args.grid_size.startswith('scale_'):
        scale = float(args.grid_size[6:])
        grid_size = args.bond_length * scale
    elif args.grid_size.lower() == 'none':
        assert args.sample_folder is not None
        if args.bond_length.is_integer():
            b = int(args.bond_length)
        else:
            b = args.bond_length
        if args.k_angle != 0:
            optimize_folder = osp.join(args.sample_folder,
                        f'optimize_grid_angle_{args.k_angle}_b_{b}_phi_{args.phi_chromatin}')
        else:
            optimize_folder = osp.join(args.sample_folder,
                        f'optimize_grid_b_{b}_phi_{args.phi_chromatin}')
        arr = np.loadtxt(osp.join(optimize_folder, 'grid_size.txt'))
        arr = np.atleast_1d(arr)
        grid_size = arr.item()
    else:
        grid_size = float(args.grid_size)
    if grid_size is None:
        raise Exception(f'Invalid grid_size {args.grid_size}')
    config['grid_size'] = grid_size

    # save k_angle
    if args.k_angle != 0:
        config['angles_on'] = True
        config['k_angle'] = args.k_angle

    # save bead volume
    config['beadvol'] = args.bead_vol

    # save update_contacts_distance
    config['update_contacts_distance'] = args.update_contacts_distance
    if args.update_contacts_distance:
        config['distance_cutoff'] = grid_size

    # save seed
    if args.TICG_seed is not None:
        config['seed'] = args.TICG_seed
    else:
        rng = np.random.default_rng()
        config['seed'] = int(rng.integers(1000)) # random int in [0, 1000)

    # save configuration filename
    if args.load_configuration_filename is None:
        config["load_configuration"] = False
        config["load_configuration_filename"] = 'none'
    else:
        config["load_configuration_filename"] = args.load_configuration_filename

    config['constant_chi'] = args.constant_chi
    if args.constant_chi > 0:
        config['constant_chi_on'] = True
    if args.max_ent and args.mode == 'all':
        config['constant_chi_on'] = True # turn on even if value = 0
        with open('chi_constant.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow([args.constant_chi])
            wr.writerow([args.constant_chi])
    if args.max_ent and args.mode.startswith('grid_size'):
        with open('grid_size.txt', 'w', newline='') as f:
            wr = csv.writer(f, delimiter = '\t')
            wr.writerow([args.grid_size])
            wr.writerow([args.grid_size])


    # save parallel
    config['parallel'] = args.parallel
    config['num_threads'] = args.num_threads


    with open(args.config_ofile, 'w') as f:
        json.dump(config, f, indent = 2)
