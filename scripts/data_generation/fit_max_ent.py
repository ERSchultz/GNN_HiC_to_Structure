'''Fit maximum entropy simulations.'''

import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import numpy as np
import pylib.analysis as analysis
from pylib.Maxent import Maxent
from pylib.Pysim import Pysim
from pylib.utils import default, epilib
from pylib.utils.utils import load_import_log, load_json
from utils.setup_configs import setup_config
from utils.utils import ROOT, get_samples


def check(dataset, sample, bl=200, phi=None, v=8, vb=None,
        aspect_ratio=1.5, bond_type='gaussian', k=10, contacts_distance=False,
        k_angle=0, theta_0=180):
    root, _, _ = setup_max_ent(dataset, sample, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contacts_distance,
                                k_angle, theta_0, False)

    if osp.exists(root):
        params = load_json(osp.join(root, 'resources/params.json'))
        max_it = params['iterations']
        prod_sweeps = params['production_sweeps']
        if not osp.exists(osp.join(root, f'iteration{max_it}')):
            it=0
            for i in range(max_it):
                if osp.exists(osp.join(root, f'iteration{i}')):
                    it = i
            prcnt = np.round(it/max_it*100, 1)
            print(f'{root}: {prcnt}')
        elif not osp.exists(osp.join(root, f'iteration{max_it}/tri.png')):
            traj_file = osp.join(root, f'iteration{max_it}/production_out/energy.traj')
            if osp.exists(traj_file):
                traj = np.loadtxt(traj_file)
                sweep = traj[-1][0]
                prcnt = np.round(sweep/prod_sweeps*100, 1)
                print(f'{root}: final {prcnt}')
            else:
                print(f'{root}: final 0.0')
        else:
            print(f'{root}: complete')
    else:
        print(f'{root}: not started')

def setup_max_ent(dataset, sample, bl, phi, v, vb,
                aspect_ratio, bond_type, k, contacts_distance,
                k_angle, theta_0, verbose=True, return_dir=False):
    '''Set up config file for maximum entropy.'''
    if verbose:
        print(sample)
    data_dir = osp.join(ROOT, dataset)
    dir = osp.join(data_dir, f'samples/sample{sample}')

    root, config = setup_config(dir, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contacts_distance,
                                k_angle, theta_0, verbose)

    hic = np.load(osp.join(dir, 'hic.npy')).astype(float)

    config['nspecies'] = k
    if k > 0:
        config['chis'] = np.zeros((k,k))
    config['dump_frequency'] = 10000
    config['dump_stats_frequency'] = 100
    config['dump_observables'] = True

    # set up diag chis
    config['diagonal_on'] = True
    config['dense_diagonal_on'] = True
    config["small_binsize"] = 1
    if len(hic) == 512:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 16
        config["big_binsize"] = 28
    elif len(hic) == 256:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 12
        config["big_binsize"] = 16
    elif len(hic) == 1024:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 32
        config["big_binsize"] = 30
    elif len(hic) == 2560:
        config['n_small_bins'] = 64
        config["n_big_bins"] = 48
        config["big_binsize"] = 52
    elif len(hic) == 3270:
        config['n_small_bins'] = 70
        config["n_big_bins"] = 32
        config["big_binsize"] = 100
    else:
        raise Exception(f'Need to specify bin sizes for size={len(hic)}')

    config['diag_chis'] = np.zeros(config['n_small_bins']+config["n_big_bins"])

    root = osp.join(dir, f'{root}-max_ent{k}')
    if osp.exists(root):
        # shutil.rmtree(root)
        if verbose:
            print(f'WARNING: root exists: {root}')

    if return_dir:
        return dir, root, config, hic
    else:
        return root, config, hic

def fit(dataset, sample, bl=200, phi=None, v=8, vb=None,
        aspect_ratio=1.5, bond_type='gaussian', k=10, contacts_distance=False,
        k_angle=0, theta_0=180):
    '''
    Runs maximum entropy optimization to get optimal parameters for contact map
    located at <ROOT>/<dataset>/samples/sample<sample>

    Inputs:
        dataset: name of data folder
        sample: id of sample
        bl:  bond length
        phi: volume fraction
        v: volume
        vb: bead volume (None to infer)
        aspect_ratio: aspect ratio of spherical/spheroid boundary
        bond_type: type of bonded potential
        k: number of particle types (principal components)
        contacts_distance: True to calculate contacts based on distance instead
                        of grid (slower)
        k_angle: interaction energy for cosine angle term
        theta_0: target angle for cosine angle term
    '''
    dir = osp.join(ROOT, dataset, f'samples/sample{sample}')
    root, config, hic = setup_max_ent(dataset, sample, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contacts_distance,
                                k_angle, theta_0)
    import_log = load_import_log(dir)
    if osp.exists(root):
        return
    os.mkdir(root, mode=0o755)

    # get sequences
    seqs = epilib.get_pcs(epilib.get_oe(hic), k, normalize=True)

    params = default.params
    goals = epilib.get_goals(hic, seqs, config)
    params["goals"] = goals
    params['iterations'] = 20
    params['equilib_sweeps'] = 10000
    params['production_sweeps'] = 300000
    params['stop_at_convergence'] = True
    params['conv_defn'] = 'normal'

    stdout = sys.stdout
    with open(osp.join(root, 'log.log'), 'w') as sys.stdout:
        me = Maxent(root, params, config, seqs, hic, fast_analysis=True,
                    final_it_sweeps=300000, mkdir=False, bound_diag_chis=False)
        t = me.fit()
        print(f'Simulation took {np.round(t, 2)} seconds')
    sys.stdout = stdout

def cleanup(dataset, sample, bl=200, phi=None, v=8, vb=None,
        aspect_ratio=1.5, bond_type='gaussian', k=10, contacts_distance=False,
        k_angle=0, theta_0=180):
    '''Delete simulations that failed to complete.'''
    root, _, _ = setup_max_ent(dataset, sample, bl, phi, v, vb,
                                aspect_ratio, bond_type, k, contacts_distance,
                                k_angle, theta_0, False)

    remove = True
    if osp.exists(root):
        # if not osp.exists(osp.join(root, 'iteration0')):
            # remove = True
        if not osp.exists(osp.join(root, 'iteration20/tri.png')):
            # assumes 20 is final iteration
            remove = True
        if remove:
            print(f'removing {root}')
            shutil.rmtree(root)

def main(njobs=1):
    dataset='dataset_all_files_512'
    samples = []
    for cell_line in ['imr90']:
        samples_cell_line = get_samples(dataset, train=True, filter_cell_lines=cell_line)
        samples.extend(samples_cell_line)
        # samples_cell_line = get_samples(dataset, test=True, filter_cell_lines=cell_line)
        # samples.extend(samples_cell_line)
        print(samples)

    bond_type='gaussian';b=200
    k_angle=0;theta_0=180;ar=1.5;phi=None;v=8
    k=10
    contacts_distance=False

    mapping = []
    for i in samples:
        data_dir = osp.join(ROOT, dataset)
        dir = osp.join(data_dir, f'samples/sample{i}')
        mapping.append((dir, b, phi, v, None, ar,
                    bond_type, k, contacts_distance, k_angle, theta_0))
    with mp.Pool(njobs) as p:
        p.starmap(setup_config, mapping)

    mapping = []
    for i in samples:
        mapping.append((dataset, i, b, phi, v, None, ar,
                    bond_type, k, contacts_distance, k_angle, theta_0))
    print('len =', len(mapping))
    with mp.Pool(njobs) as p:
        p.starmap(fit, mapping)
        # p.starmap(check, mapping)
        # p.starmap(cleanup, mapping)

    for i in mapping:
        check(*i)

if __name__ == '__main__':
    main(15)
