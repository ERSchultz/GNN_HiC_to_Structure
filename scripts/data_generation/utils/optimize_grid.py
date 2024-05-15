import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pylib.utils.epilib as epilib
from pylib.optimize import get_bonded_simulation_xyz, optimize_config
from pylib.Pysim import Pysim
from pylib.utils import default
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.plotting_utils import plot_matrix, plot_mean_dist
from pylib.utils.xyz import xyz_to_contact_distance, xyz_to_contact_grid
from scipy import optimize
from sklearn.metrics import mean_squared_error

from .utils import ROOT


def run(dir, config):
    print(dir)
    if not osp.exists(dir):
        os.mkdir(dir, mode=0o755)
        sim = Pysim(dir, config, None, randomize_seed = False, overwrite = True)
        sim.run_eq(10000, config['nSweeps'], 1)

def bonded_simulations():
    dataset = osp.join(ROOT, 'dataset_bonded')
    if not osp.exists(dataset):
        os.mkdir(dataset, mode=0o755)

    base_config = default.bonded_config
    base_config['seed'] = 1
    base_config["nSweeps"] = 350000
    base_config["dump_frequency"] = 1000
    base_config["dump_stats_frequency"] = 100
    base_config['grid_size'] = 1000

    mapping = []
    for boundary_type, ar in [('spherical', 1.0)]:
        # ('spherical', 1.0), , ('spheroid', 2.0)
        boundary_dir = f'boundary_{boundary_type}'
        if ar != 1.0:
            boundary_dir += f'_{ar}'
        boundary_dir = osp.join(dataset, boundary_dir)
        if not osp.exists(boundary_dir):
            os.mkdir(boundary_dir, mode=0o755)
        for beadvol in [65000]:
            beadvol_dir = osp.join(boundary_dir, f'beadvol_{beadvol}')
            if not osp.exists(beadvol_dir):
                os.mkdir(beadvol_dir, mode=0o755)
            for bond_type in ['SC', 'gaussian']:
                bond_dir = osp.join(beadvol_dir, f'bond_type_{bond_type}')
                if not osp.exists(bond_dir):
                    os.mkdir(bond_dir, mode=0o755)
                if bond_type == 'SC':
                    k_bond = 0.02
                else:
                    k_bond = None
                if k_bond is not None:
                    bond_dir = osp.join(bond_dir, f'k_bond_{k_bond}')
                    if not osp.exists(bond_dir):
                        os.mkdir(bond_dir, mode=0o755)
                for m in [1024]:
                    m_dir = osp.join(bond_dir, f'm_{m}')
                    if not osp.exists(m_dir):
                        os.mkdir(m_dir, mode=0o755)
                    for b in [140]:
                        b_dir = osp.join(m_dir, f'bond_length_{b}')
                        if not osp.exists(b_dir):
                            os.mkdir(b_dir, mode=0o755)
                        for v in [6, 8, 10]:
                            v_dir = osp.join(b_dir, f'v_{v}')
                            if not osp.exists(v_dir):
                                os.mkdir(v_dir, mode=0o755)
                            for k_angle in [0, 2]:
                                if bond_type == 'DSS'and k_angle != 0:
                                    continue
                                for theta_0 in [180]:
                                    if theta_0 == 180:
                                        k_angle_dir = osp.join(v_dir, f'angle_{k_angle}')
                                    elif k_angle == 0:
                                        continue
                                    else:
                                        k_angle_dir = osp.join(v_dir, f'angle_{k_angle}_theta0_{theta_0}')

                                    config = base_config.copy()
                                    config['beadvol'] = beadvol
                                    config['bond_length'] = b
                                    if k_bond is not None:
                                        config['k_bond'] = k_bond
                                    config['target_volume'] = v
                                    config['nbeads'] = m
                                    config["bond_type"] = bond_type
                                    config['boundary_type'] = boundary_type
                                    config['aspect_ratio'] = ar
                                    if k_angle != 0:
                                        config['angles_on'] = True
                                        config['k_angle'] = k_angle
                                        config['theta_0'] = theta_0
                                    mapping.append((k_angle_dir, config))

    print(len(mapping))
    with mp.Pool(min(len(mapping), 12)) as p:
        p.starmap(run, mapping)

def main(root, config, mode):
    gthic = np.load(osp.join(osp.split(root)[0], 'hic.npy')).astype(float)
    config['nbeads'] = len(gthic)

    if mode in {'grid', 'distance'}:
        optimum = optimize_config(config, gthic, mode, 0.3, 2.4, root,
                                dataset = osp.join(ROOT, 'dataset_bonded'))
        p_s_exp = DiagonalPreprocessing.genomic_distance_statistics(gthic, 'prob')
        xyz = get_bonded_simulation_xyz(config, osp.join(ROOT, 'dataset_bonded'))
        if xyz is not None:
            if mode == 'grid':
                hic_sim = xyz_to_contact_grid(xyz, optimum, dtype=float)
            elif mode == 'distance':
                hic_sim = xyz_to_contact_distance(xyz, optimum, dtype=float)
            np.save(osp.join(root, 'hic.npy'), hic_sim)
            plot_matrix(hic_sim, osp.join(root, 'hic.png'))
            p_s_exp = DiagonalPreprocessing.genomic_distance_statistics(gthic, 'prob')
            p_s_sim = DiagonalPreprocessing.genomic_distance_statistics(hic_sim, 'prob')
            rmse = mean_squared_error(p_s_sim, p_s_exp, squared = False)
            diff = np.abs(p_s_exp[1] - p_s_sim[1])
            title = f'RMSE: {np.round(rmse, 5)}'
            plot_mean_dist(p_s_sim, root, 'mean_dist.png',
                            None, False, ref = p_s_exp,
                            ref_label = 'Reference',  label = f'Bonded {mode} optimal',
                            color = 'b', title = title)
            plot_mean_dist(p_s_sim, root, 'mean_dist_log.png',
                            None, True, ref = p_s_exp,
                            ref_label = 'Reference',  label = f'Bonded {mode} optimal',
                            color = 'b', title = title)

        print(f"optimal {mode} is: {optimum}")
        with open(osp.join(root, f'{mode}.txt'), 'w') as f:
            f.write(str(optimum))
    elif mode.startswith('angle'):
        assert config['angles_on']
        optimum = optimize_config(config, gthic, 'angle', 0.0, 2.0, root, 'neighbor_10')
        print(f"optimal angle is: {optimum}")
        with open(osp.join(droot, 'angle.txt'), 'w') as f:
            f.write(str(optimum))


    if mode.startswith('grid_angle'):
        os.mkdir('temp', mode=0o755)
        # move grid temporarily
        for file in os.listdir(root):
            shutil.move(osp.join(root, file), osp.join('temp', file))

        config['grid_size'] = optimum
        config['k_angle'] = 0.0
        config['angles_on'] = True

        s = mode[10:]
        optimum = optimize_config(config, gthic, 'angle', 0.0, 2.0, root,
                                    f'neighbor_{s}', mkdir=False)
        print(f"optimal angle is: {optimum}")
        with open(osp.join(root, 'angle.txt'), 'w') as f:
            f.write(str(optimum))

        # move grid back
        shutil.move('temp', osp.join(root, 'grid'))
        shutil.move(osp.join(root, 'grid/grid_size.txt'), osp.join(root, 'grid_size.txt'))

    return root, config


if __name__ == "__main__":
    bonded_simulations()
