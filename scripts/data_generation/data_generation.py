'''
Wrapper script for all data generation scripts.
Experimental data is assumed to be located at <ROOT>/dataset_all_files_512.
By default, synthetic data will write to <ROOT>/dataset_synthetic".
'''

import os.path as osp

import fit_max_ent
import generate_synthetic_parameters
import preprocess_max_ent
import run_simulations
from utils.utils import ROOT, config_getArgs

if __name__ == '__main__':
    jobs = 15 # number of jobs to use
    fit_max_ent.main(jobs)
    preprocess_max_ent.main()

    args = generate_synthetic_parameters.getArgs() # default args
    args.dataset = 'dataset_synthetic'
    args.exp_dataset = 'dataset_all_files_512'
    args.samples = 5000 # number of samples to generate
    args.k = 10
    args.m = 512
    args.b=200; args.v=8; args.ar=1.5
    args.cell_line='imr90'
    args.root = ROOT
    args.data_dir = ROOT
    generate_synthetic_parameters.main(args)

    args2 = run_simulations.getArgs()
    args2.start=1
    args2.end=args.samples
    args2.jobs=jobs
    args2.scratch = osp.join(ROOT, args.dataset) # Change to a scratch directory if applicable
    args2.data_folder = osp.join(ROOT, args.dataset) # output directory
                                                     # must match <args.data_dir>/<args.dataset>
    args2.m = args.m
    args2.overwrite = False

    args3 = config_getArgs()
    args3.n_sweeps = 300000
    args3.dump_frequency = 50000
    args3.TICGSeed = 10
    args3.diag_bins = 512
    args3.volume = 8
    args3.bead_vol = 130000
    args3.bond_length = 200
    run_simulations.main(args2, args3)
