import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import calculate_diag_chi_step
from scipy.ndimage import uniform_filter
from torch.utils.data import Dataset

from ..argparse_utils import ArgparserConverter


def make_dataset(dir_list, minSample = 0, maxSample = float('inf'), verbose = False,
                samples = None, prefix = 'sample', use_ids = True, sub_dir = 'samples'):
    """
    Make list data file paths.

    Inputs:
        dir_list: data source directory (or list of directories)
        minSample: ignore samples < minSample
        maxSample: ignore samples > maxSample
        verbose: True for verbose mode
        samples: list/set of samples to include, None for all samples
        prefix: only folders starting with prefix will be considered
        use_ids:

    Outputs:
        data_file_arr: list of data file paths
    """
    data_file_arr = []

    if not isinstance(dir_list, list):
        dir_list = [dir_list]

    for dir in dir_list:
        samples_dir = osp.join(dir, sub_dir)
        l = len(prefix)
        sample_files = [f for f in os.listdir(samples_dir) if prefix in f]

        if use_ids:
            sample_ids = [file[l:] for file in sample_files]
            for sample_id in sorted(sample_ids):
                if sample_id.isnumeric():
                    sample_id_int = int(sample_id)
                    if sample_id_int < minSample:
                        continue
                    if sample_id_int > maxSample:
                        continue
                if (samples is None) or (sample_id in samples) or (sample_id.isnumeric() and int(sample_id) in samples):
                    data_file = osp.join(samples_dir, f'{prefix}{sample_id}')
                    data_file_arr.append(data_file)
        else:
            for file in sample_files:
                data_file_arr.append(osp.join(samples_dir, file))

    return data_file_arr
