import os
import os.path as osp

from pylib.utils.utils import load_import_log

ROOT = '/home/erschultz'
if not osp.exists(ROOT):
    print(f'import_contact_maps.py: root directory not found: {ROOT}'
            f'\n\tUsing working directory: {os.getcwd()}')
    ROOT = os.getcwd()

def get_samples(dataset, train=False, test=False, return_cell_lines=False,
                filter_cell_lines=None):
    '''
    Inputs:
        dataset: data directory
        train: True to only return sample from odd chrom (training samples)
    '''
    data_folder = osp.join(ROOT, dataset)
    assert osp.exists(data_folder)
    samples_dir = osp.join(data_folder, 'samples')

    samples = [f[6:] for f in os.listdir(samples_dir) if f.startswith('sample')]

    cell_lines = []; odd_cell_lines = []; even_cell_lines = []
    odd_samples = []; even_samples = []
    for s in samples:
        s_dir = osp.join(samples_dir, f'sample{s}')
        assert(osp.exists(s_dir)), s_dir

        result = load_import_log(s_dir)
        chrom = int(result['chrom'])
        cell_line = result['cell_line']
        if cell_line is not None:
            cell_line = cell_line.lower()
        cell_lines.append(cell_line)

        if cell_line is None:
            pass
            # print(f'cell_line is None, skipping {s}: url={result["url"]}')
            # continue
        elif filter_cell_lines is not None:
            if cell_line not in filter_cell_lines:
                continue

        if chrom % 2 == 1:
            odd_samples.append(s)
            odd_cell_lines.append(cell_line)
        else:
            even_samples.append(s)
            even_cell_lines.append(cell_line)

    if train:
        samples = odd_samples
        cell_lines = odd_cell_lines
    elif test:
        samples = even_samples
        cell_lines = even_cell_lines
    else:
        samples = even_samples + odd_samples
        cell_lines = even_cell_lines + odd_cell_lines


    if return_cell_lines:
        if cell_lines is not None:
            unique_cell_lines = set(cell_lines)
            if len(unique_cell_lines) == 1 and None in unique_cell_lines:
                cell_lines = None
        return samples, cell_lines
    else:
        return samples
