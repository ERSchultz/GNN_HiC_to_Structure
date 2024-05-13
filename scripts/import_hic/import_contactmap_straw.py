'''
Functions for importing experimental Hi-C contact maps.

import_contactmap_straw interacts with .hic files.
Most other functions are some form of wrapper.
'''

import csv
import multiprocessing
import os
import os.path as osp

import bioframe  # https://github.com/open2c/bioframe
import hicstraw  # https://github.com/aidenlab/straw
import numpy as np
import pandas as pd
from pylib.utils.hic_utils import rescale_p_s_1
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import load_import_log
from utils import *

ROOT = '/home/erschultz'
assert osp.exists(ROOT), f'neet to set root directory: {ROOT}'

def import_contactmap_straw(odir, hic_filename, chrom, start,
                            end, resolution, norm='NONE'):
    '''
    Load .hic file with hicstraw and write to disk as y.npy. Experimental details
    are logged in odir/import.log.

    Inputs:
        odir: output directory
        hic_filname: path to .hic file
        chrom: chromosome
        start: start basepair
        end: end basepair
        resolution: Hi-C resolution in basepairs
        norm: Hi-C normlalization method (e.g. KR, NONE)
    '''
    basepairs = f"{chrom}:{start}:{end}"
    print(basepairs, odir)
    result = hicstraw.straw("observed", norm, hic_filename, basepairs, basepairs, "BP", resolution)
    hic = hicstraw.HiCFile(hic_filename)

    m = int((end - start) / resolution)
    y_arr = np.zeros((m, m))
    output = []
    for row in result:
        i = int((row.binX - start) / resolution)
        j = int((row.binY - start) / resolution)
        if i >= m or j >= m:
            continue
        try:
            y_arr[i, j] = row.counts
            y_arr[j, i] = row.counts
            if multiHiCcompare:
                output.append([chrom, row.binX, row.binY, row.counts])
        except Exception as e:
            print(e)
            print(row.binX, row.binY, row.counts, i, j)

    if np.max(y_arr) == 0:
        print(f'{odir} had no reads')
        return

    os.makedirs(odir, exit_ok = True)
    with open(osp.join(odir, 'y_sparse.txt'), 'w') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerows(output)

    m, _ = y_arr.shape

    with open(osp.join(odir, 'import.log'), 'w') as f:
        if isinstance(chrom, str):
            chrom = chrom.strip('chr')
        f.write(f'{hic_filename}\nchrom={chrom}\nstart={start}\nend={end}\n')
        f.write(f'resolution={resolution}\nbeads={m}\nnorm={norm}\n')
        f.write(f'genome={hic.getGenomeID()}')

    np.save(osp.join(odir, 'y.npy'), y_arr)
    print(f'{odir} done')

def entire_chromosomes(hic_files, dataset, resolution=50000,
                        ref_genome='hg19', chroms=range(1,23), jobs=15):
    for i, filename in enumerate(hic_files):
        data_folder = osp.join(ROOT, dataset)
        os.makedirs(data_folder, exist_ok = True)
        odir = osp.join(ROOT, dataset, f'chroms_{i}')
        os.makedirs(odir, exist_ok = True)

        chromsizes = bioframe.fetch_chromsizes(ref_genome)
        mapping = []
        for i, chromosome in enumerate(chroms):
            i += 1 # switch to 1-based indexing
            start = 0
            end = chromsizes[f'chr{chromosome}']
            print(f'i={i}: chr{chromosome} {start}-{end}')
            sample_folder = osp.join(odir, f'chr{chromosome}')
            mapping.append((sample_folder, filename, chromosome, start,
                            end, resolution, norm, multiHiCcompare))

        with multiprocessing.Pool(jobs) as p:
            p.starmap(import_contactmap_straw, mapping)

        for chr in chroms:
            y = np.load(osp.join(odir, f'chr{chr}', 'y.npy'))
            plot_matrix(y, osp.join(odir, f'chr{chr}', 'y.png'), vmax='mean')

def split_chromosomes(in_dataset, out_dataset, m, chroms=range(1,23), start_index=1,
                        resolution=50000, ref_genome='hg19', seed=None,
                        file = 'y.npy', scale=None):
    data_dir = osp.join('/home/erschultz', in_dataset)
    out_data_dir = osp.join('/home/erschultz', out_dataset)
    if not osp.exists(out_data_dir):
        os.mkdir(out_data_dir, mode=0o755)
    samples_dir = osp.join(out_data_dir, 'samples')
    if not osp.exists(samples_dir):
        os.mkdir(samples_dir, mode=0o755)

    cell_lines = ['50k']

    i = start_index
    chromsizes = bioframe.fetch_chromsizes(ref_genome)
    for cell_line in cell_lines:
        print(cell_line)
        for chrom in chroms:
            chrom_dir = osp.join(data_dir, f'chroms_{cell_line}/chr{chrom}')
            y_file = osp.join(chrom_dir, file)
            if y_file.endswith('.txt'):
                y = np.loadtxt(y_file)
            elif y_file.endswith('.npy'):
                y = np.load(y_file)
            size = len(y)
            diag = y.diagonal() == 0
            ind = np.arange(0, len(diag))

            import_log = load_import_log(chrom_dir)

            start = 0
            end = start + m
            while end < size:
                if np.sum(diag[start:end]) > 0:
                    start = end - np.argmax(np.flip(diag[start:end]))
                    end = start + m
                    continue

                print(f'\ti={i}: chr{chrom} {start}-{end}')
                odir = osp.join(samples_dir, f'sample{i}')
                if not osp.exists(odir):
                    os.mkdir(odir, mode=0o755)

                with open(osp.join(odir, 'import.log'), 'w') as f:
                    if isinstance(chrom, str):
                        chrom = chrom.strip('chr')
                    f.write(f'{import_log["url"]}\nchrom={chrom}\n')
                    f.write(f'resolution={resolution}\nbeads={m}\nnorm={import_log["norm"]}\n')
                    f.write(f'start={int(start*resolution)}\nend={int(end*resolution)}\n')
                    f.write(f'genome={import_log["genome"]}')


                y_i = y[start:end,start:end]
                if scale is not None:
                    # rescale Hi-C map such that first off_diagional has mean value of 'scale'
                    y_i = rescale_p_s_1(y_i, scale)
                    np.fill_diagonal(y_i, 1)
                np.save(osp.join(odir, 'y.npy'), y_i)
                plot_matrix(y_i, osp.join(odir, 'y.png'), vmax='mean')
                i += 1
                start = end
                end = start + m

if __name__ == '__main__':
    entire_chromosomes_list(ALL_FILES_in_situ, 'dataset_all_files')
    split('dataset_all_files', 'dataset_all_files_512', 512, file='y.npy', scale=1e-1)
