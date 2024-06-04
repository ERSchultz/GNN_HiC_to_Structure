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
if not osp.exists(ROOT):
    print(f'import_contact_maps.py: root directory not found: {ROOT}'
            f'\n\tUsing working directory: {os.getcwd()}')
    ROOT = os.getcwd()

def import_contactmap_straw(odir, hic_filename, chrom, start,
                            end, resolution, norm='NONE'):
    '''
    Load .hic file with hicstraw and write to disk as hic.npy. Experimental details
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
    hic_arr = np.zeros((m, m))
    for row in result:
        i = int((row.binX - start) / resolution)
        j = int((row.binY - start) / resolution)
        if i >= m or j >= m:
            continue
        try:
            hic_arr[i, j] = row.counts
            hic_arr[j, i] = row.counts
        except Exception as e:
            print(e)
            print(row.binX, row.binY, row.counts, i, j)

    if np.max(hic_arr) == 0:
        print(f'{odir} had no reads')
        return

    os.makedirs(odir, exist_ok = True)
    with open(osp.join(odir, 'import.log'), 'w') as f:
        if isinstance(chrom, str):
            chrom = chrom.strip('chr')
        f.write(f'{hic_filename}\nchrom={chrom}\nstart={start}\nend={end}\n')
        f.write(f'resolution={resolution}\nbeads={m}\nnorm={norm}\n')
        f.write(f'genome={hic.getGenomeID()}')

    np.save(osp.join(odir, 'hic.npy'), hic_arr)
    print(f'{odir} done')

def entire_chromosomes(hic_files, dataset, resolution=50000, ref_genome='hg19',
                    chroms=range(1,23), jobs=15, plot=False):
    data_folder = osp.join(ROOT, dataset)
    os.makedirs(data_folder, exist_ok = True)

    mapping = []
    for i, filename in enumerate(hic_files):
        cell_line = filename.split(os.sep)[-3]
        odir = osp.join(ROOT, dataset, f'chroms_{cell_line}')
        os.makedirs(odir, exist_ok = True)

        chromsizes = bioframe.fetch_chromsizes(ref_genome)
        for i, chromosome in enumerate(chroms):
            i += 1 # switch to 1-based indexing
            start = 0
            end = chromsizes[f'chr{chromosome}']
            print(f'i={i}: chr{chromosome} {start}-{end}')
            sample_folder = osp.join(odir, f'chr{chromosome}')
            mapping.append((sample_folder, filename, chromosome, start,
                            end, resolution))

    with multiprocessing.Pool(jobs) as p:
        p.starmap(import_contactmap_straw, mapping)

    if plot:
        # plotting is slow and not parallel-ized
        for i, filename in enumerate(hic_files):
            cell_line = filename.split(os.sep)[-3]
            odir = osp.join(ROOT, dataset, f'chroms_{cell_line}')
            for chr in chroms:
                hic = np.load(osp.join(odir, f'chr{chr}', 'hic.npy'))
                plot_matrix(hic, osp.join(odir, f'chr{chr}', 'hic.png'), vmax='mean')

def split(in_dataset, out_dataset, m, chroms=range(1,23), start_index=1,
                        ref_genome='hg19', scale=None):
    '''
    Splits whole-chromosome contact maps from in_dataset

    Inputs:
        in_dataset: directory of whole-chromosome contact maps
                    (saved in in_dataset/chroms_<cell_line>)
        out_dataset: directory to save split contact maps
                    (saved to out_dataset/samples/sample<i>)
        m: number of beads in split contact maps
        chroms: list of chromosomes numbers
        start_index: id of first sample
        ref_genome: reference genome (to get chromosome sizes)
        scale (float or None): Re-scales first off-diagonal of contact map to
                    have value <scale? (None to not scale)
    '''
    data_dir = osp.join(ROOT, in_dataset)
    out_data_dir = osp.join(ROOT, out_dataset)
    os.makedirs(out_data_dir, exist_ok=True)
    samples_dir = osp.join(out_data_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    cell_lines = [f.split('_')[1] for f in os.listdir(data_dir) if f.startswith('chroms_')]

    i = start_index
    chromsizes = bioframe.fetch_chromsizes(ref_genome)
    for cell_line in cell_lines:
        print(cell_line)
        for chrom in chroms:
            chrom_dir = osp.join(data_dir, f'chroms_{cell_line}/chr{chrom}')
            hic_file = osp.join(chrom_dir, 'hic.npy')
            if hic_file.endswith('.txt'):
                hic = np.loadtxt(hic_file)
            elif hic_file.endswith('.npy'):
                hic = np.load(hic_file)
            size = len(hic)
            diag = hic.diagonal() == 0
            ind = np.arange(0, len(diag))

            import_log = load_import_log(chrom_dir)
            resolution = import_log['resolution']

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


                hic_i = hic[start:end,start:end]
                if scale is not None:
                    # rescale Hi-C map such that first off_diagional has mean value of 'scale'
                    hic_i = rescale_p_s_1(hic_i, scale)
                    np.fill_diagonal(hic_i, 1)
                np.save(osp.join(odir, 'hic.npy'), hic_i)
                plot_matrix(hic_i, osp.join(odir, 'hic.png'), vmax='mean')
                i += 1
                start = end
                end = start + m

def main():
    entire_chromosomes(ALL_FILES_in_situ, 'dataset_all_files_50k',
                    resolution=50_000, jobs=15)
    split('dataset_all_files_50k', 'dataset_all_files_50k_512', 512, scale=1e-1)

if __name__ == '__main__':
    main()
