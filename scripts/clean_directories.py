import argparse
import os.path as osp
import sys
from shutil import rmtree

from pylib.utils.ArgparseConverter import ArgparseConverter


def clean_directories(data_folder=None, GNN_path = None,
                    GNN_file_name = None, ofile = sys.stdout):
    AC = ArgparseConverter()
    parser = argparse.ArgumentParser(description='Simple parser', allow_abbrev=False)
    parser.add_argument('--data_folder', type=AC.str2list, default=data_folder,
                        help='Location of data')
    parser.add_argument('--GNN_file_name', type=str, default=str(GNN_file_name),
                        help='name of file graph data was saved to')
    parser.add_argument('--GNN_path', type=str, default=GNN_path,
                        help='path to graph data')
    parser.add_argument('--clean_scratch', action='store_true',
                        help='True clean scratch')
    parser.add_argument('--scratch', type=str, default=None)
    parser.add_argument('--move_data_to_scratch', type=AC.str2bool, default=False)
    opt, _ = parser.parse_known_args()

    if isinstance(opt.data_folder, list):
        opt.data_folder = opt.data_folder[0]

    if opt.move_data_to_scratch and opt.scratch is not None:
        opt.data_folder = osp.join(opt.scratch, osp.split(opt.data_folder)[-1])
        if opt.clean_scratch:
            rmtree(opt.data_folder)

    if opt.GNN_path is None and opt.GNN_file_name is not None:
        opt.GNN_path = osp.join(opt.scratch, opt.GNN_file_name)

    if osp.exists(opt.GNN_path):
        print(f'clean_directories.py: removing {opt.GNN_path}', file = ofile)
        rmtree(opt.GNN_path)
    else:
        print(f'clean_directories.py: {opt.GNN_path} does not exist - cannot remove', file = ofile)


if __name__ == '__main__':
    clean_directories()
