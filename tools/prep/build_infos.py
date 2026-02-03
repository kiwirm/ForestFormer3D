# Data prep entrypoint for RGB/intensity variant (non-destructive).
import argparse
import os
from os import path as osp

from tools.prep.build_info_index import create_info_file
from tools.prep.patch_infos import update_pkl_infos


def dataset_data_prep(root_path, info_prefix, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    create_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_val.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_test.pkl')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_test_path)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='original|cass|scene|combined', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/derived/infos',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default=None)
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()
if args.extra_tag is None:
    args.extra_tag = args.dataset

if __name__ == '__main__':
    try:
        from mmdet3d.utils import register_all_modules
        register_all_modules()
    except Exception:
        print("Warning: mmdet3d not available; skipping register_all_modules().")

    if args.dataset in ('original', 'cass', 'scene', 'combined'):
        dataset_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
