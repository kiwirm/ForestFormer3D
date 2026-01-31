# Builds cached bins and info dicts from labeled PLYs (RGB+I).
import os
from concurrent import futures as futures
from os import path as osp

import mmengine
import numpy as np


class DatasetCacheBuilder(object):
    """Dataset cache builder for RGB+intensity point features."""

    def __init__(self, root_path, split='train', save_path=None, split_prefix=None):
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.split = split
        self.split_dir = osp.join(root_path)

        self.classes = ['tree']
        self.cat_ids = np.array([1])

        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids2class = {
            treeid: i
            for i, treeid in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        if split_prefix:
            split_dir = osp.join(self.root_dir, 'splits', split_prefix)
            split_name = f'{split_prefix}_{split}_list.txt'
        else:
            split_dir = osp.join(self.root_dir, 'splits', 'original')
            split_name = f'original_{split}_list.txt'
        split_file = osp.join(split_dir, split_name)
        mmengine.check_file_exist(split_file)
        self.sample_id_list = mmengine.list_from_file(split_file)
        self.test_mode = (split == 'test')

        self.instance_dir = osp.join(self.root_dir, 'derived', 'instance_data')
        self.points_out_dir = osp.join(self.save_path, 'processed', 'points')
        self.instance_mask_out_dir = osp.join(self.save_path, 'processed', 'instance_mask')
        self.semantic_mask_out_dir = osp.join(self.save_path, 'processed', 'semantic_mask')

    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = osp.join(self.instance_dir, f'{idx}_aligned_bbox.npy')
        mmengine.check_file_exist(box_file)
        return np.load(box_file)

    def get_unaligned_box_label(self, idx):
        box_file = osp.join(self.instance_dir, f'{idx}_unaligned_bbox.npy')
        mmengine.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.instance_dir, f'{idx}_axis_align_matrix.npy')
        mmengine.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 7, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            pts_filename = osp.join(self.instance_dir, f'{sample_idx}_vert.npy')
            points = np.load(pts_filename)
            mmengine.mkdir_or_exist(self.points_out_dir)
            points.tofile(osp.join(self.points_out_dir, f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('processed', 'points', f'{sample_idx}.bin')

            pts_instance_mask_path = osp.join(self.instance_dir, f'{sample_idx}_ins_label.npy')
            pts_semantic_mask_path = osp.join(self.instance_dir, f'{sample_idx}_sem_label.npy')

            pts_instance_mask = np.load(pts_instance_mask_path).astype(np.int64)
            pts_semantic_mask = np.load(pts_semantic_mask_path).astype(np.int64)

            mmengine.mkdir_or_exist(self.instance_mask_out_dir)
            mmengine.mkdir_or_exist(self.semantic_mask_out_dir)

            pts_instance_mask.tofile(
                osp.join(self.instance_mask_out_dir, f'{sample_idx}.bin'))
            pts_semantic_mask.tofile(
                osp.join(self.semantic_mask_out_dir, f'{sample_idx}.bin'))

            info['pts_instance_mask_path'] = osp.join('processed', 'instance_mask', f'{sample_idx}.bin')
            info['pts_semantic_mask_path'] = osp.join('processed', 'semantic_mask', f'{sample_idx}.bin')

            if has_label:
                annotations = {}
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                unaligned_box_label = self.get_unaligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]
                    unaligned_box = unaligned_box_label[:, :-1]
                    classes = aligned_box_label[:, -1]
                    annotations['name'] = np.array([
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(annotations['gt_num'])
                    ])
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['unaligned_location'] = unaligned_box[:, :3]
                    annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
                    annotations['unaligned_gt_boxes_upright_depth'] = unaligned_box
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                annotations['axis_align_matrix'] = axis_align_matrix
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
