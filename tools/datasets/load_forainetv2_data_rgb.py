# Modified from load_forainetv2_data.py to include RGB + intensity features.
import argparse
import inspect
import json
import os

import numpy as np
from tools.plyutils import read_ply

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i][
                'objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def extract_bbox(mesh_vertices, label_ids, instance_ids, bg_sem=np.array([0])):
    valid_mask = ~np.isin(label_ids, bg_sem)
    mesh_vertices = mesh_vertices[valid_mask]
    instance_ids = instance_ids[valid_mask]
    label_ids = label_ids[valid_mask]

    unique_instance_ids = np.unique(instance_ids)
    num_instances = len(unique_instance_ids)
    instance_bboxes = np.zeros((num_instances, 7))

    for i, instance_id in enumerate(unique_instance_ids):
        mask = instance_ids == instance_id
        pts = mesh_vertices[mask, :3]
        if pts.shape[0] == 0:
            continue
        min_pts = pts.min(axis=0)
        max_pts = pts.max(axis=0)
        locations = (min_pts + max_pts) / 2
        dimensions = max_pts - min_pts
        instance_bboxes[i, :3] = locations
        instance_bboxes[i, 3:6] = dimensions
        instance_bboxes[i, 6] = 1

    return instance_bboxes


def export(ply_file,
           output_file=None,
           test_mode=False):
    """Export original files to vert, ins_label, sem_label and bbox file.

    This RGB/intensity variant writes 7D points: x,y,z,r,g,b,intensity.
    """
    pcd = read_ply(ply_file)

    points_xyz = np.vstack((pcd['x'], pcd['y'], pcd['z'])).astype(np.float64).T
    is_blue = 'bluepoints' in os.path.basename(ply_file)

    if is_blue:
        offsets = np.zeros(3, dtype=np.float64)
    else:
        mean_x = np.mean(points_xyz[:, 0])
        mean_y = np.mean(points_xyz[:, 1])
        min_z = np.min(points_xyz[:, 2])
        offsets = np.array([mean_x, mean_y, min_z], dtype=np.float64)

        points_xyz[:, 0] -= mean_x
        points_xyz[:, 1] -= mean_y
        points_xyz[:, 2] -= min_z

    points_xyz = points_xyz.astype(np.float32)

    # Optional attributes: default to zeros if missing
    if 'red' in pcd.dtype.names and 'green' in pcd.dtype.names and 'blue' in pcd.dtype.names:
        rgb = np.vstack((pcd['red'], pcd['green'], pcd['blue'])).astype(np.float32).T
    else:
        rgb = np.zeros((points_xyz.shape[0], 3), dtype=np.float32)

    if 'intensity' in pcd.dtype.names:
        intensity = pcd['intensity'].astype(np.float32).reshape(-1, 1)
    else:
        intensity = np.zeros((points_xyz.shape[0], 1), dtype=np.float32)

    points = np.hstack((points_xyz, rgb, intensity)).astype(np.float32)

    semantic_seg = pcd["semantic_seg"].astype(np.int64)
    treeID = pcd["treeID"].astype(np.int64)

    axis_align_matrix = np.eye(4)
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    pts = np.ones((points.shape[0], 4))
    pts[:, 0:3] = points[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())
    aligned_mesh_vertices = pts[:, 0:3]

    if not test_mode:
        bg_sem = np.array([0])
        label_ids = semantic_seg - 1
        instance_ids = treeID
        instance_ids[np.isin(label_ids, bg_sem)] = -1

        valid_mask = instance_ids != -1
        new_instance_ids = np.zeros_like(instance_ids)
        new_instance_ids[valid_mask] = instance_ids[valid_mask]
        new_instance_ids[instance_ids == -1] = 0
        instance_ids = new_instance_ids

        unaligned_bboxes = extract_bbox(points, label_ids, instance_ids, bg_sem)
        aligned_bboxes = extract_bbox(aligned_mesh_vertices, label_ids, instance_ids, bg_sem)
    else:
        label_ids = None
        instance_ids = None
        unaligned_bboxes = None
        aligned_bboxes = None

    if output_file is not None:
        np.save(output_file + '_vert.npy', points)
        if not test_mode:
            np.save(output_file + '_sem_label.npy', label_ids)
            np.save(output_file + '_ins_label.npy', instance_ids)
            np.save(output_file + '_unaligned_bbox.npy', unaligned_bboxes)
            np.save(output_file + '_aligned_bbox.npy', aligned_bboxes)
            np.save(output_file + '_axis_align_matrix.npy', axis_align_matrix)

    return points, label_ids, instance_ids, unaligned_bboxes, \
        aligned_bboxes, axis_align_matrix, offsets
