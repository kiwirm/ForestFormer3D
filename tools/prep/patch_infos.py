# Modified from mmdetection3d/tools/dataset_converters/patch_infos.py
"""Convert the annotation pkl to the standard format in OpenMMLab V2.0.

Example:
    python tools/prep/patch_infos.py
        --dataset kitti
        --pkl-path ./data/kitti/kitti_infos_train.pkl
        --out-dir ./kitti_v2/
"""

import argparse
import time
from os import path as osp
from pathlib import Path

import mmengine

def get_empty_instance():
    """Empty annotation for single instance."""
    instance = dict(
        # (list[float], required): list of 4 numbers representing
        # the bounding box of the instance, in (x1, y1, x2, y2) order.
        bbox=None,
        # (int, required): an integer in the range
        # [0, num_categories-1] representing the category label.
        bbox_label=None,
        #  (list[float], optional): list of 7 (or 9) numbers representing
        #  the 3D bounding box of the instance,
        #  in [x, y, z, w, h, l, yaw]
        #  (or [x, y, z, w, h, l, yaw, vx, vy]) order.
        bbox_3d=None,
        # (bool, optional): Whether to use the
        # 3D bounding box during training.
        bbox_3d_isvalid=None,
        # (int, optional): 3D category label
        # (typically the same as label).
        bbox_label_3d=None,
        # (float, optional): Projected center depth of the
        # 3D bounding box compared to the image plane.
        depth=None,
        #  (list[float], optional): Projected
        #  2D center of the 3D bounding box.
        center_2d=None,
        # (int, optional): Attribute labels
        # (fine-grained labels such as stopping, moving, ignore, crowd).
        attr_label=None,
        # (int, optional): The number of LiDAR
        # points in the 3D bounding box.
        num_lidar_pts=None,
        # (int, optional): The number of Radar
        # points in the 3D bounding box.
        num_radar_pts=None,
        # (int, optional): Difficulty level of
        # detecting the 3D bounding box.
        difficulty=None,
        unaligned_bbox_3d=None)
    return instance

def get_empty_lidar_points():
    lidar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of LiDAR data file.
        lidar_path=None,
        # (list[list[float]], optional): Transformation matrix
        # from lidar to ego-vehicle
        # with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        lidar2ego=None,
    )
    return lidar_points


def get_empty_radar_points():
    radar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of RADAR data file.
        radar_path=None,
        # Transformation matrix from lidar to
        # ego-vehicle with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        radar2ego=None,
    )
    return radar_points

def get_empty_img_info():
    img_info = dict(
        # (str, required): the path to the image file.
        img_path=None,
        # (int) The height of the image.
        height=None,
        # (int) The width of the image.
        width=None,
        # (str, optional): Path of the depth map file
        depth_map=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to image with
        # shape [3, 3], [3, 4] or [4, 4].
        cam2img=None,
        # (list[list[float]]): Transformation matrix from lidar
        # or depth to image with shape [4, 4].
        lidar2img=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to ego-vehicle
        # with shape [4, 4].
        cam2ego=None)
    return img_info

def get_single_image_sweep(camera_types):
    single_image_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None)
    # (dict): Information of images captured by multiple cameras
    images = dict()
    for cam_type in camera_types:
        images[cam_type] = get_empty_img_info()
    single_image_sweep['images'] = images
    return single_image_sweep

def get_empty_standard_data_info(
        camera_types=['CAM0', 'CAM1', 'CAM2', 'CAM3', 'CAM4']):

    data_info = dict(
        # (str): Sample id of the frame.
        sample_idx=None,
        # (str, optional): '000010'
        token=None,
        **get_single_image_sweep(camera_types),
        # (dict, optional): dict contains information
        # of LiDAR point cloud frame.
        lidar_points=get_empty_lidar_points(),
        # (dict, optional) Each dict contains
        # information of Radar point cloud frame.
        radar_points=get_empty_radar_points(),
        # (list[dict], optional): Image sweeps data.
        image_sweeps=[],
        lidar_sweeps=[],
        instances=[],
        # (list[dict], optional): Required by object
        # detection, instance  to be ignored during training.
        instances_ignore=[],
        # (str, optional): Path of semantic labels for each point.
        pts_semantic_mask_path=None,
        # (str, optional): Path of instance labels for each point.
        pts_instance_mask_path=None)
    return data_info


def clear_instance_unused_keys(instance):
    keys = list(instance.keys())
    for k in keys:
        if instance[k] is None:
            del instance[k]
    return instance


def clear_data_info_unused_keys(data_info):
    keys = list(data_info.keys())
    empty_flag = True
    for key in keys:
        # we allow no annotations in datainfo
        if key in ['instances', 'cam_sync_instances', 'cam_instances']:
            empty_flag = False
            continue
        if isinstance(data_info[key], list):
            if len(data_info[key]) == 0:
                del data_info[key]
            else:
                empty_flag = False
        elif data_info[key] is None:
            del data_info[key]
        elif isinstance(data_info[key], dict):
            _, sub_empty_flag = clear_data_info_unused_keys(data_info[key])
            if sub_empty_flag is False:
                empty_flag = False
            else:
                # sub field is empty
                del data_info[key]
        else:
            empty_flag = False

    return data_info, empty_flag

def update_scannet_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {
        'classes':
        ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
         'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin')
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    if isinstance(data_list, dict) and 'data_list' in data_list:
        data_list = data_list['data_list']
    print('Start updating:')
    converted_list = []
    ignore_class_name = set()
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        if 'point_cloud' in ori_info_dict:
            temp_data_info = get_empty_standard_data_info()
            temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
                'point_cloud']['num_features']
            lidar_path = ori_info_dict['pts_path']
            if isinstance(lidar_path, str):
                if '/' not in lidar_path and '\\\\' not in lidar_path:
                    lidar_path = f'processed/points/{lidar_path}'
                elif lidar_path.startswith('points/'):
                    lidar_path = f'processed/{lidar_path}'
            temp_data_info['lidar_points']['lidar_path'] = lidar_path
            if 'pts_semantic_mask_path' in ori_info_dict:
                sem_path = ori_info_dict['pts_semantic_mask_path']
                if isinstance(sem_path, str):
                    if '/' not in sem_path and '\\\\' not in sem_path:
                        sem_path = f'processed/semantic_mask/{sem_path}'
                    elif sem_path.startswith('semantic_mask/'):
                        sem_path = f'processed/{sem_path}'
                temp_data_info['pts_semantic_mask_path'] = sem_path
            if 'pts_instance_mask_path' in ori_info_dict:
                ins_path = ori_info_dict['pts_instance_mask_path']
                if isinstance(ins_path, str):
                    if '/' not in ins_path and '\\\\' not in ins_path:
                        ins_path = f'processed/instance_mask/{ins_path}'
                    elif ins_path.startswith('instance_mask/'):
                        ins_path = f'processed/{ins_path}'
                temp_data_info['pts_instance_mask_path'] = ins_path
        else:
            # Already v2 format; just normalize relative paths
            temp_data_info = ori_info_dict
            lidar = temp_data_info.get('lidar_points', {})
            lidar_path = lidar.get('lidar_path')
            if isinstance(lidar_path, str):
                if '/' not in lidar_path and '\\\\' not in lidar_path:
                    lidar['lidar_path'] = f'processed/points/{lidar_path}'
                elif lidar_path.startswith('points/'):
                    lidar['lidar_path'] = f'processed/{lidar_path}'
            if 'pts_semantic_mask_path' in temp_data_info:
                sem_path = temp_data_info['pts_semantic_mask_path']
                if isinstance(sem_path, str):
                    if '/' not in sem_path and '\\\\' not in sem_path:
                        temp_data_info['pts_semantic_mask_path'] = f'processed/semantic_mask/{sem_path}'
                    elif sem_path.startswith('semantic_mask/'):
                        temp_data_info['pts_semantic_mask_path'] = f'processed/{sem_path}'
            if 'pts_instance_mask_path' in temp_data_info:
                ins_path = temp_data_info['pts_instance_mask_path']
                if isinstance(ins_path, str):
                    if '/' not in ins_path and '\\\\' not in ins_path:
                        temp_data_info['pts_instance_mask_path'] = f'processed/instance_mask/{ins_path}'
                    elif ins_path.startswith('instance_mask/'):
                        temp_data_info['pts_instance_mask_path'] = f'processed/{ins_path}'
        if 'super_pts_path' in ori_info_dict:
            temp_data_info['super_pts_path'] = Path(
                ori_info_dict['super_pts_path']).name

        # TODO support camera
        # np.linalg.inv(info['axis_align_matrix'] @ extrinsic): depth2cam
        anns = ori_info_dict.get('annos', None)
        if anns is not None:
            temp_data_info['axis_align_matrix'] = anns[
                'axis_align_matrix'].tolist()
            if anns['gt_num'] == 0:
                instance_list = []
            else:
                num_instances = len(anns['name'])
                instance_list = []
                for instance_id in range(num_instances):
                    empty_instance = get_empty_instance()
                    empty_instance['bbox_3d'] = anns['gt_boxes_upright_depth'][
                        instance_id].tolist()

                    if anns['name'][instance_id] in METAINFO['classes']:
                        empty_instance['bbox_label_3d'] = METAINFO[
                            'classes'].index(anns['name'][instance_id])
                    else:
                        ignore_class_name.add(anns['name'][instance_id])
                        empty_instance['bbox_label_3d'] = -1

                    empty_instance = clear_instance_unused_keys(empty_instance)
                    instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'scannet'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')

def update_scannet200_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {
        'classes':
        ('chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk',
         'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
         'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair',
         'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet',
         'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool',
         'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table',
         'keyboard', 'bag', 'backpack', 'toilet paper', 'printer', 'tv stand',
         'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet',
         'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle',
         'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror',
         'copier', 'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop',
         'shower', 'paper', 'person', 'paper towel dispenser', 'oven',
         'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail',
         'radiator', 'recycling bin', 'container', 'wardrobe',
         'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light',
         'laundry basket', 'pipe', 'clothes dryer', 'guitar',
         'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle',
         'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket',
         'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll',
         'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board',
         'ironing board', 'fireplace', 'soap dish', 'kitchen counter',
         'doorframe', 'toilet paper dispenser', 'mini fridge',
         'fire extinguisher', 'ball', 'hat', 'shower curtain rod',
         'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar',
         'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
         'furniture', 'cart', 'storage container', 'scale', 'tissue box',
         'light switch', 'crate', 'power outlet', 'decoration', 'sign',
         'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger',
         'stuffed animal', 'headphones', 'dish rack', 'broom', 'guitar case',
         'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar',
         'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl',
         'paper bag', 'alarm clock', 'music stand', 'projector screen',
         'divider', 'laundry detergent', 'bathroom counter', 'object',
         'bathroom vanity', 'closet wall', 'laundry hamper',
         'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell',
         'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
         'coffee kettle', 'structure', 'shower head', 'keyboard piano',
         'case of water bottles', 'coat rack', 'storage organizer',
         'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster',
         'potted plant', 'luggage', 'mattress')
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    if isinstance(data_list, dict) and 'data_list' in data_list:
        data_list = data_list['data_list']
    print('Start updating:')
    converted_list = []
    ignore_class_name = set()
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        if 'point_cloud' in ori_info_dict:
            temp_data_info = get_empty_standard_data_info()
            temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
                'point_cloud']['num_features']
            lidar_path = ori_info_dict['pts_path']
            if isinstance(lidar_path, str):
                if '/' not in lidar_path and '\\\\' not in lidar_path:
                    lidar_path = f'processed/points/{lidar_path}'
                elif lidar_path.startswith('points/'):
                    lidar_path = f'processed/{lidar_path}'
            temp_data_info['lidar_points']['lidar_path'] = lidar_path
            if 'pts_semantic_mask_path' in ori_info_dict:
                sem_path = ori_info_dict['pts_semantic_mask_path']
                if isinstance(sem_path, str):
                    if '/' not in sem_path and '\\\\' not in sem_path:
                        sem_path = f'processed/semantic_mask/{sem_path}'
                    elif sem_path.startswith('semantic_mask/'):
                        sem_path = f'processed/{sem_path}'
                temp_data_info['pts_semantic_mask_path'] = sem_path
            if 'pts_instance_mask_path' in ori_info_dict:
                ins_path = ori_info_dict['pts_instance_mask_path']
                if isinstance(ins_path, str):
                    if '/' not in ins_path and '\\\\' not in ins_path:
                        ins_path = f'processed/instance_mask/{ins_path}'
                    elif ins_path.startswith('instance_mask/'):
                        ins_path = f'processed/{ins_path}'
                temp_data_info['pts_instance_mask_path'] = ins_path
        else:
            temp_data_info = ori_info_dict
            lidar = temp_data_info.get('lidar_points', {})
            lidar_path = lidar.get('lidar_path')
            if isinstance(lidar_path, str):
                if '/' not in lidar_path and '\\\\' not in lidar_path:
                    lidar['lidar_path'] = f'processed/points/{lidar_path}'
                elif lidar_path.startswith('points/'):
                    lidar['lidar_path'] = f'processed/{lidar_path}'
            if 'pts_semantic_mask_path' in temp_data_info:
                sem_path = temp_data_info['pts_semantic_mask_path']
                if isinstance(sem_path, str):
                    if '/' not in sem_path and '\\\\' not in sem_path:
                        temp_data_info['pts_semantic_mask_path'] = f'processed/semantic_mask/{sem_path}'
                    elif sem_path.startswith('semantic_mask/'):
                        temp_data_info['pts_semantic_mask_path'] = f'processed/{sem_path}'
            if 'pts_instance_mask_path' in temp_data_info:
                ins_path = temp_data_info['pts_instance_mask_path']
                if isinstance(ins_path, str):
                    if '/' not in ins_path and '\\\\' not in ins_path:
                        temp_data_info['pts_instance_mask_path'] = f'processed/instance_mask/{ins_path}'
                    elif ins_path.startswith('instance_mask/'):
                        temp_data_info['pts_instance_mask_path'] = f'processed/{ins_path}'
        if 'super_pts_path' in ori_info_dict:
            temp_data_info['super_pts_path'] = Path(
                ori_info_dict['super_pts_path']).name

        # TODO support camera
        # np.linalg.inv(info['axis_align_matrix'] @ extrinsic): depth2cam
        anns = ori_info_dict.get('annos', None)
        if anns is not None:
            temp_data_info['axis_align_matrix'] = anns[
                'axis_align_matrix'].tolist()
            if anns['gt_num'] == 0:
                instance_list = []
            else:
                num_instances = len(anns['name'])
                instance_list = []
                for instance_id in range(num_instances):
                    empty_instance = get_empty_instance()
                    empty_instance['bbox_3d'] = anns['gt_boxes_upright_depth'][
                        instance_id].tolist()

                    if anns['name'][instance_id] in METAINFO['classes']:
                        empty_instance['bbox_label_3d'] = METAINFO[
                            'classes'].index(anns['name'][instance_id])
                    else:
                        ignore_class_name.add(anns['name'][instance_id])
                        empty_instance['bbox_label_3d'] = -1

                    empty_instance = clear_instance_unused_keys(empty_instance)
                    instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'scannet200'
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')

def update_dataset_infos(pkl_path, out_dir, dataset):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {
        'classes':
        ('tree')
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    if isinstance(data_list, dict) and 'data_list' in data_list:
        data_list = data_list['data_list']
    print('Start updating:')
    converted_list = []
    ignore_class_name = set()
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        if 'point_cloud' in ori_info_dict:
            temp_data_info = get_empty_standard_data_info()
            temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
                'point_cloud']['num_features']
            lidar_path = ori_info_dict['pts_path']
            if isinstance(lidar_path, str):
                if '/' not in lidar_path and '\\\\' not in lidar_path:
                    lidar_path = f'processed/points/{lidar_path}'
                elif lidar_path.startswith('points/'):
                    lidar_path = f'processed/{lidar_path}'
            temp_data_info['lidar_points']['lidar_path'] = lidar_path
            if 'pts_semantic_mask_path' in ori_info_dict:
                sem_path = ori_info_dict['pts_semantic_mask_path']
                if isinstance(sem_path, str):
                    if '/' not in sem_path and '\\\\' not in sem_path:
                        sem_path = f'processed/semantic_mask/{sem_path}'
                    elif sem_path.startswith('semantic_mask/'):
                        sem_path = f'processed/{sem_path}'
                temp_data_info['pts_semantic_mask_path'] = sem_path
            if 'pts_instance_mask_path' in ori_info_dict:
                ins_path = ori_info_dict['pts_instance_mask_path']
                if isinstance(ins_path, str):
                    if '/' not in ins_path and '\\\\' not in ins_path:
                        ins_path = f'processed/instance_mask/{ins_path}'
                    elif ins_path.startswith('instance_mask/'):
                        ins_path = f'processed/{ins_path}'
                temp_data_info['pts_instance_mask_path'] = ins_path
        else:
            # Already v2 format; normalize relative paths only.
            temp_data_info = ori_info_dict
            lidar = temp_data_info.get('lidar_points', {})
            lidar_path = lidar.get('lidar_path')
            if isinstance(lidar_path, str):
                if '/' not in lidar_path and '\\\\' not in lidar_path:
                    lidar['lidar_path'] = f'processed/points/{lidar_path}'
                elif lidar_path.startswith('points/'):
                    lidar['lidar_path'] = f'processed/{lidar_path}'
            if 'pts_semantic_mask_path' in temp_data_info:
                sem_path = temp_data_info['pts_semantic_mask_path']
                if isinstance(sem_path, str):
                    if '/' not in sem_path and '\\\\' not in sem_path:
                        temp_data_info['pts_semantic_mask_path'] = f'processed/semantic_mask/{sem_path}'
                    elif sem_path.startswith('semantic_mask/'):
                        temp_data_info['pts_semantic_mask_path'] = f'processed/{sem_path}'
            if 'pts_instance_mask_path' in temp_data_info:
                ins_path = temp_data_info['pts_instance_mask_path']
                if isinstance(ins_path, str):
                    if '/' not in ins_path and '\\\\' not in ins_path:
                        temp_data_info['pts_instance_mask_path'] = f'processed/instance_mask/{ins_path}'
                    elif ins_path.startswith('instance_mask/'):
                        temp_data_info['pts_instance_mask_path'] = f'processed/{ins_path}'

        # TODO support camera
        # np.linalg.inv(info['axis_align_matrix'] @ extrinsic): depth2cam
        anns = ori_info_dict.get('annos', None)
        if anns is not None:
            temp_data_info['axis_align_matrix'] = anns[
                'axis_align_matrix'].tolist()
            if anns['gt_num'] == 0:
                instance_list = []
            else:
                num_instances = len(anns['name'])
                instance_list = []
                for instance_id in range(num_instances):
                    empty_instance = get_empty_instance()
                    empty_instance['bbox_3d'] = anns['gt_boxes_upright_depth'][
                        instance_id].tolist()

                    if anns['name'][instance_id] in METAINFO['classes']:
                        empty_instance['bbox_label_3d'] = METAINFO[
                            'classes'].index(anns['name'][instance_id])
                    else:
                        ignore_class_name.add(anns['name'][instance_id])
                        empty_instance['bbox_label_3d'] = -1

                    empty_instance = clear_instance_unused_keys(empty_instance)
                    instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = dataset
    metainfo['info_version'] = '1.1'

    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    mmengine.dump(converted_data_info, out_path, 'pkl')

def parse_args():
    parser = argparse.ArgumentParser(description='Arg parser for data coords '
                                     'update due to coords sys refactor.')
    parser.add_argument(
        '--dataset', type=str, default='kitti', help='name of dataset')
    parser.add_argument(
        '--pkl-path',
        type=str,
        default='./data/kitti/kitti_infos_train.pkl ',
        help='specify the root dir of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='converted_annotations',
        required=False,
        help='output direction of info pkl')
    args = parser.parse_args()
    return args


def update_pkl_infos(dataset, out_dir, pkl_path):
    if dataset.lower() == 'scannet':
        update_scannet_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'scannet200':
        update_scannet200_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() in {
        'original',
        'cass',
        'combined',
    }:
        update_dataset_infos(pkl_path=pkl_path, out_dir=out_dir, dataset=dataset)
    else:
        raise NotImplementedError(f'Do not support convert {dataset} to v2.')


if __name__ == '__main__':
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.root_dir
    update_pkl_infos(
        dataset=args.dataset, out_dir=args.out_dir, pkl_path=args.pkl_path)
