# Modified from converter_forainetv2.py to use RGB/intensity data utils.
import os

import mmengine

from forainetv2_data_utils_rgb import ForAINetV2DataRGB


def create_info_file(data_path,
                        pkl_prefix='forainetv2_rgb',
                        save_path=None,
                        workers=4):
    assert os.path.exists(data_path)
    split_prefix_map = {
        'forainetv2_rgb': None,
        'forainetv2_rgb_cass': 'cass',
        'forainetv2_rgb_combined': 'combined',
    }
    assert pkl_prefix in split_prefix_map, \
        f'unsupported dataset {pkl_prefix}'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    train_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_train.pkl')
    val_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_val.pkl')
    test_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_test.pkl')

    split_prefix = split_prefix_map[pkl_prefix]
    train_dataset = ForAINetV2DataRGB(root_path=data_path, split='train', split_prefix=split_prefix)
    val_dataset = ForAINetV2DataRGB(root_path=data_path, split='val', split_prefix=split_prefix)
    test_dataset = ForAINetV2DataRGB(root_path=data_path, split='test', split_prefix=split_prefix)

    infos_train = train_dataset.get_infos(
        num_workers=workers, has_label=True)
    mmengine.dump(infos_train, train_filename, 'pkl')
    print(f'{pkl_prefix} info train file is saved to {train_filename}')

    infos_val = val_dataset.get_infos(
        num_workers=workers, has_label=True)
    mmengine.dump(infos_val, val_filename, 'pkl')
    print(f'{pkl_prefix} info val file is saved to {val_filename}')

    infos_test = test_dataset.get_infos(
        num_workers=workers, has_label=True)
    mmengine.dump(infos_test, test_filename, 'pkl')
    print(f'{pkl_prefix} info test file is saved to {test_filename}')
