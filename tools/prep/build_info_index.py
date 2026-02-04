# Builds dataset info PKLs from cached arrays.
import os

import mmengine

from tools.prep.cache_builder import DatasetCacheBuilder


def create_info_file(data_path,
                        pkl_prefix='original',
                        save_path=None,
                        workers=4):
    assert os.path.exists(data_path)
    split_prefix_map = {
        'original': 'original',
        'cass': 'cass',
        'scene': 'scene',
        'combined': 'combined',
    }
    assert pkl_prefix in split_prefix_map, \
        f'unsupported dataset {pkl_prefix}'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    train_filename = os.path.join(save_path, 'train.pkl')
    test_filename = os.path.join(save_path, 'test.pkl')

    split_prefix = split_prefix_map[pkl_prefix]
    train_dataset = DatasetCacheBuilder(root_path=data_path, split='train', split_prefix=split_prefix)
    test_dataset = DatasetCacheBuilder(root_path=data_path, split='test', split_prefix=split_prefix)

    infos_train = train_dataset.get_infos(
        num_workers=workers, has_label=True)
    mmengine.dump(infos_train, train_filename, 'pkl')
    print(f'{pkl_prefix} info train file is saved to {train_filename}')

    infos_test = test_dataset.get_infos(
        num_workers=workers, has_label=True)
    mmengine.dump(infos_test, test_filename, 'pkl')
    print(f'{pkl_prefix} info test file is saved to {test_filename}')

    return [train_filename, test_filename]
