# This is an exact copy of tools/train.py from open-mmlab/mmdetection3d.
import argparse
import logging
import os
import os.path as osp
import types

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import RUNNERS
from mmengine.runner import Runner, find_latest_checkpoint
import torch

from mmdet3d.utils import replace_ceph_backend

#import sys
#print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
#print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
#print("Current working directory:", os.getcwd())
#print("sys.path:", sys.path)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    def _copy_checkpoint_to_model(model, checkpoint_state):
        model_state = model.state_dict()
        loaded = 0
        permuted = 0
        missing = 0
        skipped = 0
        perms = (
            (0, 2, 3, 4, 1),  # (out,in,k,k,k) -> (out,k,k,k,in)
            (1, 2, 3, 4, 0),  # (in,out,k,k,k) -> (out,k,k,k,in)
            (4, 0, 1, 2, 3),  # (k,k,k,in,out) -> (out,k,k,k,in)
        )

        for key, dst in model_state.items():
            src = checkpoint_state.get(key)
            if src is None:
                src = checkpoint_state.get(f'module.{key}')
            if src is None:
                missing += 1
                continue
            if not torch.is_tensor(src):
                skipped += 1
                continue
            if src.shape == dst.shape:
                dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
                loaded += 1
                continue
            if src.ndim == 5 and dst.ndim == 5:
                matched = False
                for perm in perms:
                    if src.permute(perm).shape == dst.shape:
                        dst.copy_(src.permute(perm).contiguous().to(device=dst.device, dtype=dst.dtype))
                        permuted += 1
                        matched = True
                        break
                if matched:
                    continue
            skipped += 1
        return loaded, permuted, missing, skipped

    def _load_or_resume_with_manual_spconv(self):
        if self._has_loaded:
            return None

        resume_from = None
        if self._resume and self._load_from is None:
            resume_from = find_latest_checkpoint(self.work_dir)
            self.logger.info(f'Auto resumed from the latest checkpoint {resume_from}.')
        elif self._resume and self._load_from is not None:
            resume_from = self._load_from

        ckpt_path = resume_from if resume_from is not None else self._load_from
        if ckpt_path is None:
            return None

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        model = self.model.module if is_model_wrapper(self.model) else self.model

        loaded, permuted, missing, skipped = _copy_checkpoint_to_model(model, state)
        self.logger.info(
            f'Manual checkpoint load: loaded={loaded}, permuted={permuted}, '
            f'missing={missing}, skipped={skipped}.')

        if resume_from is not None:
            meta = checkpoint.get('meta', {})
            self.train_loop._epoch = meta.get('epoch', 0)
            self.train_loop._iter = meta.get('iter', 0)

            if 'message_hub' in checkpoint:
                self.message_hub.load_state_dict(checkpoint['message_hub'])
            if 'optimizer' in checkpoint:
                self.optim_wrapper.load_state_dict(checkpoint['optimizer'])

            if self.param_schedulers is not None and 'param_schedulers' in checkpoint:
                if isinstance(self.param_schedulers, dict):
                    for name, schedulers in self.param_schedulers.items():
                        for scheduler, ckpt_scheduler in zip(
                                schedulers, checkpoint['param_schedulers'][name]):
                            scheduler.load_state_dict(ckpt_scheduler)
                else:
                    for scheduler, ckpt_scheduler in zip(
                            self.param_schedulers, checkpoint['param_schedulers']):
                        scheduler.load_state_dict(ckpt_scheduler)

            self.logger.info(f'resumed epoch: {self.epoch}, iter: {self.iter}')

        self._has_loaded = True

    runner.load_or_resume = types.MethodType(_load_or_resume_with_manual_spconv, runner)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
