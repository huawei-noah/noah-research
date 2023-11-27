# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Modified from: https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0

import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger)

import mmseg_custom


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--data-root', type=str, help='the dir to datasets')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--pretrained_from', default='model/pretrained_model.pth',
        help='the checkpoint file to load pretrained weights from')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
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
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--train-iter', type=int, default=80000)
    parser.add_argument('--checkpoint-save-freq', type=int, default=2000)
    parser.add_argument('--log-freq', type=int, default=1)
    parser.add_argument('--eval-freq', type=int, default=2000)

    parser.add_argument('--lr', type=float, default=6e-06)
    parser.add_argument('--prompt-lr-mult', default=5, type=float)

    # hyperparameters
    parser.add_argument('--freeze-backbone', action='store_true', help="whether freeze backbone parameters")

    parser.add_argument('--curve-update-interval', type=int, default=200, help="the interval to update IoU curve")
    parser.add_argument('--IoU-queue-length', type=int, default=1000, help="the max imgs for the queue to store "
                                                                           "the IoU data in the training")
    parser.add_argument('--slope-diff-threshold', type=float, default=0.95, help="pseudo label will begin to update "
                                                                                "after the difference of the curve's "
                                                                                "slope bigger than the threshold")
    parser.add_argument('--trustable-quantile', type=float, default=0.66)

    # multiscale consistency loss hyperparameters
    parser.add_argument('--feature-consistency-loss-weight', type=float, default=0.001)
    parser.add_argument('--prediction-consistency-loss-weight', type=float, default=1.0)

    # pseudo-label correction ablation study
    parser.add_argument('--no-update-pseudo', action='store_true')
    parser.add_argument('--pseudo-update', choices=["ELR", "offline", "online"], default="online")
    parser.add_argument('--former-train-IOU-record', type=str)

    # prompt adapter ablation
    parser.add_argument('--singlescale-stem', action='store_true', help="single scale stem in prompt generator")
    parser.add_argument('--no-level-embedding', action='store_true', help="prompt generator with no level embedding")
    parser.add_argument('--no-stem', action='store_true', help="prompt generator with no stem")
    parser.add_argument('--no-PG', action='store_true', help="no prompt generator")
    parser.add_argument('--single-PI', action='store_true', help="single prompt interactor")

    # loss ablation
    parser.add_argument('--no-feature-consistency-loss', action='store_true')
    parser.add_argument('--no-prediction-consistency-loss', action='store_true')


    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args
    

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.data_root is not None:
        cfg.data.train["data_root"] = args.data_root
        cfg.data.test["data_root"] = args.data_root
        cfg.data.val["data_root"] = args.data_root
    cfg.model['pretrained'] = args.pretrained_from
    cfg.data.train['ann_dir'] = os.path.join("pretrain", args.pretrained_from.split('/')[-1][0:-4],'train')
    cfg.model['backbone']['freeze_backbone'] = args.freeze_backbone
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    # setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)                

    cfg.model.decode_head.loss_decode = dict(type='GtASelfTrainingLoss', loss_weight=1.0)
    cfg.optimizer.lr = args.lr
    cfg.optimizer.paramwise_cfg.custom_keys.level_embed.lr_mult = args.prompt_lr_mult
    cfg.optimizer.paramwise_cfg.custom_keys.spm.lr_mult = args.prompt_lr_mult
    cfg.optimizer.paramwise_cfg.custom_keys.interactions.lr_mult = args.prompt_lr_mult

    cfg.runner.max_iters = args.train_iter
    cfg.log_config.interval = args.log_freq
    cfg.checkpoint_config.interval = args.checkpoint_save_freq
    cfg.evaluation.interval = args.eval_freq
    cfg.data.samples_per_gpu = args.batch_size


    cfg.pseudo_update = args.pseudo_update
    cfg.no_update_pseudo = args.no_update_pseudo or cfg.pseudo_update == "ETP"
    cfg.curve_update_interval = args.curve_update_interval
    cfg.IoU_queue_length = args.IoU_queue_length
    cfg.slope_diff_threshold = args.slope_diff_threshold
    cfg.trustable_quantile = args.trustable_quantile

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.no_update_pseudo = cfg.no_update_pseudo
    model.no_feature_consistency_loss = args.no_feature_consistency_loss
    model.no_prediction_consistency_loss = args.no_prediction_consistency_loss
    model.feature_consistency_loss_weight = args.feature_consistency_loss_weight
    model.prediction_consistency_loss_weight = args.prediction_consistency_loss_weight

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

if __name__ == '__main__':
    main()
