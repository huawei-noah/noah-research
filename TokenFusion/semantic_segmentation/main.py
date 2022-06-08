#Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.
#
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

# general libs
import os, sys, argparse, re
import random, time
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from utils import *
import utils.helpers as helpers
from utils.optimizer import PolyWarmupAdamW
from models.segformer import WeTr

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Full Pipeline Training')

    # Dataset
    parser.add_argument('-d', '--train-dir', type=str, default=TRAIN_DIR,
                        help='Path to the training set directory.')
    parser.add_argument('--val-dir', type=str, default=VAL_DIR,
                        help='Path to the validation set directory.')
    parser.add_argument('--train-list', type=str, default=TRAIN_LIST,
                        help='Path to the training set list.')
    parser.add_argument('--val-list', type=str, default=VAL_LIST,
                        help='Path to the validation set list.')
    parser.add_argument('--shorter-side', type=int, default=SHORTER_SIDE,
                        help='Shorter side transformation.')
    parser.add_argument('--crop-size', type=int, default=CROP_SIZE,
                        help='Crop size for training,')
    parser.add_argument('--input-size', type=int, default=RESIZE_SIZE,
                        help='Final input size of the model')
    parser.add_argument('--normalise-params', type=list, default=NORMALISE_PARAMS,
                        help='Normalisation parameters [scale, mean, std],')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size to train the segmenter model.')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                        help='Number of output classes for each task.')
    parser.add_argument('--low-scale', type=float, default=LOW_SCALE,
                        help='Lower bound for random scale')
    parser.add_argument('--high-scale', type=float, default=HIGH_SCALE,
                        help='Upper bound for random scale')
    parser.add_argument('--ignore-label', type=int, default=IGNORE_LABEL,
                        help='Label to ignore during training')

    # Encoder
    parser.add_argument('--enc', type=str, default=ENC,
                        help='Encoder net type.')
    parser.add_argument('--enc-pretrained', type=bool, default=ENC_PRETRAINED,
                        help='Whether to init with imagenet weights.')
    # General
    parser.add_argument('--name', default='', type=str,
                        help='model name')
    parser.add_argument('--gpu', type=int, nargs='+', default=[1],
                        help='select gpu.')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='If true, only validate segmentation.')
    parser.add_argument('--freeze-bn', type=bool, nargs='+', default=FREEZE_BN,
                        help='Whether to keep batch norm statistics intact.')
    parser.add_argument('--num-epoch', type=int, nargs='+', default=NUM_SEGM_EPOCHS,
                        help='Number of epochs to train for segmentation network.')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument('-c', '--ckpt', default='model', type=str, metavar='PATH',
                        help='path to save checkpoint (default: model)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--val-every', type=int, default=VAL_EVERY,
                        help='How often to validate current architecture.')
    parser.add_argument('--print-network', action='store_true', default=False,
                        help='Whether print newtork paramemters.')
    parser.add_argument('--print-loss', action='store_true', default=False,
                        help='Whether print losses during training.')
    parser.add_argument('--save-image', type=int, default=100,
                        help='Number to save images during evaluating, -1 to save all.')
    parser.add_argument('-i', '--input', default=['rgb', 'depth'], type=str, nargs='+', 
                        help='input type (image, depth)')

    # Optimisers
    parser.add_argument('--lr-enc', type=float, nargs='+', default=LR_ENC,
                        help='Learning rate for encoder.')
    parser.add_argument('--lr-dec', type=float, nargs='+', default=LR_DEC,
                        help='Learning rate for decoder.')
    parser.add_argument('--mom-enc', type=float, default=MOM_ENC,
                        help='Momentum for encoder.')
    parser.add_argument('--mom-dec', type=float, default=MOM_DEC,
                        help='Momentum for decoder.')
    parser.add_argument('--wd-enc', type=float, default=WD_ENC,
                        help='Weight decay for encoder.')
    parser.add_argument('--wd-dec', type=float, default=WD_DEC,
                        help='Weight decay for decoder.')
    parser.add_argument('--optim-dec', type=str, default=OPTIM_DEC,
                        help='Optimiser algorithm for decoder.')
    parser.add_argument('--lamda', type=float, default=LAMDA,
                        help='Lamda for L1 norm.')
    # parser.add_argument('-t', '--bn-threshold', type=float, default=BN_threshold,
    #                     help='Threshold for slimming BNs.')
    parser.add_argument('--backbone', default='mit_b1', type=str)
    return parser.parse_args()


def create_segmenter(num_classes, gpu, backbone):
    """Create Encoder; for now only ResNet [50,101,152]"""
    segmenter = WeTr(backbone, num_classes)
    param_groups = segmenter.get_param_groups()
    assert(torch.cuda.is_available())
    segmenter.to(gpu[0])
    segmenter = torch.nn.DataParallel(segmenter, gpu)
    # segmenter = DistributedDataParallel(wetr, device_ids=[-1], find_unused_parameters=True)
    return segmenter, param_groups


def create_loaders(dataset, inputs, train_dir, val_dir, train_list, val_list,
                   shorter_side, crop_size, input_size, low_scale, high_scale,
                   normalise_params, batch_size, num_workers, ignore_label):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      crop_size (int) : square crop to apply during the training.
      low_scale (float) : lowest scale ratio for augmentations.
      high_scale (float) : highest scale ratio for augmentations.
      normalise_params (list / tuple) : img_scale, img_mean, img_std.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.
      ignore_label (int) : label to pad segmentation masks with

    Returns:
      train_loader, val loader

    """
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    # Custom libraries
    from utils.datasets import SegDataset as Dataset
    from utils.transforms import Normalise, Pad, RandomCrop, RandomMirror, ResizeAndScale, \
                                 CropAlignToMask, ResizeAlignToMask, ToTensor, ResizeInputs

    input_names, input_mask_idxs = ['rgb', 'depth'], [0, 2, 1]

    AlignToMask = CropAlignToMask if dataset == 'nyudv2' else ResizeAlignToMask
    composed_trn = transforms.Compose([
        AlignToMask(),
        ResizeAndScale(shorter_side, low_scale, high_scale),
        Pad(crop_size, [123.675, 116.28 , 103.53], ignore_label),
        RandomMirror(),
        RandomCrop(crop_size),
        ResizeInputs(input_size),
        Normalise(*normalise_params),
        ToTensor()
    ])
    composed_val = transforms.Compose([
        AlignToMask(),
        ResizeInputs(input_size),
        Normalise(*normalise_params),
        ToTensor()
    ])
    # Training and validation sets
    trainset = Dataset(dataset=dataset, data_file=train_list, data_dir=train_dir,
                       input_names=input_names, input_mask_idxs=input_mask_idxs,
                       transform_trn=composed_trn, transform_val=composed_val,
                       stage='train', ignore_label=ignore_label)

    validset = Dataset(dataset=dataset, data_file=val_list, data_dir=val_dir,
                       input_names=input_names, input_mask_idxs=input_mask_idxs,
                       transform_trn=None, transform_val=composed_val, stage='val',
                       ignore_label=ignore_label)
    print_log('Created train set {} examples, val set {} examples'.format(len(trainset), len(validset)))
    # Training and validation loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def create_optimisers(lr_enc, lr_dec, mom_enc, mom_dec, wd_enc, wd_dec, param_enc, param_dec, optim_dec):
    """Create optimisers for encoder, decoder and controller"""
    optim_enc = torch.optim.SGD(param_enc, lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc)
    if optim_dec == 'sgd':
        optim_dec = torch.optim.SGD(param_dec, lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec)
    elif optim_dec == 'adam':
        optim_dec = torch.optim.Adam(param_dec, lr=lr_dec, weight_decay=wd_dec, eps=1e-3)

    return optim_enc, optim_dec


def load_ckpt(ckpt_path, ckpt_dict):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    for (k, v) in ckpt_dict.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k])
    best_val = ckpt.get('best_val', 0)
    epoch_start = ckpt.get('epoch_start', 0)
    print_log('Found checkpoint at {} with best_val {:.4f} at epoch {}'.
        format(ckpt_path, best_val, epoch_start))
    return best_val, epoch_start


def train(segmenter, input_types, train_loader, optimizer, epoch,
          segm_crit, freeze_bn, lamda, print_loss=False):
    """Training segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      train_loader (DataLoader) : training data iterator
      optim_enc (optim) : optimiser for encoder
      optim_dec (optim) : optimiser for decoder
      epoch (int) : current epoch
      segm_crit (nn.Loss) : segmentation criterion
      freeze_bn (bool) : whether to keep BN params intact

    """
    train_loader.dataset.set_stage('train')
    segmenter.train()
    if freeze_bn:
        for module in segmenter.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
        # print('train input:', sample['rgb'].shape, sample['depth'].shape, sample['mask'].shape)
        start = time.time()
        inputs = [sample[key].cuda().float() for key in input_types]
        target = sample['mask'].cuda().long()
        # Compute outputs
        outputs, masks = segmenter(inputs)
        loss = 0
        for output in outputs:
            output = nn.functional.interpolate(output, size=target.size()[1:],
                                               mode='bilinear', align_corners=False)
            soft_output = nn.LogSoftmax()(output)
            # Compute loss and backpropagate
            loss += segm_crit(soft_output, target)

        if lamda > 0:
            L1_loss = 0
            for mask in masks:
                L1_loss += sum([torch.abs(m).sum().cuda() for m in mask])
            loss += lamda * L1_loss
        
        optimizer.zero_grad()
        loss.backward()
        if print_loss:
            print('step: %-3d: loss=%.2f' % (i, loss), flush=True)
        optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)
    # slim_params_list = []
    # for slim_param in slim_params:
    #     slim_params_list.extend(slim_param.cpu().data.numpy())
    # slim_params_list = np.array(sorted(slim_params_list))
    # print('Epoch %d, 3%% smallest slim_params: %.4f' % (epoch, \
    #     slim_params_list[len(slim_params_list) // 33]), flush=True)
    # print('Epoch %d, portion of slim_params < %.e: %.4f' % (epoch, bn_threshold, \
    #     sum(slim_params_list < bn_threshold) / len(slim_params_list)), flush=True)
    portion_rgbs, portion_depths = [], []
    for idx, mask in enumerate(masks):
        portion_rgb = (mask[0] < 0.02).sum() / mask[0].flatten().shape[0]
        portion_depth = (mask[1] < 0.02).sum() / mask[1].flatten().shape[0]
        portion_rgbs.append(portion_rgb)
        portion_depths.append(portion_depth)
    portion_rgbs = sum(portion_rgbs) / len(portion_rgbs)
    portion_depths = sum(portion_depths) / len(portion_depths)
    print('Epoch %d, portion of scores<0.02 (rgb depth): %.2f%% %.2f%%' %\
        (epoch, portion_rgbs * 100, portion_depths * 100), flush=True)


def validate(segmenter, input_types, val_loader, epoch, num_classes=-1, save_image=0):
    """Validate segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      val_loader (DataLoader) : training data iterator
      epoch (int) : current epoch
      num_classes (int) : number of classes to consider

    Returns:
      Mean IoU (float)
    """
    global best_iou
    val_loader.dataset.set_stage('val')
    segmenter.eval()
    conf_mat = []
    for _ in range(len(input_types) + 1):
        conf_mat.append(np.zeros((num_classes, num_classes), dtype=int))
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            # print('valid input:', sample['rgb'].shape, sample['depth'].shape, sample['mask'].shape)
            start = time.time()
            inputs = [sample[key].float().cuda() for key in input_types]
            target = sample['mask']
            gt = target[0].data.cpu().numpy().astype(np.uint8)
            gt_idx = gt < num_classes  # Ignore every class index larger than the number of classes
            # Compute outputs
            # outputs, alpha_soft = segmenter(inputs)
            outputs, _ = segmenter(inputs)
            for idx, output in enumerate(outputs):
                output = cv2.resize(output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                                    target.size()[1:][::-1],
                                    interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
                # Compute IoU
                conf_mat[idx] += confusion_matrix(gt[gt_idx], output[gt_idx], num_classes)
                if i < save_image or save_image == -1:
                    img = make_validation_img(inputs[0].data.cpu().numpy(),
                                              inputs[1].data.cpu().numpy(),
                                              sample['mask'].data.cpu().numpy(),
                                              output[np.newaxis,:])
                    os.makedirs('imgs', exist_ok=True)
                    cv2.imwrite('imgs/validate_%d.png' % i, img[:,:,::-1])
                    print('imwrite at imgs/validate_%d.png' % i)

    for idx, input_type in enumerate(input_types + ['ens']):
        glob, mean, iou = getScores(conf_mat[idx])
        best_iou_note = ''
        if iou > best_iou:
            best_iou = iou
            best_iou_note = '    (best)'
        alpha = '        '
        # if idx < len(alpha_soft):
        #     alpha = '    %.2f' % alpha_soft[idx]
        input_type_str = '(%s)' % input_type
        print_log('Epoch %-4d %-7s   glob_acc=%-5.2f    mean_acc=%-5.2f    IoU=%-5.2f%s%s' %
            (epoch, input_type_str, glob, mean, iou, alpha, best_iou_note))
    print_log('')
    return iou


def main():
    global args, best_iou
    best_iou = 0
    args = get_arguments()
    args.num_stages = len(args.lr_enc)

    ckpt_dir = os.path.join('ckpt', args.ckpt)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.system('cp -r *py models utils data %s' % ckpt_dir)
    helpers.logger = open(os.path.join(ckpt_dir, 'log.txt'), 'w+')
    print_log(' '.join(sys.argv))

    # Set random seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # Generate Segmenter
    torch.cuda.set_device(args.gpu[0])
    segmenter, param_groups = create_segmenter(args.num_classes, args.gpu, args.backbone)

    if args.print_network:
        print_log('')
    # segmenter = model_init(segmenter, args.enc, len(args.input), imagenet=args.enc_pretrained)
    print_log('Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M'
          .format(args.backbone, args.enc_pretrained, compute_params(segmenter) / 1e6))
    # Restore if any
    best_val, epoch_start = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            best_val, epoch_start = load_ckpt(args.resume, {'segmenter': segmenter})
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume))
            return
    epoch_current = epoch_start
    # Criterion
    segm_crit = nn.NLLLoss(ignore_index=args.ignore_label).cuda()
    # Saver
    saver = Saver(args=vars(args), ckpt_dir=ckpt_dir, best_val=best_val,
                  condition=lambda x, y: x > y)  # keep checkpoint with the best validation score
    
    lrs = [6e-5, 3e-5, 1.5e-5]

    for task_idx in range(args.num_stages):
        optimizer = PolyWarmupAdamW(
        params=[
                {
                    "params": param_groups[0],
                    "lr": lrs[task_idx],
                    "weight_decay": 0.01,
                },
                {
                    "params": param_groups[1],
                    "lr": lrs[task_idx],
                    "weight_decay": 0.0,
                },
                {
                    "params": param_groups[2],
                    "lr": lrs[task_idx] * 10,
                    "weight_decay": 0.01,
                },
            ],
            lr = lrs[task_idx],
            weight_decay = 0.01,
            betas = [0.9, 0.999],
            warmup_iter = 1500,
            max_iter = 40000,
            warmup_ratio = 1e-6,
            power = 1.0
        )
        total_epoch = sum([args.num_epoch[idx] for idx in range(task_idx + 1)])
        if epoch_start >= total_epoch:
            continue
        start = time.time()
        torch.cuda.empty_cache()
        # Create dataloaders
        train_loader, val_loader = create_loaders(
            DATASET, args.input, args.train_dir, args.val_dir, args.train_list, args.val_list,
            args.shorter_side, args.crop_size, args.input_size, args.low_scale, args.high_scale,
            args.normalise_params, args.batch_size, args.num_workers, args.ignore_label)
        if args.evaluate:
            return validate(segmenter, args.input, val_loader, 0, num_classes=args.num_classes,
                            save_image=args.save_image)

        # Optimisers
        print_log('Training Stage {}'.format(str(task_idx)))
        # optim_enc, optim_dec = create_optimisers(
        #     args.lr_enc[task_idx], args.lr_dec[task_idx],
        #     args.mom_enc, args.mom_dec,
        #     args.wd_enc, args.wd_dec,
        #     enc_params, dec_params, args.optim_dec)

        for epoch in range(min(args.num_epoch[task_idx], total_epoch - epoch_start)):
            train(segmenter, args.input, train_loader, optimizer, epoch_current,
                  segm_crit, args.freeze_bn, args.lamda, args.print_loss)
            if (epoch + 1) % (args.val_every) == 0:
                miou = validate(segmenter, args.input, val_loader, epoch_current, args.num_classes)
                saver.save(miou, {'segmenter' : segmenter.state_dict(), 'epoch_start' : epoch_current})
            epoch_current += 1

        print_log('Stage {} finished, time spent {:.3f}min\n'.format(task_idx, (time.time() - start) / 60.))

    print_log('All stages are now finished. Best Val is {:.3f}'.format(saver.best_val))
    helpers.logger.close()


if __name__ == '__main__':
    main()

