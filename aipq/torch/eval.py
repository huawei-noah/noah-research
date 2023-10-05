# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


# --- P Y T O R C H -----------------------------------------------------
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
# -----------------------------------------------------------------------







sys.path.append('..')
from core.common.utils import create_folder
from core.common.scoring import score, score_tqd, score_shrq
from core.pt import data
from core.pt import models



def get_args():
    parser = argparse.ArgumentParser(description='PyTorch IQA')
    parser.add_argument('--data-dir', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', type=str,
                        choices=['live', 'csiq', 'tid2013',
                                 'liu', 'ma', 'shrq', 'dibr', 'pipal',
                                 'qads', 'tqd', 'kadid', 'pieapp','custom'],
                        help='dataset name')
    parser.add_argument('--model-path',
                        help='path to checkpoint')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                        help='mini-batch size (default: 16)')
    parser.add_argument('--backbone', default='vgg', type=str,
                        choices=['resnet', 'alex', 'vgg', 'squeeze'],help='backbone network')
    parser.add_argument('--lpips', dest='lpips', action='store_true',
                        help='use lpips model')
    parser.add_argument('--dists', dest='dists', action='store_true',
                        help='use dists model')
    parser.add_argument('--iqt', dest='iqt', action='store_true',
                        help='use iqt model')
    parser.add_argument('--l2pooling', action='store_true',
                        help='l2pooling in vgg (similar to DISTS)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--loader', default='pil', type=str,
                        choices=['cv2', 'pil'],
                        help='image loader')
    parser.add_argument('--resize', default=0, type=int,
                        help='image resize')
    parser.add_argument('--patch_sampler', default=None, type=str,
                        choices=['rand', 'entire'],    
                        help='patch sampler strategy')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='turn DEBUG on')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    cudnn.benchmark = True

    main_worker(args)


def main_worker(args):

    # IQA model ----------------------------------------------------
    # FR-IQA
    if args.lpips:
        model = models.LPIPSsinglewRef(net=args.backbone, lpips=True,
                           pretrained=True, pnet_tune=True, l2pooling=args.l2pooling)
    elif args.dists:
        if args.backbone != 'vgg':
            print("[WARNING] ignoring argument '--backbone {}', DISTS only supports vgg.".format(args.backbone))
        model = models.DISTSsinglewRef(pretrained=True, pnet_tune=True)
    elif args.iqt:
        model = models.IQT(train=False)
    # ----------------------------------------------------
    # B-IQA
    elif args.backbone == 'resnet':
        model = models.ResNet50(n_classes=1, pretrained=True)
    elif args.backbone == 'vgg':
        model = models.VGG16(1, args.pretrained)
    elif args.backbone == 'alex':
        model = models.AlexNet(1, args.pretrained)



    checkpoint = torch.load(args.model_path)
    # ----------------------------------------------------
    if args.iqt:
        ckpt_backbone={'state_dict':{} }
        ckpt_transformer={'state_dict':{} }
        for k in checkpoint['state_dict'].keys():
            if 'model_backbone' in k:
                ckpt_backbone['state_dict'][k.replace('model_backbone.','')] = checkpoint['state_dict'][k]
            elif 'model_transformer' in k:
                ckpt_transformer['state_dict'][k.replace('model_transformer.','')] = checkpoint['state_dict'][k]
            else:
                print("NOT FOUND:",k)
        model.model_transformer.load_state_dict(ckpt_transformer['state_dict'], strict=True)
        model.model_transformer.eval()
        model.model_backbone.load_state_dict(ckpt_backbone['state_dict'], strict=True)
        model.model_backbone.eval()
    else:
    # ----------------------------------------------------
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    print("=> loading checkpoint '{}'".format(args.model_path))
    print('from epoch %d' % checkpoint['epoch'])

    model = model.cuda()



    # Data loading code
    # LPIPS
    if args.lpips:
        normalize = transforms.Normalize((.5, .5, .5),(.5, .5, .5))
    # DISTS or IQT
    elif args.dists or args.iqt:
        normalize = transforms.Normalize((0., 0., 0.),(1., 1., 1.))
    # else ...
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])



    if args.loader == 'cv2':
        img_loader = data.cv2_loader
        val_transforms = data.ValTransforms(normalize=normalize, resize=args.resize)
    elif args.loader == 'pil':
        img_loader = data.pil_loader
        val_transforms = data.ValTransformsPIL(normalize=normalize, resize=args.resize)




    #-- PieAPP (test split) ------------------------------------------------
    if args.dataset == 'pieapp':
        df_path = '../data/' + args.dataset + '/test.csv'
        loader = torch.utils.data.DataLoader(
            data.IQAsinglewRef(args.data_dir, val_transforms,
                               df_path=df_path, loader=img_loader),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        y_pred = extract(loader, model, args)

    #-- PIPAL (train split) ------------------------------------------------
    elif args.dataset == 'pipal':
        df_path = '../data/' + args.dataset + '/train.csv'

        loader = torch.utils.data.DataLoader(
            data.PIPALsinglewRef(args.data_dir, val_transforms,
                               df_path=df_path, split='all', loader=img_loader),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        y_pred = extract(loader, model, args)

    #-- LIVE / CSIQ / TID2013 / QADS (...) --------------------------------------
    else:
        #print("DATA_DIR:",args.data_dir)
        df_path = '../data/' + args.dataset + '/all.csv'

        loader = torch.utils.data.DataLoader(
            data.IQAsinglewRef(args.data_dir, val_transforms,
                               df_path=df_path, loader=img_loader),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        y_pred = extract(loader, model, args)

        res = loader.dataset.df
        res['pred'] = np.asarray(y_pred)
        fdir = os.path.join(os.path.dirname(args.model_path), args.dataset)
        if args.resize == 0:
            fpath_txt = os.path.join(fdir, 'output.txt')
            fpath_csv = os.path.join(fdir, 'res.csv')
        else:
            fpath_txt = os.path.join(fdir, 'output_%d.txt' % args.resize)
            fpath_csv = os.path.join(fdir, 'res_%d.csv' % args.resize)

        save_results(res, fpath_txt)
        res.to_csv(fpath_csv)





def save_results(res, fpath):
    final = []
    for i in range(len(res)):
        final.append((res.index[i], res['pred'][i]))
    final = sorted(final, key=lambda x: x[0])

    txt = ''
    for i in range(len(res)):
        if i != 0:
            txt += '\n'
        txt += '%s,%f' % (final[i][0], final[i][1])

    fdir = os.path.dirname(fpath)
    create_folder(fdir)
    with open(fpath, 'w+') as f:
        f.write(txt)


def plot_scores(x, y, fpath):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.xlabel('Predicted scores')
    plt.ylabel('MOS')
    plt.savefig(fpath, dpi=100)
    plt.close()


def extract(val_loader, model, args):
    # switch to evaluate mode
    model.eval()

    all_output = []
    all_target = []
    test_ensemble=args.patch_sampler

    print("inference strategy: '{}', resize={}".format(test_ensemble,args.resize))

    with torch.no_grad():
        end = time.time()
        if isinstance(val_loader.dataset,data.IQAsingleNoRef):
            keys=[]
        else:
            keys = val_loader.dataset.df.keys()

        if 'elo' in keys or 'mos' in keys:
            for i, (ref, (x0, y0)) in tqdm(enumerate(val_loader)):
                if test_ensemble == 'rand':
                    # ensemble eval w/ random patches ------------------------
                    n_ensemble=20
                    crop_size=256
                    pred=0
                    for j in range(n_ensemble):                                      
                        _, c, h, w = ref.size()                                              
                        new_h = crop_size
                        new_w = crop_size
                        top = np.random.randint(0, h - new_h)                               
                        left = np.random.randint(0, w - new_w)                              
                        
                        r_img_crop = ref[:, :, top: top+new_h, left: left+new_w]
                        d_img_crop =  x0[:, :, top: top+new_h, left: left+new_w]

                        r_img_crop = r_img_crop.cuda(non_blocking=True)                           
                        d_img_crop = d_img_crop.cuda(non_blocking=True)
                        y0_patch = y0.cuda(non_blocking=True) # always the same...

                        # compute output
                        if isinstance(model,models.ResNet50):
                            # B-IQA: disregard 'ref' image
                            output, _ = model(d_img_crop)
                            #outref, _ = model(r_img_crop)
                        else:
                            # FR-IQA: 'ref' is part of model input
                            output, _ = model(r_img_crop, d_img_crop)
                        pred += output.cpu().numpy().squeeze()

                    # ensemble average
                    pred /= n_ensemble
                    # record
                    if len(ref) == 1:
                        all_output.append(pred)
                    else:
                        all_output.extend(pred)
                    all_target.extend(y0.cpu().numpy())


                elif test_ensemble == 'entire':
                    # ensemble eval w/ entire image patchified ---------------
                    crop_size=256
                    # LIVE or CSIQ:256 , TID:128, QADS:120
                    if args.dataset in ['tid2013']:
                        step = 128
                    elif args.dataset == 'qads':
                        step=120
                    else:
                        step = 256
                    if args.debug:
                        print("  crop_sz={}, step={}".format(crop_size,step))
                    pred=0
                    from patchify import patchify 
                    ref_img = np.moveaxis(np.squeeze(ref.cpu().detach().numpy()),0,-1)
                    r_patches = patchify(ref_img , (crop_size,crop_size,3) , step)
                    if args.debug:
                        print(" ",r_patches.shape[0]*r_patches.shape[1], \
                                "patches extracted from image:",ref.size())

                    dis_img = np.moveaxis(np.squeeze(x0.cpu().detach().numpy()),0,-1)
                    d_patches = patchify(dis_img , (crop_size,crop_size,3) , step)
                    for j in range(r_patches.shape[0]):
                        for k in range(r_patches.shape[1]):
                            r_patch = torch.from_numpy(np.expand_dims(np.moveaxis(r_patches[j, k, 0],-1,0),0))
                            d_patch = torch.from_numpy(np.expand_dims(np.moveaxis(d_patches[j, k, 0],-1,0),0))

                            # single-patch eval
                            r_patch = r_patch.cuda(non_blocking=True)
                            d_patch = d_patch.cuda(non_blocking=True)
                            y0 = y0.cuda(non_blocking=True)

                            # compute output
                            if isinstance(model,models.ResNet50):
                                # B-IQA: disregard 'ref' image
                                output, _ = model(d_patch)
                            else:
                                # FR-IQA: 'ref' is part of model input
                                output, _ = model(r_patch, d_patch)
                            pred += output.cpu().numpy().squeeze()

                    # ensemble average
                    pred /= (r_patches.shape[0] * r_patches.shape[1])
                    # record
                    if len(ref) == 1:
                        all_output.append(output.cpu().numpy().squeeze())
                    else:
                        all_output.extend(output.cpu().numpy().squeeze())
                        all_target.extend(y0.cpu().numpy())
                    all_target.extend(y0.cpu().numpy())



                else:
                    # single-patch eval --------------------------------------
                    ref = ref.cuda(non_blocking=True)
                    x0 = x0.cuda(non_blocking=True)
                    y0 = y0.cuda(non_blocking=True)

                    if args.debug:
                        print("model input:",ref.shape,x0.shape,y0.shape)
                    # compute output
                    if isinstance(model,models.ResNet50):
                        # B-IQA: disregard 'ref' image
                        output, _ = model(x0)
                    else:
                        # FR-IQA: 'ref' is part of model input
                        output, _ = model(ref, x0)
                    if args.debug:
                        print("model output:",output.shape)

                    # record
                    if len(ref) == 1:
                        all_output.append(output.cpu().numpy().squeeze())
                    else:
                        all_output.extend(output.cpu().numpy().squeeze())
                    all_target.extend(y0.cpu().numpy())
            #-----------------------------------------------------------------

            txt = score(all_output, all_target, prefix='\n(all) ')
            print(txt)

            all_output = np.asarray(all_output)
            all_target = np.asarray(all_target)
            if args.dataset == 'tqd':
                txt = score_tqd(all_output, all_target, val_loader.dataset.df)
                print(txt)
            elif args.dataset == 'shrq':
                txt = score_shrq(all_output, all_target)
                print(txt)

            fdir = os.path.join(os.path.dirname(args.model_path), args.dataset)
            create_folder(fdir)
            if args.resize == 0:
                fpath = os.path.join(fdir, 'plot.png')
            else:
                fpath = os.path.join(fdir, 'plot_%d.png' % args.resize)
            plot_scores(all_output, all_target, fpath=fpath)
        else:
            print("LEN val_loader:",len(val_loader))
            if test_ensemble == 'rand':
                # ensemble eval w/ random patches ------------------------
                n_ensemble=20
                crop_size=256
                pred=0
            elif test_ensemble == 'entire':
                # ensemble eval w/ entire image patchified ---------------
                crop_size=256
                # LIVE or CSIQ:256 , TID:128, QADS:120
            else:
                # single-patch eval --------------------------------------
                all_files=[]
                all_preds=[]
                for i, (x0,fname) in tqdm(enumerate(val_loader)):
                    x0 = x0[0].cuda(non_blocking=True)
                    fname = fname[0]

                    # compute output
                    # in this part of the code, models are BIQA (i.e. no-Ref)
                    output, _ = model(x0)

                    # record
                    all_files.append(fname)
                    all_preds.append(output.cpu().numpy().squeeze())
                all_output = (all_files,all_preds)
    return all_output


def accuracy(d0s, d1s, gts, dists):

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)

    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    txt = 'Overall accuracy: %.3f' % (np.mean(scores))

    for dist in np.unique(dists):
        cond = dists == dist
        scores = (d0s[cond]<d1s[cond])*(1.-gts[cond]) + \
            (d1s[cond]<d0s[cond])*gts[cond] + (d1s[cond]==d0s[cond])*.5

        txt += '\n.. %s: %.3f' % (dist, np.mean(scores))
    return txt


def extract_pair(val_loader, model, args):
    # switch to evaluate mode
    model.eval()

    all_output0 = []
    all_output1 = []
    all_target = []
    all_shape = []
    with torch.no_grad():
        end = time.time()

        for i, (ref, (x0, x1, y)) in tqdm(enumerate(val_loader)):
            ref = ref.cuda(non_blocking=True)
            x0 = x0.cuda(non_blocking=True)
            x1 = x1.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            # compute output
            output0, _ = model(ref, x0)
            output1, _ = model(ref, x1)

            # record
            if len(ref) == 1:
                all_output0.append(output0.cpu().numpy().squeeze())
                all_output1.append(output1.cpu().numpy().squeeze())
            else:
                all_output0.extend(output0.cpu().numpy().squeeze())
                all_output1.extend(output1.cpu().numpy().squeeze())
            all_target.extend(y.cpu().numpy())

        txt = accuracy(all_output1, all_output0, all_target,
                       val_loader.dataset.df['dist'].values)

        print(txt)


if __name__ == '__main__':
    main()
