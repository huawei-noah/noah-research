# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""The training loop: data, optimiser, epochs, validation and checkpointing."""
import logging
import os

import torch
import torch.optim as optim

import curvelog
import metric
import model
from data import Adobe5kDataLoader, Dataset
from device import DEVICE


def run_training(args, log_dirpath, writer):
    """Trains CURL, writing a checkpoint whenever validation PSNR improves."""
    BATCH_SIZE = args.batch_size
    EVAL_BATCH_SIZE = 1
    num_epoch = args.num_epoch
    valid_every = args.valid_every
    checkpoint_filepath = args.checkpoint_filepath
    inference_img_dirpath = args.inference_img_dirpath
    training_img_dirpath = args.training_img_dirpath

    training_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                             img_ids_filepath=training_img_dirpath+"/images_train.txt")
    training_data_dict = training_data_loader.load_data()

    training_dataset = Dataset(data_dict=training_data_dict, normaliser=1, is_valid=False,
                               crop_size=args.crop_size)

    validation_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                           img_ids_filepath=training_img_dirpath+"/images_valid.txt")
    validation_data_dict = validation_data_loader.load_data()
    validation_dataset = Dataset(data_dict=validation_data_dict, normaliser=1, is_valid=True)

    testing_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                        img_ids_filepath=training_img_dirpath+"/images_test.txt")
    testing_data_dict = testing_data_loader.load_data()
    testing_dataset = Dataset(data_dict=testing_data_dict, normaliser=1,is_valid=True)

    training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=args.num_workers)
    testing_data_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                                                  num_workers=args.num_workers)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=EVAL_BATCH_SIZE,
                                                     shuffle=False,
                                                     num_workers=args.num_workers)
   
    net = model.CURLNet()
    net.to(DEVICE)

    logging.info('######### Network created #########')
    logging.info('Architecture:\n' + str(net))

    criterion = model.CURLLoss(ssim_window_size=args.ssim_window_size,
                                reg_weight=args.reg_weight)

    '''
    The following objects allow for evaluation of a model on the testing and validation splits of a dataset
    '''
    validation_evaluator = metric.Evaluator(
        criterion, validation_data_loader, "valid", log_dirpath)
    testing_evaluator = metric.Evaluator(
        criterion, testing_data_loader, "test", log_dirpath)

    
    start_epoch=0

    if (checkpoint_filepath is not None) and (inference_img_dirpath is None):
        logging.info('######### Loading Checkpoint #########')
        checkpoint = torch.load(checkpoint_filepath, map_location=DEVICE)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  net.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=args.weight_decay)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for g in optimizer.param_groups:
            # Resuming previously forced 1e-5 regardless of the rate the
            # run had been using, silently changing it by 10x.
            if args.resume_lr is not None:
                g['lr'] = args.resume_lr

        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        net.to(DEVICE)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  net.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epoch, eta_min=args.lr_min)

    best_valid_psnr = 0.0

    # One fixed validation image, held for the whole run, so successive
    # rows differ because the network changed and not the input.
    curve_logger = curvelog.CurveLogger(net, log_dirpath)
    curve_probe = None
    if args.dump_curves_every > 0 and len(validation_dataset) > 0:
        curve_probe = validation_dataset[0]['input_img'].unsqueeze(0).to(DEVICE)
        logging.info('Recording predicted curves every %d epochs into %s'
                     % (args.dump_curves_every, curve_logger.path))

    if args.use_compile:
        net = torch.compile(net)

    # bf16 needs no loss scaling; fp16 does, so it gets a GradScaler.
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(args.amp)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=(args.amp == 'fp16'))

    net.train()

    running_loss = 0.0
    examples = 0
    total_examples = 0

    for epoch in range(start_epoch,num_epoch):

        # train loss
        examples = 0.0
        running_loss = 0.0
    
        for batch_num, data in enumerate(training_data_loader, 0):

            input_img_batch = data['input_img'].to(DEVICE)
            gt_img_batch = data['output_img'].to(DEVICE)

            with torch.autocast(DEVICE.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                net_img_batch, gradient_regulariser = net(input_img_batch)
                net_img_batch = torch.clamp(net_img_batch, 0.0, 1.0)
                loss = criterion(net_img_batch, gt_img_batch,
                                 gradient_regulariser)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            examples += BATCH_SIZE
            total_examples+=BATCH_SIZE

            writer.add_scalar('Loss/train', loss.item(), total_examples)

        logging.info('[%d] train loss: %.15f' %
                     (epoch + 1, running_loss / examples))
        writer.add_scalar('Loss/train_smooth', running_loss / examples, epoch + 1)

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch + 1)


        if curve_probe is not None and (epoch + 1) % args.dump_curves_every == 0:
            curve_logger.log(net, curve_probe, epoch + 1)

        if (epoch + 1) % valid_every == 0:

            logging.info("Evaluating model on validation dataset")

            valid_loss, valid_psnr, valid_ssim = validation_evaluator.evaluate(
                net, epoch)
            test_loss, test_psnr, test_ssim = testing_evaluator.evaluate(
                net, epoch)

            # update best validation set psnr
            if valid_psnr > best_valid_psnr:

                logging.info(
                    "Validation PSNR has increased. Saving the more accurate model to file: " + 'curl_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(valid_psnr,
                                                                                                                                                                                     float(valid_loss), test_psnr, float(test_loss),
                                                                                                                                                                                     epoch))

                best_valid_psnr = valid_psnr
                snapshot_prefix = os.path.join(
                    log_dirpath, 'curl')
                snapshot_path = snapshot_prefix + '_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(valid_psnr,
                                                                                                                                float(valid_loss),
                                                                                                                                test_psnr, float(test_loss),
                                                                                                                                epoch +1)
                '''
                torch.save(net, snapshot_path)
                '''

                torch.save({
                    'epoch': epoch+1,
                     'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                     'loss': loss,
                     }, snapshot_path)

            net.train()

    '''
    Run the network over the testing dataset split
    '''
    snapshot_prefix = os.path.join(
                    log_dirpath, 'curl')

    valid_loss, valid_psnr, valid_ssim = validation_evaluator.evaluate(
                net, epoch)
    test_loss, test_psnr, test_ssim = testing_evaluator.evaluate(
                net, epoch)

    snapshot_path = snapshot_prefix + '_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(valid_psnr,
                                                                                                                                float(valid_loss),
                                                                                                                                test_psnr, float(test_loss),
                                                                                                                                epoch +1)
    snapshot_prefix = os.path.join(log_dirpath, 'curl')
    torch.save({
                    'epoch': epoch+1,
                     'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                     'loss': loss,
                     }, snapshot_path)

