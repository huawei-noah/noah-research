#Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.
#
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os, argparse
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from models.model_cfg import gen_b2, dis_b2
import cfg
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1,
                    help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input-size', type=int, default=256,
                    help='input size')
parser.add_argument('--resize-scale', type=int, default=286,
                    help='resize scale (0 is false)')
parser.add_argument('--crop-size', type=int, default=256,
                    help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True,
                    help='random fliplr True of False')
parser.add_argument('--num-epochs', type=int, default=300,
                    help='number of train epochs')
parser.add_argument('--val-every', type=int, default=5,
                    help='how often to validate current architecture')
parser.add_argument('--lrG', type=float, default=0.0002,
                    help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002,
                    help='learning rate for discriminator, default=0.0002')
parser.add_argument('--gama', type=float, default=100,
                    help='gama for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 for Adam optimizer')
parser.add_argument('--print-loss', action='store_true', default=False,
                    help='whether print losses during training')
parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                        help='select gpu.')
parser.add_argument('-c', '--ckpt', default='model', type=str, metavar='PATH',
                    help='path to save checkpoint (default: model)')
parser.add_argument('-i', '--img-types', default=[2, 7, 0], type=int, nargs='+', 
                    help='image types, last image is target, others are inputs')
parser.add_argument('--exchange', type=int, default=1,
                    help='whether use feature exchange')
parser.add_argument('-l', '--lamda', type=float, default=1e-3,
                    help='lamda for L1 norm on BN scales.')
parser.add_argument('-t', '--insnorm-threshold', type=float, default=1e-2,
                    help='threshold for slimming BNs')
parser.add_argument('--enc', default=[0], type=int, nargs='+')
parser.add_argument('--dec', default=[0], type=int, nargs='+')
params = parser.parse_args()

# Directories for loading data and saving results
data_dir = '/.data/taskonomy-sample-model-1'  # 'Modify data path'
# data_dir = '/data/wyk/datasets/taskonomy-sample-model-1'
# data_dir = '/home1/wyk/data/taskonomy-sample-model-1'
model_dir = os.path.join('ckpt', params.ckpt)
save_dir = os.path.join(model_dir, 'results')
save_dir_best = os.path.join(save_dir, 'best')
os.makedirs(save_dir_best, exist_ok=True)
os.makedirs(os.path.join(model_dir, 'insnorm_params'), exist_ok=True)
os.system('cp -r *py models utils data %s' % model_dir)
cfg.logger = open(os.path.join(model_dir, 'log.txt'), 'w+')
print_log(params)

train_file = './data/train_domain.txt'
val_file = './data/val_domain.txt'

domain_dicts = {0: 'rgb', 1: 'normal', 2: 'reshading', 3: 'depth_euclidean', 4: 'depth_zbuffer', 
                5: 'principal_curvature', 6: 'edge_occlusion', 7: 'edge_texture',
                8: 'segment_unsup2d', 9: 'segment_unsup25d'}
params.img_types = [domain_dicts[img_type] for img_type in params.img_types]
print_log('\n' + ', '.join(params.img_types[:-1]) + ' -> ' + params.img_types[-1])
num_parallel = len(params.img_types) - 1

cfg.num_parallel = num_parallel
cfg.use_exchange = params.exchange == 1
cfg.insnorm_threshold = params.insnorm_threshold
cfg.enc, cfg.dec = params.enc, params.dec

# Data pre-processing
transform = transforms.Compose([transforms.Resize(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Train data
train_data = DatasetFromFolder(data_dir, train_file, params.img_types, transform=transform,
                               resize_scale=params.resize_scale, crop_size=params.crop_size,
                               fliplr=params.fliplr)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=params.batch_size,
                                                shuffle=True, drop_last=False)

# Test data
test_data = DatasetFromFolder(data_dir, val_file, params.img_types, transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=params.batch_size,
                                               shuffle=False, drop_last=False)
# test_input, test_target = test_data_loader.__iter__().__next__()

# Models
torch.cuda.set_device(params.gpu[0])
# G = Generator(3, params.ngf, 3)
G = gen_b2()
# D = Discriminator(6, params.ndf, 1)
D = dis_b2(img_size=256, patch_size=4)
G.cuda()
G = torch.nn.DataParallel(G, params.gpu)
D.cuda()
D = torch.nn.DataParallel(D, params.gpu)

# G.normal_weight_init(mean=0.0, std=0.02)
# D.normal_weight_init(mean=0.0, std=0.02)

# slim_params, insnorm_params = [], []
# for name, param in G.named_parameters():
#     if param.requires_grad and name.endswith('weight') and 'insnorm_conv' in name:
#         insnorm_params.append(param)
#         if len(slim_params) % 2 == 0:
#             slim_params.append(param[:len(param) // 2])
#         else:
#             slim_params.append(param[len(param) // 2:])

# Loss function
BCE_loss = torch.nn.BCELoss().cuda()
L2_loss = torch.nn.MSELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG * params.batch_size, betas=(params.beta1, params.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD * params.batch_size, betas=(params.beta1, params.beta2))


def evaluate(G, epoch, training):
    num_parallel_ = 1 if num_parallel == 1 else num_parallel + 1
    l1_losses = init_lists(num_parallel_)
    l2_losses = init_lists(num_parallel_)
    fids = init_lists(num_parallel_)
    kids = init_lists(num_parallel_)
    for i, (test_inputs, test_target) in tqdm(enumerate(test_data_loader), miniters=25, total=len(test_data_loader)):
    # for i, (test_inputs, test_target) in enumerate(test_data_loader):
        # Show result for test image
        test_inputs_cuda = [test_input.cuda() for test_input in test_inputs]
        gen_images, alpha_soft, _ = G(test_inputs_cuda)
        test_target_cuda = test_target.cuda()
        for l, gen_image in enumerate(gen_images):
            if l < num_parallel or num_parallel > 1:
                l1_losses[l].append(L1_loss(gen_image, test_target_cuda).item())
                l2_losses[l].append(L2_loss(gen_image, test_target_cuda).item())
                gen_image = gen_image.cpu().data
                save_dir_ = os.path.join(save_dir, 'fake%d' % l)
                plot_test_result_single(gen_image, i, save_dir=save_dir_)
                if l < num_parallel:
                    save_dir_ = os.path.join(save_dir, 'input%d' % l)
                    if not os.path.exists(os.path.join(save_dir_, '%03d.png' % i)):
                        plot_test_result_single(test_inputs[l], i, save_dir=save_dir_)
        save_dir_ = os.path.join(save_dir, 'real')
        if not os.path.exists(os.path.join(save_dir_, '%03d.png' % i)):
            plot_test_result_single(test_target, i, save_dir=save_dir_)
        # break
        
    for l in range(num_parallel_):
        paths = [os.path.join(save_dir, 'fake%d' % l), os.path.join(save_dir, 'real')]
        fid, kid = calculate_given_paths(paths, batch_size=50, cuda=True, dims=2048)
        fids[l], kids[l] = fid, kid

    l1_avg_losses = [torch.mean(torch.FloatTensor(l1_losses_)) for l1_losses_ in l1_losses]
    l2_avg_losses = [torch.mean(torch.FloatTensor(l2_losses_)) for l2_losses_ in l2_losses]
    return l1_avg_losses, l2_avg_losses, fids, kids, alpha_soft


# Training GAN
D_avg_losses, G_avg_losses = [], []
min_fid = np.inf
slim_penalty = lambda var: torch.abs(var).sum().cuda()
step = 0
for epoch in range(params.num_epochs):
    # l1_avg_losses, l2_avg_losses, fids, kids, alpha_soft = evaluate(G, epoch, training=True)
    # print(fids, kids, flush=True)
    D_losses, G_losses = [], []
    # training
    for i, (inputs, target) in tqdm(enumerate(train_data_loader), miniters=25, total=len(train_data_loader)):
    # for i, (inputs, target) in enumerate(train_data_loader):
        # input & target image data
        x = [input.cuda() for input in inputs]
        y = [target.cuda()] * len(inputs)
        gen_images, _, _ = G(x)

        # Train discriminator with real data
        D_real_decisions = D(x, y)
        D_real_decisions = [D_real_decision.squeeze() for D_real_decision in D_real_decisions]
        real = torch.ones(D_real_decisions[0].size()).cuda()
        D_real_losses = [BCE_loss(D_real_decision, real) for D_real_decision in D_real_decisions]

        # Train discriminator with fake data
        gen_images, _, _ = G(x)
        D_fake_decisions = D(x, gen_images)
        D_fake_decisions = [D_fake_decision.squeeze() for D_fake_decision in D_fake_decisions]
        fake = torch.zeros(D_fake_decisions[0].size()).cuda()
        D_fake_losses = [BCE_loss(D_fake_decision, fake) for D_fake_decision in D_fake_decisions]

        # Back propagation
        D_loss = sum([(D_real_loss + D_fake_loss) * 0.5 for (D_real_loss, D_fake_loss) \
            in zip(D_real_losses, D_fake_losses)])
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_images, _, masks = G(x)
        D_fake_decisions = D(x, gen_images)
        D_fake_decisions = [D_fake_decision.squeeze() for D_fake_decision in D_fake_decisions]
        G_fake_loss = sum([BCE_loss(D_fake_decision, real) for D_fake_decision in D_fake_decisions])

        # L1 loss
        l1_loss = sum([L1_loss(gen_image, y[0]) for gen_image in gen_images])
        slim_loss = 0
        if params.lamda > 0:
            # slim_loss = sum([slim_penalty(m) for m in slim_params]) + sum([slim_penalty(m) for m in masks])
            for mask in masks:
                slim_loss += sum([slim_penalty(m) for m in mask])
                # for m in mask:
                #     print(m.min(), m.max(), m.shape)

        # Back propagation
        G_loss = G_fake_loss + params.gama * l1_loss + params.lamda * slim_loss
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())

        if params.print_loss:
            print('Epoch [%d/%d], Step [%d/%d], ' % \
                (epoch + 1, params.num_epochs, i + 1, len(train_data_loader)), end='')
            print('D_loss: %.4f, G_loss: %.4f' % \
                (D_loss.item(), G_loss.item()), flush=True)

        step += 1
        # break

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    # torch.save(insnorm_params, '{}/insnorm_params/insnorm_params_{:03d}.pth'.format(model_dir, epoch))
    
    if epoch % params.val_every == 0:
        update_best_img = False
        l1_avg_losses, l2_avg_losses, fids, kids, alpha_soft = evaluate(G, epoch, training=True)
        for l in range(len(l1_avg_losses)):
            l1_avg_loss, rl2_avg_loss = l1_avg_losses[l], l2_avg_losses[l]** 0.5
            fid, kid = fids[l], kids[l]
            best_note = ''
            if min_fid > fid:
                min_fid = fid
                best_note = '    (best)'
                update_best_img = True
            if l < num_parallel:
                alpha = '    %.2f' % alpha_soft[l]
                img_type_str = '(%s)' % params.img_types[l][:10]
            else:
                alpha = '        '
                img_type_str = '(ens)'
            print_log('Epoch %3d %-15s   l1_avg_loss: %.5f   rl2_avg_loss: %.5f   fid: %.3f   kid: %.3f%s%s' % \
                (epoch, img_type_str, l1_avg_loss, rl2_avg_loss, fid, kid, alpha, best_note))
        print_log('')
        if update_best_img:
            os.system('cp -r %s/fake* %s' % (save_dir, save_dir_best))
    
    if (epoch + 1) % 100 == 0:
        torch.save(G.state_dict(), os.path.join(model_dir, 'checkpoint-gen-%d.pkl' % epoch))
        torch.save(D.state_dict(), os.path.join(model_dir, 'checkpoint-dis-%d.pkl' % epoch))

# Plot average losses
plot_loss(D_avg_losses, G_avg_losses, params.num_epochs, save_dir=save_dir)
# Save trained parameters of model
torch.save(G.state_dict(), os.path.join(model_dir, 'checkpoint-gen.pkl'))
torch.save(D.state_dict(), os.path.join(model_dir, 'checkpoint-dis.pkl'))
cfg.logger.close()
