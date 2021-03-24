#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the BSD 0-Clause License for more details.


from src.CycleGAN import *
from src.train_cycleGAN import train
import argparse

parser = argparse.ArgumentParser(description='Single-task: One Task')
parser.add_argument('--data_root', default='/', type=str, help='path to dataset')
parser.add_argument('--project_root', default='/', type=str, help='path to dataset')
parser.add_argument('--task', default='A2B', type=str, help='path to dataset')
########################### CycleGAN parameters ############################################################
parser.add_argument('--im_w', default=256, type=int, help='Training image width')
parser.add_argument('--im_h', default=256, type=int, help='Training image height')
parser.add_argument('--clipvalue', default=1.0, type=float, help='Clipping value')
parser.add_argument('--generator_architecture', default='unet_mini', type=str, help='Model architecture')
parser.add_argument('--use_identity_learning', default=False, type=bool, help='Use identity learning')
parser.add_argument('--lambda_1', default=10.0, type=float, help='Lambda 1')
parser.add_argument('--lambda_2', default=10.0, type=float, help='Lambda 2')
parser.add_argument('--lambda_D', default=1.0, type=float, help='Lambda D')
parser.add_argument('--use_resize_convolution', default=True, type=bool, help='Use resize convolution')
parser.add_argument('--use_supervised_learning', default=True, type=bool, help='Use supervised learning')
parser.add_argument('--use_norm', default=True, type=bool, help='')
parser.add_argument('--add_extra_conv', default=False, type=bool, help='')
parser.add_argument('--add_noise', default=False, type=bool, help='')
parser.add_argument('--use_patchgan', default=True, type=bool, help='Use patchgan discriminator')
parser.add_argument('--supervised_weight', default=6.0, type=float, help='Supervised weight')
parser.add_argument('--supervised_loss', default='MAE', type=str, help='Supervised loss type')
parser.add_argument('--epsilon_norm', default=1e-5, type=float, help='Supervised loss type')
########################### Training parameters ############################################################
parser.add_argument('--epochs', default=1000000, type=int, help='Nuumber of training epochs')
parser.add_argument('--exp_name', default='long_exposure_to_short_exposure', type=str, help='Experiment name')
parser.add_argument('--real_label', default=0.8, type=float, help='Real label ground truth')
parser.add_argument('--identity_mapping_modulus', default=10, type=int, help='')
parser.add_argument('--generator_iterations', default=3, type=int, help='Training iterations for generator')
parser.add_argument('--discriminator_iterations', default=1, type=int, help='Training iterations for discriminator')
parser.add_argument('--synthetic_pool_size', default=50, type=int, help='Size of synthetic pool')
parser.add_argument('--use_linear_decay', default=False, type=bool, help='Use linear decay')
parser.add_argument('--decay_epoch', default=101, type=int, help='Decay epoch')
parser.add_argument('--lr_D', default=2e-4, type=float, help='Discriminator learning rate')
parser.add_argument('--lr_G', default=2e-4, type=float, help='Generator learning rate')
parser.add_argument('--beta_1', default=0.5, type=float, help='beta_1')
parser.add_argument('--beta_2', default=0.99, type=float, help='beta_2')
parser.add_argument('--weights_path', default=None, type=str, help='Path to weights')
parser.add_argument('--weights_iter', default=2, type=int, help='Path to weights')

parser.add_argument('--save_images_interval', default=1000, type=int, help='')
parser.add_argument('--save_models_interval', default=1000, type=int, help='')
parser.add_argument('--batch_size', default=1, type=int, help='')

################################################################################################################
parser.add_argument('--gpu_id', dest='gpu_id', help='GPU device id to use [0]', type=str, default=7)
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
cyclegan = CycleGAN(opt, normalization=InstanceNormalization)
train(opt, cyclegan)
