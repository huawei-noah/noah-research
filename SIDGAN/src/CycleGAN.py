#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the BSD 0-Clause License for more details.



from keras.optimizers import Adam
from models.ICCV_architectures import *
from models.unet import *
from keras.engine.topology import Network
import sys

import tensorflow as tf
from utilities.data_loader import *


class CycleGAN():
    def __init__(self,
                 opt,
                 image_shape=(256 * 1, 256 * 1, 3),
                 load_training_data=True,
                 normalization=InstanceNormalization,
                 ):

        self.task = opt.task
        self.im_w = opt.im_w
        self.im_h = opt.im_h

        self.data_root = opt.data_root

        self.img_shape = image_shape
        self.channels = self.img_shape[-1]

        # Fetch data during training instead of pre caching all images
        self.use_data_generator = True
        self.generator_architecture = opt.generator_architecture
        self.use_norm = opt.use_norm
        self.add_extra_conv = opt.add_extra_conv

        self.image_shapeA = (opt.im_w * 1, opt.im_h * 1, 3)
        self.image_shapeA_in = (None, None, 3)
        if self.task == 'Long2Short_raw':
            self.image_shapeB = (opt.im_w * 1, opt.im_h * 1, 1)
            self.image_shapeB_in = (None, None, 3)
        else:
            self.image_shapeB = (opt.im_w * 1, opt.im_h * 1, 3)
            self.image_shapeB_in = (None, None, 3)

        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = opt.use_identity_learning
        self.identity_mapping_modulus = opt.identity_mapping_modulus  # Identity mapping will be done each time the iteration number is divisable with this number

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = opt.use_patchgan

        self.normalization = normalization

        # Loss hyperparameters
        self.lambda_1 = opt.lambda_1  # Cyclic loss weight A_2_B
        self.lambda_2 = opt.lambda_2  # Cyclic loss weight B_2_A
        self.lambda_D = opt.lambda_D  # Weight for loss from discriminator guess on synthetic images

        # Learning rates
        self.learning_rate_D = opt.lr_D
        self.learning_rate_G = opt.lr_G

        self.beta_1 = opt.beta_1
        self.beta_2 = opt.beta_2
        self.batch_size = 1
        self.clipvalue = opt.clipvalue
        self.epsilon_norm = opt.epsilon_norm

        # self.crop_res = opt.crop_res

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = opt.use_resize_convolution

        # Supervised learning part
        self.use_supervised_learning = opt.use_supervised_learning
        self.supervised_weight = opt.supervised_weight
        self.supervised_loss = opt.supervised_loss

        # optimizer
        if opt.clipvalue is not None:
            self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2, clipvalue=self.clipvalue)
            self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2, clipvalue=self.clipvalue)
        else:
            self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
            self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # # ======= Discriminator model ==========

        if self.generator_architecture == 'ICCV':

            D_A = modelDiscriminator(self.image_shapeA, use_patchgan=self.use_patchgan,
                                     disc_use_4_layers=True)
            D_B = modelDiscriminator(self.image_shapeB, use_patchgan=self.use_patchgan,
                                     disc_use_4_layers=True)
            loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images

            loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images
        elif self.generator_architecture == 'unet_mini':
            D_A = unet_discriminator_mini(self.image_shapeA, use_norm=self.use_norm, epsilon=self.epsilon_norm,
                                          use_patchgan=self.use_patchgan)
            D_B = unet_discriminator_mini(self.image_shapeB, use_norm=self.use_norm, epsilon=self.epsilon_norm,
                                          use_patchgan=self.use_patchgan)
            loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images

        # Discriminator builds
        image_A = Input(self.image_shapeA)
        image_B = Input(self.image_shapeB)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
        self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')

        if self.use_patchgan:

            self.D_A.compile(optimizer=self.opt_D,
                             loss=self.lse,
                             loss_weights=loss_weights_D)
            self.D_B.compile(optimizer=self.opt_D,
                             loss=self.lse,
                             loss_weights=loss_weights_D)
        else:
            self.D_A.compile(optimizer=self.opt_D,
                             loss='binary_crossentropy',
                             loss_weights=loss_weights_D)
            self.D_B.compile(optimizer=self.opt_D,
                             loss='binary_crossentropy',
                             loss_weights=loss_weights_D)

        # Use Networks to avoid falsy keras error about weight descripancies
        self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')

        # ============= Generator models =======================
        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Generators
        if self.generator_architecture == 'ICCV':
            self.G_A2B = modelGenerator(conv_kernel_c7Ak=7,
                                        use_resize_convolution=self.use_resize_convolution, input=self.image_shapeA,
                                        output=self.image_shapeB, name='G_A2B_model')
            self.G_B2A = modelGenerator(conv_kernel_c7Ak=7,
                                        use_resize_convolution=self.use_resize_convolution, input=self.image_shapeB,
                                        output=self.image_shapeA, name='G_B2A_model')

        elif self.generator_architecture == 'unet_mini':
            self.G_A2B = unet_generator_mini(input=self.image_shapeA,
                                             output=self.image_shapeB,
                                             normalization=normalization,
                                             epsilon=self.epsilon_norm,
                                             use_norm=self.use_norm,
                                             add_extra_conv=self.add_extra_conv,
                                             use_resize_convolution=self.use_resize_convolution,
                                             name='G_A2B_model')
            self.G_B2A = unet_generator_mini(input=self.image_shapeB,
                                             output=self.image_shapeA,
                                             normalization=normalization,
                                             epsilon=self.epsilon_norm,
                                             use_norm=self.use_norm,
                                             add_extra_conv=self.add_extra_conv,
                                             use_resize_convolution=self.use_resize_convolution,
                                             name='G_B2A_model')

        if self.use_identity_learning:
            self.G_A2B.compile(optimizer=self.opt_G, loss='MAE')
            self.G_B2A.compile(optimizer=self.opt_G, loss='MAE')

        # Generator builds
        real_A = Input(shape=self.image_shapeA, name='real_A')
        real_B = Input(shape=self.image_shapeB, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        dA_guess_synthetic = self.D_A_static(synthetic_A)
        dB_guess_synthetic = self.D_B_static(synthetic_B)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)

        model_outputs = [reconstructed_A, reconstructed_B]

        compile_losses = [self.cycle_loss, self.cycle_loss, self.lse, self.lse]
        compile_weights = [self.lambda_1, self.lambda_2, self.lambda_D, self.lambda_D]

        model_outputs.append(dA_guess_synthetic)
        model_outputs.append(dB_guess_synthetic)

        if self.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            if self.supervised_loss == 'MAE':
                compile_losses.append('MAE')
                compile_losses.append('MAE')

            compile_weights.append(self.supervised_weight)
            compile_weights.append(self.supervised_weight)

        self.G_model = Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)

        # ======= Data ==========
        # Use 'None' to fetch all available images
        nr_A_test_imgs = 1000
        nr_B_test_imgs = 1000

        if self.use_data_generator:
            print('--- Using dataloader during training ---')
        else:
            print('--- Caching data ---')
        sys.stdout.flush()

        if load_training_data:
            if self.use_data_generator:
                self.data_generator = load_data(task=self.task, root=self.data_root, batch_size=self.batch_size,
                                                crop_size=self.im_w, generator=True)
                # Only store test images

            if opt.task == 'Vimeo2Long_SID':
                self.A_test, self.B_test, test_A_image_names, test_B_image_names = get_test_data(nr_A_test_imgs,
                                                                                                 nr_B_test_imgs)
            else:
                self.A_test = []
                self.B_test = []

        self.A_train = []
        self.B_train = []

        if not self.use_data_generator:
            print('Data has been loaded')

    def load_model_and_weights(self, model, weights_path, iteration, by_name):
        name = model.name + '_weights_epoch_' + str(iteration)
        final_path = os.path.join(root, weights_path, '{}.hdf5'.format(name))
        model.load_weights(final_path, by_name=by_name)

    def print_info(self):
        print('fInitializing Cycle GAN with parameters ...')
        print('task: ', self.task)
        print('generator architecture: ', self.generator_architecture)
        print('image width: ', self.im_w)
        print('image height: ', self.im_h)
        print('learning date G: ', self.learning_rate_G)
        print('learning date D: ', self.learning_rate_D)
        print('use patchGAN: ', self.use_patchgan)
        print('use_identity_learning: ', self.use_identity_learning)
        print('normalization: ', self.normalization)
        print('identity_mapping_modulus: ', self.identity_mapping_modulus)
        print('lambda_1: ', self.lambda_1)
        print('lambda_2: ', self.lambda_2)
        print('lambda_D: ', self.lambda_D)
        print('beta_1: ', self.beta_1)
        print('beta_2: ', self.beta_2)
        print('use_supervised_learning: ', self.use_supervised_learning)
        print('supervised_weight: ', self.supervised_weight)
        print('supervised_loss: ', self.supervised_loss)

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss
