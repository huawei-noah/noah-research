#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the BSD 0-Clause License for more details.



import tensorflow as tf
from collections import OrderedDict
from utilities.image_pool import *
from scipy.misc import imsave, toimage
import cv2 as cv2
import keras.backend as K
import csv
import json
import sys
import os


def truncateAndSave(real_, real, synthetic, reconstructed, path_name, channels):
    if len(real.shape) > 3:
        real = real[0]
        synthetic = synthetic[0]
        reconstructed = reconstructed[0]
    if real_ is not None:
        print(real_)
        if len(real_.shape) > 4:
            real_ = real_[0]
        image = np.hstack((real_[0], real, synthetic, reconstructed))
    else:
        image = np.hstack((real, synthetic, reconstructed))

    if channels == 1:
        image = image[:, :, 0]

    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = toimage(im, cmin=-1, cmax=1)
    imsave(path_name, im)


def save_tmp_images(task,
                    root,
                    EXP_name,
                    im_w,
                    im_h,
                    real_images_A,
                    real_images_B,
                    synthetic_images_A,
                    synthetic_images_B,
                    reconstructed_images_A,
                    reconstructed_images_B,
                    loop_index,
                    channels):
    real_images = np.vstack((real_images_A[0], real_images_B[0]))
    synthetic_images = np.vstack((synthetic_images_B[0], synthetic_images_A[0]))
    reconstructed_images = np.vstack((reconstructed_images_A[0], reconstructed_images_B[0]))

    truncateAndSave(None, real_images, synthetic_images, reconstructed_images,
                    root + '/experiments/{}/images/{}.png'.format(EXP_name, 'tmp_' + str(loop_index)), channels)


def saveModel(model, epoch, root, exp_name):
    model_path_w = root + '/experiments//{}/saved_models/{}_weights_epoch_{}.hdf5'.format(exp_name, model.name, epoch)
    model.save_weights(model_path_w)
    model_path_m = root + '/experiments//{}/saved_models/{}_model_epoch_{}.json'.format(exp_name, model.name, epoch)
    model.save_weights(model_path_m)
    json_string = model.to_json()
    with open(model_path_m, 'w') as outfile:
        json.dump(json_string, outfile)
    print('{} has been saved in saved_models/{}/'.format(model.name, exp_name))


def writeMetaDataToJSON(root,
                        exp_name,
                        cyclegan,
                        real_label,
                        generator_iterations,
                        discriminator_iterations,
                        synthetic_pool_size,
                        use_linear_decay,
                        decay_epoch,
                        epochs
                        ):
    print('Training parameters ...')
    print('task: ', cyclegan.task)
    print('use_data_generator: ', cyclegan.use_data_generator)
    print('img shape A: height,width,channels: ', cyclegan.image_shapeA)
    print('img shape B: height,width,channels: ', cyclegan.image_shapeB)
    print('epochs: ', epochs)
    print('batch size: ', cyclegan.batch_size)
    print('synthetic_pool_size: ', synthetic_pool_size)
    print('use linear decay on learning rates: ', use_linear_decay)
    print('epoch where learning rate linear decay is initialized (if use_linear_decay): ', decay_epoch)
    print('generator iterations: ', generator_iterations)
    print('discriminator iterations: ', discriminator_iterations)
    print('learning date G: ', cyclegan.learning_rate_G)
    print('learning date D: ', cyclegan.learning_rate_D)
    print('use patchGAN: ', cyclegan.use_patchgan)
    print('normalization: ', cyclegan.normalization)
    print('use_identity_learning: ', cyclegan.use_identity_learning)
    print('identity_mapping_modulus: ', cyclegan.identity_mapping_modulus)
    print('lambda_1: ', cyclegan.lambda_1)
    print('lambda_2: ', cyclegan.lambda_2)
    print('lambda_D: ', cyclegan.lambda_D)
    print('beta_1: ', cyclegan.beta_1)
    print('beta_2: ', cyclegan.beta_2)
    print('use_supervised_learning: ', cyclegan.use_supervised_learning)
    print('supervised_weight: ', cyclegan.supervised_weight)
    print('supervised_loss: ', cyclegan.supervised_loss)
    print('REAL_LABEL: ', real_label)
    print('number of A train examples: ', len(cyclegan.A_train))
    print('number of B train examples: ', len(cyclegan.B_train))
    print('number of A test examples: ', len(cyclegan.A_test))
    print('number of B test examples:', len(cyclegan.B_test))

    # Save meta_data
    data = {}
    data['meta_data'] = []
    data['meta_data'].append({
        'task': cyclegan.task,
        'img shape A: height,width,channels': cyclegan.image_shapeA,
        'img shape B: height,width,channels': cyclegan.image_shapeB,
        'batch size': cyclegan.batch_size,
        'use_identity_mapping_modulus': cyclegan.use_identity_learning,
        'identity_mapping_modulus': cyclegan.identity_mapping_modulus,
        'synthetic_pool_size': synthetic_pool_size,
        'lambda_1': cyclegan.lambda_1,
        'lambda_2': cyclegan.lambda_2,
        'lambda_d': cyclegan.lambda_D,
        'learning_rate_D': cyclegan.learning_rate_D,
        'learning rate G': cyclegan.learning_rate_G,
        'epochs': epochs,
        'use linear decay on learning rates': use_linear_decay,
        'epoch where learning rate linear decay is initialized (if use_linear_decay)': decay_epoch,
        'generator iterations': generator_iterations,
        'discriminator iterations': discriminator_iterations,
        'use_patchGan': cyclegan.use_patchgan,
        'beta 1': cyclegan.beta_1,
        'beta 2': cyclegan.beta_2,
        'REAL_LABEL': real_label,
        'number of A train examples': len(cyclegan.A_train),
        'number of B train examples': len(cyclegan.B_train),
        'number of A test examples': len(cyclegan.A_test),
        'number of B test examples': len(cyclegan.B_test),
        'use_supervised_learning': cyclegan.use_supervised_learning,
        'supervised_weight': cyclegan.supervised_weight,
        'supervised_loss': cyclegan.supervised_loss,
    })

    with open(root + '/experiments/{}/meta_data.json'.format(exp_name), 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)


def writeLossDataToFile(history, root, exp_name):
    keys = sorted(history.keys())
    with open(root + '/experiments/{}/loss_output.csv'.format(exp_name), 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(keys)
        writer.writerows(zip(*[history[key] for key in keys]))


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_directories(root, exp_name):
    make_dir(root + '/experiments/' + exp_name + '/log/')
    make_dir(root + '/experiments/' + exp_name + '/saved_models/')
    make_dir(root + '/experiments/' + exp_name + '/images/')


def train(opt,
          cyclegan,
          ):
    root = opt.project_root
    exp_name = opt.exp_name
    save_images_interval = opt.save_images_interval
    save_models_interval = opt.save_models_interval
    batch_size = opt.batch_size
    real_label = opt.real_label
    identity_mapping_modulus = opt.identity_mapping_modulus
    generator_iterations = opt.generator_iterations
    discriminator_iterations = opt.discriminator_iterations
    synthetic_pool_size = opt.synthetic_pool_size
    use_linear_decay = opt.use_linear_decay
    decay_epoch = opt.decay_epoch
    weights_path = opt.weights_path
    weights_iter = opt.weights_iter
    epochs = opt.epochs

    make_directories(root, exp_name)

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=root + '/experiments/' + str(exp_name) + '/log/',
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=True
    )
    tensorboard.set_model(cyclegan.G_model)

    def write_log(callback, names, logs):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary)
            callback.writer.flush()

    # ======= Store meta data ==========
    writeMetaDataToJSON(root, exp_name, cyclegan, real_label, generator_iterations, discriminator_iterations,
                        synthetic_pool_size, use_linear_decay, decay_epoch, epochs)

    # ======= Avoid pre-allocating GPU memory ==========
    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    K.tensorflow_backend.set_session(tf.Session(config=config))

    sys.stdout.flush()

    def run_training_iteration(loop_index, epoch_iterations):

        # ======= Discriminator training ==========

        synthetic_images_B = cyclegan.G_A2B.predict(real_images_A)
        synthetic_images_A = cyclegan.G_B2A.predict(real_images_B)
        synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
        synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

        # print(np.amax(real_images_A))
        # print(np.amax(real_images_B))
        # print(np.amin(real_images_A))
        # print(np.amin(real_images_B))

        for _ in range(discriminator_iterations):

            n = 1
            if epoch == 10:
                n = 0.95
            if epoch == 50:
                n = 1

            real_images_A_noise = real_images_A
            real_images_B_noise = real_images_B
            synthetic_images_A_noise = real_images_A
            synthetic_images_B_noise = real_images_B

            DA_loss_real = cyclegan.D_A.train_on_batch(x=real_images_A_noise, y=ones)
            DB_loss_real = cyclegan.D_B.train_on_batch(x=real_images_B_noise, y=ones)
            DA_loss_synthetic = cyclegan.D_A.train_on_batch(x=synthetic_images_A_noise, y=zeros)
            DB_loss_synthetic = cyclegan.D_B.train_on_batch(x=synthetic_images_B_noise, y=zeros)

            DA_loss = DA_loss_real + DA_loss_synthetic
            DB_loss = DB_loss_real + DB_loss_synthetic
            D_loss = DA_loss + DB_loss

            if discriminator_iterations > 1:
                print('D_loss:', D_loss)
                sys.stdout.flush()

        # ======= Generator training ==========
        target_data = [real_images_A, real_images_B]  # Compare reconstructed images to real images

        target_data.append(ones)
        target_data.append(ones)

        if cyclegan.use_supervised_learning:
            target_data.append(real_images_A)
            target_data.append(real_images_B)

        for _ in range(generator_iterations):
            G_loss = cyclegan.G_model.train_on_batch(x=[real_images_A, real_images_B], y=target_data)
            if generator_iterations > 1:
                print('G_loss:', G_loss)
                sys.stdout.flush()

        gA_d_loss_synthetic = G_loss[3]
        gB_d_loss_synthetic = G_loss[4]
        reconstruction_loss_A = G_loss[1]
        reconstruction_loss_B = G_loss[2]

        if cyclegan.use_supervised_learning:
            gA_supervised = G_loss[5]
            gB_supervised = G_loss[6]

        # Identity training
        if cyclegan.use_identity_learning and loop_index % identity_mapping_modulus == 0:
            G_A2B_identity_loss = cyclegan.G_A2B.train_on_batch(x=real_images_B, y=real_images_B)
            G_B2A_identity_loss = cyclegan.G_B2A.train_on_batch(x=real_images_A, y=real_images_A)
            print('G_A2B_identity_loss:', G_A2B_identity_loss)
            print('G_B2A_identity_loss:', G_B2A_identity_loss)

        # Update learning rates
        if use_linear_decay and epoch > decay_epoch:
            cyclegan.update_lr(cyclegan.D_A, decay_D)
            cyclegan.update_lr(cyclegan.D_B, decay_D)
            cyclegan.update_lr(cyclegan.G_model, decay_G)

        # Store training data
        DA_losses.append(DA_loss)
        DB_losses.append(DB_loss)
        gA_d_losses_synthetic.append(gA_d_loss_synthetic)
        gB_d_losses_synthetic.append(gB_d_loss_synthetic)
        gA_losses_reconstructed.append(reconstruction_loss_A)
        gB_losses_reconstructed.append(reconstruction_loss_B)
        if cyclegan.use_supervised_learning:
            gA_losses_supervised.append(gA_supervised)
            gB_losses_supervised.append(gB_supervised)

        GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
        GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
        D_losses.append(D_loss)
        GA_losses.append(GA_loss)
        GB_losses.append(GB_loss)
        G_losses.append(G_loss)
        reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
        reconstruction_losses.append(reconstruction_loss)

        print('\n')
        print('Epoch----------------', epoch, '/', epochs)
        print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
        print('D_loss: ', D_loss)
        print('G_loss: ', G_loss[0])
        print('reconstruction_loss: ', reconstruction_loss)
        print('dA_loss:', DA_loss)
        print('DB_loss:', DB_loss)
        if cyclegan.use_supervised_learning:
            print('gA_supervised:', gA_supervised)
            print('gB_supervised:', gB_supervised)

        write_log(tensorboard, ['D_loss'], [D_loss])

        write_log(tensorboard, ['GA_loss'], [GA_loss])
        write_log(tensorboard, ['GB_loss'], [GB_loss])

        write_log(tensorboard, ['gA_d_loss_synthetic'], [gA_d_loss_synthetic])
        write_log(tensorboard, ['gB_d_loss_synthetic'], [gB_d_loss_synthetic])

        write_log(tensorboard, ['reconstruction_loss_A'], [reconstruction_loss_A])
        write_log(tensorboard, ['reconstruction_loss_B'], [reconstruction_loss_B])

        write_log(tensorboard, ['DA_loss'], [DA_loss])
        write_log(tensorboard, ['DB_loss'], [DB_loss])

        if cyclegan.use_supervised_learning:
            write_log(tensorboard, ['GA_supervised'], [gA_supervised])
            write_log(tensorboard, ['GB_supervised'], [gA_supervised])

        if loop_index % save_images_interval == 0:
            print('time to save images')

            reconstructed_images_A = cyclegan.G_B2A.predict(synthetic_images_B)
            reconstructed_images_B = cyclegan.G_A2B.predict(synthetic_images_A)

            save_tmp_images(cyclegan.task, root, exp_name, cyclegan.im_w, cyclegan.im_h, real_images_A, real_images_B,
                            synthetic_images_A, synthetic_images_B, reconstructed_images_A, reconstructed_images_B,
                            loop_index, cyclegan.channels)

        if loop_index % save_models_interval == 0:
            saveModel(cyclegan.D_A, loop_index, root, exp_name)
            saveModel(cyclegan.D_B, loop_index, root, exp_name)
            saveModel(cyclegan.G_A2B, loop_index, root, exp_name)
            saveModel(cyclegan.G_B2A, loop_index, root, exp_name)

        # if validation_interval is not None:
        #    if loop_index % validation_interval == 0:
        #        print('benchmarking .... ')
        #        if cyclegan.task == 'Vimeo2Long_SID':
        #            fid_A2B, fid_B2A = benchmark_v2l_SID_online(cyclegan, loop_index, cyclegan.A_test, cyclegan.B_test)

        #            write_log(tensorboard, ['fid_A2B'], [fid_A2B])
        #            write_log(tensorboard, ['fid_B2A'], [fid_B2A])

        #           if not os.path.exists(root + '/fid_scores'):
        #               os.makedirs(root + '/fid_scores')
        #           f = open(root + '/fid_scores' + '/fid_' + str(loop_index) + '_' + str(epoch) + '.txt', "w+")
        #           f.write('fid_A2B ' + str(fid_A2B) + '\n')
        #           f.write('fid_B2A ' + str(fid_B2A) + '\n')

        #           print('fid_A2B: ', fid_A2B)
        #           print('fid_B2A: ', fid_B2A)
        #       elif cyclegan.task == 'Long2Short_SID' or cyclegan.task == 'Long2Short_SID_sup':
        #           all_psnr_syn_realA, all_psnr_syn_realB, all_MAPE_syn_realA, all_MAPE_syn_realB, all_ssim_syn_realA, all_ssim_syn_realB = benchmark_l2s_SID_test_v2(
        #               v_100, epoch, loop_index, cyclegan, root + '/test_results', do_print=True, do_save=True)

        # all_psnr_syn_realA, all_psnr_syn_realB, all_MAPE_syn_realA, all_MAPE_syn_realB, all_ssim_syn_realA, all_ssim_syn_realB = benchmark_set(
        # cyclegan, 'valid', v_100, root, loop_index, do_save=save_validation_results,
        # validation_no_images=validation_no_images, do_print=False)

        # print('all_psnr_syn_realB: ', all_psnr_syn_realB)
        # print('all_MAPE_syn_realB: ', all_MAPE_syn_realB)

        #           write_log(tensorboard, ['all_psnr_syn_realB'], [all_psnr_syn_realB])
        #           write_log(tensorboard, ['all_MAPE_syn_realB'], [all_MAPE_syn_realB])
        #           write_log(tensorboard, ['all_ssim_syn_realB'], [all_ssim_syn_realB])

    # ======================================================================
    # Begin training
    # ======================================================================
    training_history = OrderedDict()

    DA_losses = []
    DB_losses = []
    gA_d_losses_synthetic = []
    gB_d_losses_synthetic = []
    gA_losses_reconstructed = []
    gB_losses_reconstructed = []

    if cyclegan.use_supervised_learning:
        gA_losses_supervised = []
        gB_losses_supervised = []

    GA_losses = []
    GB_losses = []
    reconstruction_losses = []
    D_losses = []
    G_losses = []

    # Image pools used to update the discriminators
    synthetic_pool_A = ImagePool(synthetic_pool_size)
    synthetic_pool_B = ImagePool(synthetic_pool_size)

    # load trained weights
    if weights_path is not None:
        cyclegan.load_model_and_weights(cyclegan.G_A2B, weights_path, weights_iter, by_name=True)
        cyclegan.load_model_and_weights(cyclegan.G_B2A, weights_path, weights_iter, by_name=True)
        cyclegan.load_model_and_weights(cyclegan.D_A, weights_path, weights_iter, by_name=True)
        cyclegan.load_model_and_weights(cyclegan.D_B, weights_path, weights_iter, by_name=True)
        print('Weights loaded from ', weights_path)

    label_shape = (batch_size,) + cyclegan.D_A.output_shape[1:]
    ones = np.ones(shape=label_shape) * real_label
    zeros = ones * 0

    # Linear decay
    if use_linear_decay:
        decay_D, decay_G = cyclegan.get_lr_linear_decay_rate()

    for epoch in range(1, epochs + 1):
        if cyclegan.use_data_generator:

            print('images in generator, ', len(cyclegan.data_generator))
            loop_index = 1
            for images in cyclegan.data_generator:
                real_images_A = images[0]
                real_images_B = images[1]
                if len(real_images_A.shape) == 3:
                    real_images_A = real_images_A[:, :, :, np.newaxis]
                    real_images_B = real_images_B[:, :, :, np.newaxis]

                if real_images_A.shape[0] != 0 and real_images_B.shape[0] != 0:
                    if real_images_A[0].shape == cyclegan.image_shapeA and real_images_B[
                        0].shape == cyclegan.image_shapeB:
                        # Run all training steps
                        run_training_iteration(loop_index, cyclegan.data_generator.__len__())

                else:
                    continue

                    # Break if loop has ended
                if loop_index >= cyclegan.data_generator.__len__():
                    break

                loop_index += 1

        else:  # Train with all data in cache
            A_train = cyclegan.A_train
            B_train = cyclegan.B_train
            random_order_A = np.random.randint(len(A_train), size=len(A_train))
            random_order_B = np.random.randint(len(B_train), size=len(B_train))
            epoch_iterations = max(len(random_order_A), len(random_order_B))
            min_nr_imgs = min(len(random_order_A), len(random_order_B))

            # If we want supervised learning the same images form
            # the two domains are needed during each training iteration
            if cyclegan.use_supervised_learning:
                # random_order_B = random_order_A
                indexes_A = np.random.randint(len(A_train), size=batch_size)
                indexes_B = indexes_A
            else:
                # random_order_B = random_order_A
                indexes_A = np.random.randint(len(A_train), size=batch_size)
                indexes_B = indexes_A

            for loop_index in range(0, epoch_iterations, batch_size):

                real_images_A = A_train[indexes_A]
                real_images_B = B_train[indexes_B]

                ###### supervised data!

                if cyclegan.use_supervised_learning:
                    A_train = cyclegan.A_train
                    B_train = cyclegan.B_train
                    random_order_A = np.random.randint(len(A_train), size=len(A_train))
                    random_order_B = np.random.randint(len(B_train), size=len(B_train))
                    epoch_iterations = max(len(random_order_A), len(random_order_B))
                    min_nr_imgs = min(len(random_order_A), len(random_order_B))

                    random_order_B = random_order_A
                    indexes_A = random_order_A[loop_index:
                                               loop_index + batch_size]
                    indexes_B = random_order_B[loop_index:
                                               loop_index + batch_size]
                    real_images_A_sup = A_train[indexes_A]
                    real_images_B_sup = B_train[indexes_B]

                # Run all training steps
                run_training_iteration(loop_index, epoch_iterations)

        training_history = {
            'DA_losses': DA_losses,
            'DB_losses': DB_losses,
            'gA_d_losses_synthetic': gA_d_losses_synthetic,
            'gB_d_losses_synthetic': gB_d_losses_synthetic,
            'gA_losses_reconstructed': gA_losses_reconstructed,
            'gB_losses_reconstructed': gB_losses_reconstructed,
            'D_losses': D_losses,
            'G_losses': G_losses,
            'reconstruction_losses': reconstruction_losses}
        writeLossDataToFile(training_history, root, exp_name)

        # Flush out prints each loop iteration
        sys.stdout.flush()
