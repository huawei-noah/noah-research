#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the BSD 0-Clause License for more details.



from keras.layers import Input, BatchNormalization, Activation, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, \
    Flatten, Dense, UpSampling2D, Cropping2D, ZeroPadding2D
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
#from keras_contrib.layers.normalization.groupnormalization import GroupNormalization

from utilities.instance_normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


def unet_generator_mini(input, output, use_resize_convolution=False, add_extra_conv=False, epsilon=1e-5, name=None,
                        normalization=InstanceNormalization, use_norm=True):
    input_img = Input(input)

    # Block 1
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(input_img)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=64)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=64)(x, training=True)
    block_1_out = LeakyReLU(alpha=0.2)(x)

    x = MaxPooling2D()(block_1_out)

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=128)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=128)(x, training=True)
    block_2_out = LeakyReLU(alpha=0.2)(x)

    x = MaxPooling2D()(block_2_out)

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=256)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=256)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (3, 3), padding='same', name='block3_conv3')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=256)(x, training=True)
    block_3_out = LeakyReLU(alpha=0.2)(x)

    x = MaxPooling2D()(block_3_out)

    # Block 4
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), padding='same', name='block4_conv3')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    block_4_out = LeakyReLU(alpha=0.2)(x)

    x = MaxPooling2D()(block_4_out)

    # Block 5
    x = Conv2D(256, (3, 3), padding='same', name='block5_conv1')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), padding='same', name='block5_conv2')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), padding='same', name='block5_conv3')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    # UP 1

    x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
    ch, cw = get_crop_shape(block_4_out, x)
    crop_conv4 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(block_4_out)
    x = concatenate([x, crop_conv4])

    x = Conv2D(256, (3, 3), padding='same')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    # UP 2
    x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
    ch, cw = get_crop_shape(block_3_out, x)
    crop_conv3 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(block_3_out)
    x = concatenate([x, crop_conv3])

    x = Conv2D(128, (3, 3), padding='same')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=256)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=256)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    # UP 3
    x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
    ch, cw = get_crop_shape(block_2_out, x)
    crop_conv2 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(block_2_out)
    x = concatenate([x, crop_conv2])

    x = Conv2D(64, (3, 3), padding='same')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=128)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=128)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    # UP 4
    x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
    ch, cw = get_crop_shape(block_1_out, x)
    crop_conv1 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(block_1_out)
    x = concatenate([x, crop_conv1])

    x = Conv2D(32, (3, 3), padding='same')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=64)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    if add_extra_conv:
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

    ch, cw = get_crop_shape(input_img, x)
    x = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(x)
    x = Conv2D(output[-1], (3, 3), padding='same')(x)

    x = Activation('tanh')(x)

    return Model(inputs=input_img, outputs=x, name=name)


def unet_discriminator_mini(input_shape, name=None, epsilon=1e-5, use_patchgan=True,  use_norm=True,
                            normalization=InstanceNormalization):
    input_img = Input(input_shape)
    # Block 1
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(input_img)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=64)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=64)(x, training=True)
    block_1_out = LeakyReLU(alpha=0.2)(x)

    x = MaxPooling2D()(block_1_out)

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=128)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=128)(x, training=True)
    block_2_out = LeakyReLU(alpha=0.2)(x)

    x = MaxPooling2D()(block_2_out)

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=256)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=256)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (3, 3), padding='same', name='block3_conv3')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=256)(x, training=True)
    block_3_out = LeakyReLU(alpha=0.2)(x)

    x = MaxPooling2D()(block_3_out)

    # Block 4
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), padding='same', name='block4_conv3')(x)
    if use_norm: x = normalization(axis=3, center=True, epsilon=epsilon, groups=512)(x, training=True)
    block_4_out = LeakyReLU(alpha=0.2)(x)

    x = MaxPooling2D()(block_4_out)

    if use_patchgan:
        x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
    else:
        x = Flatten()(x)
        x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    return Model(inputs=input_img, outputs=x, name=name)

