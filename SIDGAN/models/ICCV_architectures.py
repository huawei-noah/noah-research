#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the BSD 0-Clause License for more details.


from keras.layers import Layer, Input, Conv2D, Activation, add, UpSampling2D, Conv2DTranspose, Flatten, AveragePooling2D, InputSpec
from utilities.instance_normalization import InstanceNormalization

from keras.layers.core import Dense
from keras.models import Model
import tensorflow as tf

def modelDiscriminator(input_shape, name=None, use_patchgan=True, disc_use_4_layers=True):
    # Specify input
    input_img = Input(input_shape)
    # Layer 1 (#Instance normalization is not used for this layer)
    x = ck(input_img, 64, False)
    # Layer 2
    x = ck(x, 128, True)
    # Layer 3
    x = ck(x, 256, True)
    if disc_use_4_layers:
    # Layer 4
        x = ck(x, 512, True)
    # Output layer
    if use_patchgan:
        x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
    else:
        x = Flatten()(x)
        x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    return Model(inputs=input_img, outputs=x, name=name)

def modelGenerator(conv_kernel_c7Ak, input, output, use_resize_convolution, name=None):
        # Specify input
    input_img = Input(input)
    # Layer 1
    x = ReflectionPadding2D((3, 3))(input_img)
    x = c7Ak(x, 32, conv_kernel_c7Ak)
    # Layer 2
    x = dk(x, 64)
    # Layer 3
    x = dk(x, 128)

    # Layer 4-12: Residual layer
    for _ in range(4, 13):
        x = Rk(x)

    # Layer 13
    x = uk(x, 64,use_resize_convolution)
    # Layer 14
    x = uk(x, 32,use_resize_convolution)
    x = ReflectionPadding2D((3, 3))(x)

    x = Conv2D(output[-1], kernel_size=7, strides=1)(x)
    x = Activation('tanh')(x)  # They say they use Relu but really they do not

    return Model(inputs=input_img, outputs=x, name=name)

def ck(x, k, use_normalization, normalization=InstanceNormalization):
    x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
    # Normalization is not done on the first discriminator layer
    if use_normalization:
        x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def c7Ak(x, k, conv_kernel, normalization=InstanceNormalization):
    x = Conv2D(filters=k, kernel_size=conv_kernel, strides=1, padding='valid')(x)
    x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x

def dk(x, k, normalization=InstanceNormalization):
    x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
    x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x

def Rk(x0, normalization=InstanceNormalization):
    k = int(x0.shape[-1])
    # first layer
    x = ReflectionPadding2D((1,1))(x0)
    x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
    x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    # second layer
    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
    x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    # merge
    x = add([x, x0])
    return x

def uk(x, k, use_resize_convolution, normalization=InstanceNormalization ):
    # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
    if use_resize_convolution:
        x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
    else:
        x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractinoally stided with stride 1/2
    x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x

#===============================================================================
# Models

def modelMultiScaleDiscriminator(input_shape, name=None):
    x1 = Input(input_shape)
    x2 = AveragePooling2D(pool_size=(2, 2))(x1)
    #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

    out_x1 = modelDiscriminator('D1')(x1)
    out_x2 = modelDiscriminator('D2')(x2)
    #out_x4 = self.modelDiscriminator('D4')(x4)

    return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')





