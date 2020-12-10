#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
import tensorflow as tf
import numpy as np

from stereonet.model import stereonet
from stereonet.utils import conv2d, resnet_block
from refocus_algorithms.layered_dof_tf import do_radial_blur

def _image_feature_net(img,is_training):
    with tf.variable_scope("image_features"):
        image_out = conv2d(inputs=img, filters=16, kernel_size=3, name = 'refine_image_conv1')
        image_out = tf.layers.batch_normalization(inputs=image_out, training=is_training, name='refine_image_bn1')
        image_out = tf.nn.leaky_relu(features=image_out, name='refine_image_leaky1')
        image_out = resnet_block(inputs=image_out, filters=16, kernel_size=3, dilation_rate=1,name='refine_image_res1')
        image_out = resnet_block(inputs=image_out, filters=16, kernel_size=3, dilation_rate=2, name='refine_image_res2')
        return image_out

def _conf_slice_refinement_net(conf_slice, img_features, is_training):
    """
    Small network with shared weigts leearning to refine the confidence volume using the input texture.
    Uses shared weigts for all.
    """
    h = int(conf_slice.shape[1])
    w = int(conf_slice.shape[2])
    #with tf.variable_scope("conf_slice_refinement_size_%d_%d"%(h,w), reuse=tf.AUTO_REUSE):
    with tf.variable_scope("conf_slice_refinement", reuse=tf.AUTO_REUSE):
        conf_slice_out = conv2d(inputs=conf_slice, filters=16, kernel_size=3, name = 'refine_conf_conv1' )
        conf_slice_out = tf.layers.batch_normalization(inputs=conf_slice_out, training=is_training, name='refine_conf_bn1' )
        conf_slice_out = tf.nn.leaky_relu(features=conf_slice_out, name='refine_conf_leaky1')
        conf_slice_out = resnet_block(inputs=conf_slice_out, filters=16, kernel_size=3, dilation_rate=1,name='refine_conf_res1')
        conf_slice_out = resnet_block(inputs=conf_slice_out, filters=16, kernel_size=3, dilation_rate=2, name='refine_conf_res2')
        #Resize image features
        image_out = tf.image.resize_bilinear(img_features, conf_slice.shape[1:3])
        #
        concat_out = tf.concat([image_out,conf_slice_out],axis=3,name='refine_concat')
        #
        concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=4, name='refine_concat_res0')
        #concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=8, name='refine_concat_res1')
        #concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=1, name='refine_concat_res2')
        #concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=1, name='refine_concat_res3')
        #note can use tanh
        conf_residual = conv2d(inputs=concat_out, filters=1, kernel_size=3, name = 'refine_concat_conv1')
        conf_residual =  tf.nn.tanh(conf_residual, name='conf_residual_tanh')
        #
        refined_conf_slice = tf.clip_by_value(conf_slice + conf_residual, 0, 1)
            
        return refined_conf_slice


def _layered_bluring_from_cost_volume_with_learned_upsampling(image, conf_volume, disparity_range,
                                     target_disparity,blur_magnitude,
                                     downsampling_trick_max_kernel_size=None, differenciable=False, 
                                     is_training=False
                                     ):
    """
    Apply a depth-variying radial blur an images using upsampled silces of a cost volume.
    to an dowsampled image from the low resolution cost volume, using a disk kenrel.
    Sweeps the scene from back to front, upsample the cost volume slice according to the blur that needs to be applied,
    apply a single blur kernel on this zones, integarate by plane sweep all the 
    blured partial images.
    
    The slices are upsampled using a small CNN taking the input image for guidance.
    
    image: a batch on images of size BxNxMxC
    conf_volume: the coresponding confidence volume of size BxnxmxD, potentially much smaller that the input image. Must sum to one for each pixel.
    disparity_range: the disparty value for each slice of the cost volume, note:should be on the full resolution disparity referencial.
    target_disparity: the focus plane
    """
    if len(disparity_range) != conf_volume.shape[-1]:
        raise BaseException("Disparity range and cost volume are of different dinentions (%d and %d) respectively."%(len(disparity_range),conf_volume.shape[-1]))
        
    with tf.variable_scope("refocus_cost_volume"):

        with tf.variable_scope("depth_slices_computation"):
            h = int(image.shape[1])
            w = int(image.shape[2])
            #_, hc,wc, _ = cost_volume.shape
            
            #precompute image features used for refinements
            image_features  = _image_feature_net(image, is_training)
                                             
            blur_masked_images = []
            blur_masks = []
            refined_conf_volume = []#note:omly used for dbug
            for d_idx in range(len(disparity_range)):
                d=disparity_range[d_idx]
                print("Disparity slice %f"%d)
                #blur radius and equivalent kernel size we are supposed to blur with
                radius = blur_magnitude*abs(d-target_disparity)
                rounded_radius = int(radius)+1
                kernel_size = 2*rounded_radius+1
                
                conf_slice = tf.expand_dims(conf_volume[:,:,:,d_idx] , -1)
                
                #upample the cost volume to full res
                #if bellow kernel size, otherwise, just upsamples to match the max kernel size
                upsample_fullres = kernel_size <= downsampling_trick_max_kernel_size

                if upsample_fullres:
                    #if target kernel size bellow threshold, upsample full scales
                    conf_slice=tf.image.resize_bilinear(conf_slice, [h,w])
                    resized_img = image
                else:
                    #if not only upsample accordingly and adapt kernel
                    #size of the image we need for eaquivalent blur with kernel size downsampling_trick_max_kernel_size
                    target_downsampling_factor = downsampling_trick_max_kernel_size/float(kernel_size)
                    target_size = (int(np.round(h*target_downsampling_factor)),
                                   int(np.round(w*target_downsampling_factor)))
                    conf_slice=tf.image.resize_bilinear(conf_slice, target_size)
                    resized_img = tf.image.resize_bilinear(image, target_size)#this is ok, just downsampling
                    radius = radius*target_downsampling_factor
                    kernel_size = downsampling_trick_max_kernel_size
                
                #refine the confidence slice with a small neural net
                #Note: the conf volume does not have  o sum to one ?
                # FIXME: div per 0 and no way to apply softmax?
                conf_slice=_conf_slice_refinement_net(conf_slice, image_features, is_training)+0.00000001#needed because the refinement can cause pixels to go to :((
                refined_conf_volume.append(conf_slice)
                
                #mask image with cost slice
                masked_image = conf_slice*resized_img

                #blur both mask and masked image
                blur_conf_slice = do_radial_blur(conf_slice, radius, None)
                blur_masked_image = do_radial_blur(masked_image, radius, None)

                #upsample to full res (bilinear) if needed
                if not upsample_fullres:
                    blur_conf_slice = tf.image.resize_bilinear(blur_conf_slice, [h,w])  
                    blur_masked_image = tf.image.resize_bilinear(blur_masked_image, [h,w])  
                
                #put it in the stack
                blur_masks.append(blur_conf_slice)
                blur_masked_images.append(blur_masked_image)
                        
 
        with tf.variable_scope("image_composition"):
            tf.logging.info("Image composition")
            #composes the final image (this cannot be done in parallell)
            integrated_images = tf.zeros_like(image)
            integrated_masks = tf.zeros_like(image)
            for d in range(len(disparity_range)):
                integrated_images=integrated_images*(1-blur_masks[d])+blur_masked_images[d]
                integrated_masks=integrated_masks*(1-blur_masks[d])+blur_masks[d]
    
        #normalise and return
        return integrated_images/integrated_masks, refined_conf_volume

def refnet_cv_refinement(left_image, right_image, target_disparity, blur_magnitude, 
                         min_disp = 0, max_disp = 300,
                         downsampling_trick_max_kernel_size=11,
                         is_training=True, stop_grads = True,
                         ):
    """
    Model that refocuses at the coarsest level of the cost volume and then tryies
    to upsample with residual learning the blured image.
    """
 
    heigt=int(left_image.shape[1])
    width=int(left_image.shape[2])
                     
    #Get cost volume from the first stage of stereonet
    _, intermediate_steps  = stereonet(left_image, right_image, is_training=True)#FIXME: not working with true
    cost_volume = intermediate_steps["cost_volume_left_view"]
    
    if stop_grads:
        cost_volume = tf.stop_gradient(cost_volume)

    #this if given by stereonet architecture
    disparity_range = np.arange(1,18+1)*8#FIXME: see ben for disp=0
    
    #need to normalise cost volume, note: can play with beta
    #sofmtin turn it into a confidence volume
    beta = 1
    conf_volume = tf.nn.softmax(-beta*cost_volume, dim=-1)


    #refocus at low res direclty from the cost volume
    refocus, refined_conf_volume = _layered_bluring_from_cost_volume_with_learned_upsampling( left_image, conf_volume, disparity_range,
                                                                         target_disparity, blur_magnitude, 
                                                                         downsampling_trick_max_kernel_size,
                                                                         differenciable=False)
    refocus=tf.check_numerics(refocus,    "NaN in refocused image")

    
    return refocus, conf_volume, refined_conf_volume

