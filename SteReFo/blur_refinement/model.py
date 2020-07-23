import tensorflow as tf
import numpy as np

from stereonet.model import stereonet
from stereonet.utils import conv2d, resnet_block
from refocus_algorithms.layered_dof_tf import layered_bluring

def _blur_image_refinement_net(img, upsampled_blur_image, scale_name = "1", is_training=False):
    """
    Refines the upsampled blur image using the AIF image.
    This is the same block as stereonet.
    """
    with tf.variable_scope("guided_upsampling_scale_"+scale_name):
        #get features from the upsampled blur image
        blur_image_features = conv2d(inputs=upsampled_blur_image, filters=16, kernel_size=3, name = 'refine_blur_image_conv1_' + scale_name )
        blur_image_features = tf.layers.batch_normalization(inputs=blur_image_features, training=is_training, name='refine_blur_image_bn1_' + scale_name )
        blur_image_features = tf.nn.leaky_relu(features=blur_image_features, name='refine_blur_image_leaky1_' + scale_name)
        blur_image_features = resnet_block(inputs=blur_image_features, filters=16, kernel_size=3, dilation_rate=1,name='refine_blur_image_res1_' + scale_name)
        blur_image_features = resnet_block(inputs=blur_image_features, filters=16, kernel_size=3, dilation_rate=2, name='refine_blur_image_res2_' + scale_name)
        
        #get features from the AIF image (downsapled b4 if needed)
        aif_image_features = conv2d(inputs=img, filters=16, kernel_size=3, name = 'refine_image_conv1_' + scale_name)
        aif_image_features = tf.layers.batch_normalization(inputs=aif_image_features, training=is_training, name='refine_image_bn1_' + scale_name)
        aif_image_features = tf.nn.leaky_relu(features=aif_image_features, name='refine_image_leaky1_' + scale_name)
        aif_image_features = resnet_block(inputs=aif_image_features, filters=16, kernel_size=3, dilation_rate=1,name='refine_image_res1_' + scale_name)
        aif_image_features = resnet_block(inputs=aif_image_features, filters=16, kernel_size=3, dilation_rate=2, name='refine_image_res2_' + scale_name)
        #cat
        concat_out = tf.concat([aif_image_features,blur_image_features],axis=3,name='refine_concat_' + scale_name)
        #some seridual blocks
        concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=4, name='refine_concat_res0_' + scale_name)
        concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=8, name='refine_concat_res1_' + scale_name)
        concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=1, name='refine_concat_res2_' + scale_name)
        concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=1, name='refine_concat_res3_' + scale_name)
        #residual to add to the refocus image
        blur_image_residual = conv2d(inputs=concat_out, filters=3, kernel_size=3, name = 'refine_concat_conv1_' + scale_name)
        #normalise it
        blur_image_residual = tf.nn.tanh(blur_image_residual)
        
        #add it and return (note: normalised)
        high_res_blur_image = upsampled_blur_image + blur_image_residual
        
        #clipping?
        #high_res_blur_image = tf.clip_by_value(high_res_blur_image, 0, 1)

        
        return high_res_blur_image

def refnet_blur_refinement(left_image, right_image, target_disparity, blur_magnitude, 
                           min_disp = 0, max_disp = 300,
                           is_training=True, stop_grads = True,
                           from_scale=1):
    """
    Model that refocuses at the coarsest level of the cost volume and then tryies
    to upsample with residual learning the blured image.
    """
 
    heigt=int(left_image.shape[1])
    width=int(left_image.shape[2])
    #Get cost volume from the first stage of stereonet

    _, intermediate_steps  = stereonet(left_image, right_image, is_training=True)#FIXME: not working with true
    
    disparity_1_2 = intermediate_steps["disparity_map_1_2"]
    disparity_1_4 = intermediate_steps["disparity_map_1_4"]
    disparity_1_8 = intermediate_steps["disparity_map_1_8"]
    
    with tf.variable_scope("blur_upsampling"):
        if from_scale == 3:
            lowres_disparity = disparity_1_8
        elif from_scale == 2:
            lowres_disparity = disparity_1_4
        elif from_scale == 1: 
            lowres_disparity = disparity_1_2
        else:
            raise BaseException("Scale id not recognised")   

        heigt_lowres=int(lowres_disparity.shape[1])
        width_lowres=int(lowres_disparity.shape[2])

        if stop_grads:
            lowres_disparity = tf.stop_gradient(lowres_disparity)

        target_disparity = target_disparity/(2**from_scale) #defined by deep stereo
        min_disp = min_disp/(2**from_scale)
        max_disp = max_disp/(2**from_scale)

        print("At low res focus is %f and magnitude %f min max disparity %f %f"%(target_disparity, blur_magnitude, 
                                                                                 min_disp, max_disp))

        #refocus at low res direclty from the cost volume
        left_image_lowres = tf.image.resize_images(left_image,size=(heigt_lowres,width_lowres),align_corners=True)
        refocus_lowres = layered_bluring(left_image_lowres, lowres_disparity, target_disparity,blur_magnitude, 
                                    min_disp, max_disp,
                                    downsampling_trick_max_kernel_size=None,#No need trick
                                    differenciable=True)
        
        initial_refocus = refocus_lowres
        #upsampling with residual learning
        if from_scale == 3:
            tf.logging.info("Upsampling from scale 1/8 to 1/4")
            #get downsampled/upsampled images
            left_image_1_4 = tf.image.resize_images(left_image,size=(int(heigt/4),int(width/4)),align_corners=True)
            #scale 1/4
            refocus_1_4_coarse = tf.image.resize_images(refocus_lowres,size=(int(heigt/4),int(width/4)),align_corners=True)
            refocus_lowres = _blur_image_refinement_net(left_image_1_4, refocus_1_4_coarse, scale_name = "1_4", is_training=is_training)
        
        if from_scale >= 2 :
            tf.logging.info("Upsampling from scale 1/4 to 1/2")
            left_image_1_2 = tf.image.resize_images(left_image,size=(int(heigt/2),int(width/2)),align_corners=True)
            #scale 1/2
            refocus_1_2_coarse = tf.image.resize_images(refocus_lowres,size=(int(heigt/2),int(width/2)),align_corners=True)
            refocus_lowres = _blur_image_refinement_net(left_image_1_2, refocus_1_2_coarse, scale_name = "1_2", is_training=is_training)
        
        if from_scale >= 1 :
            tf.logging.info("Upsampling from scale 1/2 to 1/1")
            #full scale
            refocus_1_1_coarse = tf.image.resize_images(refocus_lowres,size=(int(heigt),int(width)),align_corners=True)
            refocus_1_1 = _blur_image_refinement_net(left_image, refocus_1_1_coarse, scale_name = "1_1", is_training=is_training)
            
 
    return refocus_1_1, initial_refocus, lowres_disparity

