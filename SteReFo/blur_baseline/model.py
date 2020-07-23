import tensorflow as tf
import numpy as np

from stereonet.model import stereonet
from refocus_algorithms.layered_dof_tf import layered_bluring, layered_bluring_from_cost_volume


def refnet_blur_baseline(left_image, right_image, target_disparities, blur_magnitude, 
                         is_training=True, stop_grads = True,
                         min_disp = 0, max_disp = 300, downsampling_trick_max_kernel_size=11,
                         from_stage = "disparity_map", differenciable=False
                         ):
    """
    Model that refocuses at the coarsest level of the cost volume and then tryies
    to upsample with residual learning the blured image.
    
    left_image, right_image: placeholder for BxNxMx3 images
    target_disparities: single or list of focus planes to refocus with
    blur_magnitude: virtual aperture
    min_disp max_disp = 300: min and max possible disparity (determine kernel sizes)
    downsampling_trick_max_kernel_size: set a maximum kernel size to be used, if overflowed, downsampling will be used.
    from_stage: uses the disparity map at full resolution or the cost volume to perform refocusing.
    
    Returns: a list of refocused images, the disparity or cost volume placeholders.
    """
    #FIXME: bug when is_training=False
    disparity, intermediate_steps  = stereonet(left_image, right_image, is_training=True)#is_training=(not stop_grads))
    cost_volume = intermediate_steps["cost_volume_left_view"]
    
    refocus_images = []
    intermediate_result = []
    if from_stage == "disparity_map":
        if stop_grads:
            disparity = tf.stop_gradient(disparity)

        for target_disparity in target_disparities:
            refocus_image = layered_bluring(left_image, disparity, target_disparity,blur_magnitude, 
                                    min_disp, max_disp,downsampling_trick_max_kernel_size,
                                    differenciable=differenciable)
            refocus_images.append(refocus_image)

        return refocus_images, disparity, intermediate_steps
    elif from_stage == "cost_volume":
        
        if stop_grads:
            cost_volume = tf.stop_gradient(cost_volume)

        #need to normalise cost volume, note: can play with beta
        #note sofmtin turn it into a confidence volume
        beta = 1
        conf_volume = tf.nn.softmax(-beta*cost_volume, dim=-1)

        #this if given by stereonet architecture
        disparity_range = np.arange(1,18+1)*8#FIXME: see ben for disp=0

        refocus_images = []
        for target_disparity in target_disparities:
            refocus_image = layered_bluring_from_cost_volume(left_image, conf_volume, disparity_range,
                                                             target_disparity, blur_magnitude, 
                                                             downsampling_trick_max_kernel_size,
                                                             differenciable=differenciable)
            refocus_images.append(refocus_image)

        return refocus_images, conf_volume, intermediate_steps
    else:
        raise BaseException("Stage type not understood. Needs to be 'disparity_map' or 'cost_volume' not '%s'"%from_stage)

