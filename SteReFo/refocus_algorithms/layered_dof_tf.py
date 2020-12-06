# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""
Collection of several differenciable refocusing algorithms to refocus from several modalities.
"""

import tensorflow as tf
import numpy as np
 
def is_lower(x, y, differenciable=False, approx_coef=1000):
    """
    Return a tensor with 1 if x<=y and 0 otherwise.
    
    differenciable: will use a numerical apporximation of the heaviside setp fumction to make x<=y differenciable.
    approx_coef: controls the sharpness of the step function approximation
    """
    if differenciable:
        #approx of the heaviside function
        #https://en.wikipedia.org/wiki/Heaviside_step_function#Analytic_approximations
        return (tf.tanh(approx_coef*(y-x))+1)/2
    else:
        #non differentiable
        return x<=y

def do_radial_blur(image, radius, downsampling_trick_max_kernel_size=None, differenciable=False):
    """
    Apply a radial kernel blurring, with fixed diameter.
    
    image: input image
    radius: radius of the disk kernel to use. Can be float.
    downsampling_trick_max_kernel_size: if not none, will downsample the image and kernel
    to apply the convolution on low resolution. Upsampling is then carried out.
    """
    with tf.variable_scope("blur_radius_%f"%radius):
        b,h,w,c = image.shape
        #Use the smallest kernel size possible for each conv
        rounded_radius = int(radius)+1
        kernel_size = 2*rounded_radius+1
        tf.logging.info("Bluring with radius size %f kernel size %d"%(radius,kernel_size))
        #detect if we should downsample or not
        do_trick = (downsampling_trick_max_kernel_size is not None) and (kernel_size > downsampling_trick_max_kernel_size)
        #precompute infos if we use the downsampling trick
        if do_trick:
            assert(downsampling_trick_max_kernel_size%2!=0)
            tf.logging.info("Downsampling to match max radius size %d"%downsampling_trick_max_kernel_size)
            downsampling_factor = downsampling_trick_max_kernel_size/float(kernel_size)
            #FIXME: not really correct? rounding issue
            size = [int(int(h)*downsampling_factor),int(int(w)*downsampling_factor)]
            image=tf.image.resize_bilinear(image,size)
            #also alter the radius, kenrel size etc
            rounded_radius = int((downsampling_trick_max_kernel_size-1)/2)
            radius = radius*downsampling_factor
            kernel_size = 2*rounded_radius+1
        #build the kernel
        xs,ys = np.meshgrid(range(-rounded_radius,rounded_radius+1), range(-rounded_radius,rounded_radius+1))
        #compute where the kernel should be 0 and 1
        #note: that is non differenciable but we dont care because radius and coordinates are known at compile time (they arent variables)
        kernel = ((xs**2+ys**2)**0.5<=radius).astype(np.float32)
        #normalise it
        kernel = kernel/np.sum(kernel)
        #tile for the batch and channel
        kernel = np.reshape(kernel,[kernel_size,kernel_size,1,1])
        #convolve the image by the kernel (for each channel of the image)
        blur_img = []
        for chan in range(c):
            blur_img.append(tf.nn.conv2d(tf.expand_dims(image[:,:,:,chan],-1),
                             filter=kernel,
                             strides=[1, 1, 1, 1],
                             padding='SAME'))
        blur_img = tf.concat(blur_img,-1)
        #upsamples result if needed
        if do_trick:
            tf.logging.info("Upsampling")
            blur_img = tf.image.resize_bilinear(blur_img,[h,w])
    #return the good stuff
    return blur_img    

def lowres_layered_bluring_from_lowres_cost_volume(image, cost_volume, disparity_range,
                                                 target_disparity,blur_magnitude
                                                 ):
    """
    Apply a depth-variying radial blur to an dowsampled image from the low resolution cost volume, using a disk kenrel.
    Sweeps the scene from back to front, using the disparity slices of the cost volume, 
    get the correponding zones in the cost volume,
    apply a single blur kernel on this zones, integarate by plane sweep all the 
    blured partial images.
    
    image: a batch on images of size BxNxMxC
    cost_volume: the coresponding cost volume of size BxnxmxD, potentially much smaller that the input image
    disparity_range: the disparty value for each slice of the cost volume
    target_disparity: the focus plane
    """
    with tf.variable_scope("refocus_cost_volume"):
        
        with tf.variable_scope("depth_slices_computation"):
            _, h,_, _ = image.shape
            _, hc,wc, _ = cost_volume.shape
            
            target_downsampling_factor = int(hc)/float(int(h))
            
            target_disparity = target_disparity/8.0
       
            dowsampled_image = tf.image.resize_bilinear(image,[hc,wc])
         
            blur_masked_images = []
            blur_masks = []
            for d_idx in range(len(disparity_range)):
                d=disparity_range[d_idx]
                #blur radius and equivalent kernel size we are supposed to blur with
                radius = blur_magnitude*abs(d-target_disparity)
                #equivalent radius in the low res space of the cost volume
                radius = radius*target_downsampling_factor
                print("Bluring lowres with radius %f"%radius)
         
                conf_slice = 1-tf.expand_dims(cost_volume[:,:,:,d_idx] ,-1)
                
                #mask image with cost slice
                masked_image = tf.tile(conf_slice,[1,1,1,3])*dowsampled_image
                
                #blur both mask and masked image
                blur_masks.append(do_radial_blur(conf_slice, radius, None))
                blur_masked_images.append(do_radial_blur(masked_image, radius, None))
              
        with tf.variable_scope("image_composition"):
            tf.logging.info("Image composition")
            #composes the final image (this cannot be done in parallell)
            integrated_images = tf.zeros_like(dowsampled_image)
            integrated_masks = tf.zeros_like(dowsampled_image)
            for d in range(len(disparity_range)):
                integrated_images=integrated_images*(1-blur_masks[d])+blur_masked_images[d]
                integrated_masks=integrated_masks*(1-blur_masks[d])+blur_masks[d]
    
        #normalise and return
        return integrated_images/integrated_masks
    
def layered_bluring_from_cost_volume(image, conf_volume, disparity_range,
                                     target_disparity,blur_magnitude,
                                     downsampling_trick_max_kernel_size=None, differenciable=False
                                     ):
    """
    Apply a depth-variying radial blur an images using upsampled silces of a cost volume.
    to an dowsampled image from the low resolution cost volume, using a disk kenrel.
    Sweeps the scene from back to front, upsample the cost volume slice according to the blur that needs to be applied,
    apply a single blur kernel on this zones, integarate by plane sweep all the 
    blured partial images.
    
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
            
            blur_masked_images = []
            blur_masks = []
            for d_idx in range(len(disparity_range)):
                d=disparity_range[d_idx]
                print("Disparity slice %f"%d)
                #blur radius and equivalent kernel size we are supposed to blur with
                radius = blur_magnitude*abs(d-target_disparity)
                rounded_radius = int(radius)+1
                kernel_size = 2*rounded_radius+1
                #turn the cost into a conf
                conf_slice = tf.expand_dims(conf_volume[:,:,:,d_idx] , -1)
                
                #upample the cost volume to full res
                #if bellow kernel size, otherwise, just upsamples to match the max kernel size
                upsample_fullres = kernel_size <= downsampling_trick_max_kernel_size

                if upsample_fullres:
                    #if target kernel size bellow threshold, upsample full scales
                    conf_slice=tf.image.resize_bilinear(conf_slice, [h,w])#FIXME:just bilinear upsampling, this can be done with a net
                    resized_img = image
      
                else:
                    #if not only upsample accordingly and adapt kernel
                    #size of the image we need for eaquivalent blur with kernel size downsampling_trick_max_kernel_size
                    target_downsampling_factor = downsampling_trick_max_kernel_size/float(kernel_size)
                    target_size = (int(np.round(h*target_downsampling_factor)),
                                   int(np.round(w*target_downsampling_factor)))
                    conf_slice=tf.image.resize_bilinear(conf_slice, target_size)#FIXME:just bilinear upsampling
                    resized_img = tf.image.resize_bilinear(image, target_size)#this is ok, just downsampling
                    radius = radius*target_downsampling_factor
                    kernel_size = downsampling_trick_max_kernel_size
                    
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
    
        #normalise and return FIXME: mind the division per 0
        return integrated_images/integrated_masks

        
def layered_bluring(image, disparity, 
                    target_disparity,blur_magnitude, 
                    min_disp = 0, max_disp = 300,
                    downsampling_trick_max_kernel_size=11,
                    differenciable=False, return_intermediate=False):
    """
    Apply a depth-variying radial blur to an image, given its disparity, using a disk kenrel.
    Sweeps the scene from back to front, get the zones near the sweept depth plane,
    apply a single blur kernel on this zones, integarate by plane sweep all the 
    blured partial images.
    Note: if the disparity is outside of the rpovided range, it is clipped
    
    image: a batch on images of size BxNxMxC
    disparity: the coresponding disparity map BxNxMx1
    target_disparity: the focus plane
    min_disp, max_disp: the disparity range to consider
    downsampling_trick_max_kernel_size: when this is not None, when a blur with
    a kernel of size greater than this paraneter is goig to be applied, the input 
    image is downampled and aplied an equivalent blur kernel with smaller radius.
    Greatly reduces the temporal and spacial complexity.
    """
    with tf.variable_scope("refocus_%f"%target_disparity):
        #Compute the disparity step
        disp_blur_step = 1.0/blur_magnitude
        #Initialise accumalation arrays
        integrated_images = tf.zeros_like(image)
        integrated_masks = tf.zeros_like(image)#
        disparity_range = np.arange(min_disp, max_disp, disp_blur_step)
        disparity = tf.clip_by_value(disparity, min_disp, max_disp)
        tf.logging.info("Refocusing with disparity range %f %f"%(min_disp, max_disp))
        
        #FIXME: background in focus object are not bleeding over foreground out of focus objects
        #this happens because we dont have the occluded pixels :(
        #FIXME: mask computation can be done in parallel, only the clipping is done sequencialy, make sure it is done in parallell
        #FIXME: graph static with respect of the blur intensity and focus parameter  
        with tf.variable_scope("depth_slices_computation"):
            blur_masked_images = []
            blur_masks = []
            #for each disparity slice, compute the corresponding mask and blured partial image
            for d in disparity_range :
                with tf.variable_scope("depth_slice_%f"%d):
                    tf.logging.info("Disparity slice %f"%d)
                    #get the mask of pixels inside the are we want to blur at the current disparity level
                    in_range_mask = tf.to_float(is_lower(tf.abs(disparity-d), disp_blur_step, differenciable))#NOTE: here we use the differenciability trick
                    #extract corresponding area in the image
                    masked_image = image*in_range_mask
                    #get the corresponding blur radius 
                    radii = blur_magnitude*abs(d-target_disparity)
                    #blur the mask and the masked image
                    blur_mask = do_radial_blur(in_range_mask, radii, downsampling_trick_max_kernel_size)#Note: the kernel size is known at compile time so we dont care if its non diffeenciable
                    blur_masked_image = do_radial_blur(masked_image, radii, downsampling_trick_max_kernel_size)
                    #saves mask and image for later
                    blur_masked_images.append(blur_masked_image)
                    blur_masks.append(blur_mask)
            
        with tf.variable_scope("image_composition"):
            tf.logging.info("Image composition")
            #composes the final image (this cannot be done in parallell)
            integrated_images = tf.zeros_like(image)
            integrated_masks = tf.zeros_like(image)
            for d in range(len(disparity_range)):
                integrated_images=integrated_images*(1-blur_masks[d])+blur_masked_images[d]
                integrated_masks=integrated_masks*(1-blur_masks[d])+blur_masks[d]

    #normalise and return
    #normalise and return FIXME: mind the division per 0, shoould not happen tho
    if not return_intermediate:
        return integrated_images/integrated_masks
    else:
        intermediate_steps = {}
        intermediate_steps["blur_masked_images"] = blur_masked_images
        intermediate_steps["blur_masks"] = blur_masks
        intermediate_steps["integrated_images"] = integrated_images
        intermediate_steps["integrated_masks"] = integrated_masks
        return integrated_images/integrated_masks, intermediate_steps
