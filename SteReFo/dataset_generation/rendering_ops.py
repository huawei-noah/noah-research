import numpy as np
from scipy import signal
from utils import resize

def clip_disparities(disparities):
    """
    Generates the disparity map by occlusion clipping.
    """
    disparities = np.stack(disparities, -1)
    return np.amax(disparities, axis=-1)

def clip_images(images, disparities, alpha_matting=True):
    """
    Generates the final image by occlusion clipping.
    Can use the alpha maks, is so, make sure the disparity is not 0 when alpha is mot 0.
    This is differnet from you final disparity.
    """
    nb_im  = len(images)
    disparities = np.stack(disparities, -1)
    images =   np.stack(images, -1)

    if not alpha_matting: #no matting in disparity
        final_disp = np.amax(disparities, axis=-1)
        masks = disparities==np.tile(np.expand_dims(final_disp,-1), [1,1,nb_im])
    else:
        masks = []
        for i in range(nb_im):
            alpha = images[:,:,3,i]/255.0
            for j in range(nb_im):
                is_in_front =  disparities[:,:, i]<disparities[:,:, j] 
                #leaving that here to show how retarded i am, lost a day because i forgot that alpha was multiplicative and not aditive
                #alpha -= (images[:,:,3,j]/255.0)*is_in_front
                alpha *= (1-(images[:,:,3,j]/255.0)*is_in_front)
            masks.append(alpha)#remove values <0 (i e occluded by several objects)
        masks = np.stack(masks, -1)
    #expand for color channel and sum
    final_image = np.sum(images*np.tile(np.expand_dims(masks, 2), [1,1,4,1]), axis=-1)
    return final_image[:,:,0:3], masks

def gen_disp_map(image, disparity_plane, oppacity_threshold = 0 ):
    """
    Get corresponding diaprity map from a image with apha, all zones outside will be 0.
    Note that non zero aplha will be treated as opaque.
    
    image: a nxmx4 image with alpha either a pil image or np array
    disparity_plane: a single disparity plane for the object
    """
    valid_px = np.asarray(image)[:,:,3]>oppacity_threshold*255
    return disparity_plane*(valid_px.astype(np.float32))

def do_radial_blur(image, radius, downsampling_trick_max_kernel_size=None, interpolation=False):
    """
    CPU version of appyling a radial blur. Uses scipy
    """
    #Use the smallest kernel size possible
    rounded_radius = int(radius)+1
    kernel_size = 2*rounded_radius+1
    print("Bluring with kernel size %d"%kernel_size)
    
    do_trick = (downsampling_trick_max_kernel_size is not None) and (kernel_size > downsampling_trick_max_kernel_size)

    if do_trick:
        assert(downsampling_trick_max_kernel_size%2!=0)
        print("*Will do blur with kernel size %d instead"%(downsampling_trick_max_kernel_size))
        downsampling_factor = downsampling_trick_max_kernel_size/float(kernel_size)
        print("*Image downsampled by ratio %f "%(downsampling_factor))
        orig_size = image.shape[0:2]
        print(image.shape)
        image = resize(image, downsampling_factor) 
        print(image.shape)
        print("*It is of size %d %d now"%(image.shape[0], image.shape[1]))
        rounded_radius = int((downsampling_trick_max_kernel_size-1)/2)
        radius = radius*downsampling_factor
        print("*New radius %f"%radius)

    #build the kernel
    xs,ys = np.meshgrid(range(-rounded_radius,rounded_radius+1), range(-rounded_radius,rounded_radius+1))
    if not interpolation:
        kernel = ((xs**2+ys**2)**0.5<=radius).astype(np.float32)
    else:
        raise BaseException("Not supported yet")

    #normalise it
    kernel = kernel/np.sum(kernel)
    #return the conv2 of the image comvoved with the kernel
    if len(image.shape) == 2:
        blur_img = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    elif len(image.shape) == 3:
        rimage = []
        for c in range(image.shape[2]):
            rimage.append(signal.convolve2d(image[:,:,c], kernel, boundary='symm', mode='same'))
        blur_img = np.stack(rimage, -1)
    else:
        raise BaseException("Wrong rank")

    if do_trick:
        print("*Upsampling")
        blur_img = resize(blur_img, orig_size)  
 
    return blur_img   


def refocus_noleak(images, disparities,
                    blur_magnitude, target_disparity, downsampling_trick_max_kernel_size,
                    background_idx = None, do_alpha = True
                    ):
    """
    Do the refocusing with leak correction for the objects.
    Note: not doing it for the backgroud so you still have intensity leakage.    
    """   
    if background_idx is None:
        background_idx = len(images)-1
        
    #FIXME: how to make sure we have occlusion? How to hanfle missing values in real data (can assune missing but always background?)
    #refocus image with classical layered DOF
    min_disp = int(np.floor(np.amin(disparities)))
    max_disp = int(np.ceil(np.amax(disparities)))
    disp_blur_step = 1.0/blur_magnitude

    integrated_images = np.zeros_like(images[0][:,:,0:3])
    integrated_masks = np.zeros_like(images[0][:,:,0])
    
    depth_range = np.arange(min_disp, max_disp, disp_blur_step)
    
    max_radius= np.ceil(np.amax(blur_magnitude*(depth_range-target_disparity)))
    print("Maximum possible radius %d"%max_radius)

    #for each disparity slice
    #FIXME: problem with that, the foreground object are clipped over backgroud not sure why, might be the order of the rendering. seems to only happen when they are on the same depth slice.
    #FIXME: I think an object need to be on a single depth plane for this to work
    for d_idx in range(len(depth_range)):
        d=depth_range[d_idx]
        print("Disparity slice %f"%d)
           
        for i in range(len(images)):
            in_range_mask=(np.abs(disparities[i]-d) <= disp_blur_step).astype(np.float32)
    
            #misc.imsave("out/mask_%d_%f.png"%(i,d), in_range_mask)
             #do clipping but only inside of a specific depth plane that means no dissoclusion in objects in the same depth plane but prevent transparent background artifacts
            #check if occluded by background (needed to avoid transparency in the final imnage, asssumes background)
            if i != background_idx :
                is_occluded = (disparities[i] <= disparities[background_idx])
                in_range_mask *= (1-is_occluded)
         
            #misc.imsave("out/mask_%d_%f_AFTER.png"%(i,d), in_range_mask)
            masked_image = images[i]*np.stack([in_range_mask,in_range_mask,in_range_mask, in_range_mask],-1)
           
            #get the corresponding blur radius 
            radii = blur_magnitude*abs(d-target_disparity)
            #blur the mask and the masked image
            blur_mask = do_radial_blur(in_range_mask, radii, downsampling_trick_max_kernel_size)
            blur_masked_image = do_radial_blur(masked_image, radii, downsampling_trick_max_kernel_size)
    
            #filter with aplha channel to handle transparency
            if do_alpha:
                alpha = blur_masked_image[:,:,3]/255.0
                blur_mask = blur_mask*alpha
                blur_masked_image = np.stack([alpha,alpha,alpha], -1)*blur_masked_image[:,:,0:3]
                
            #integrate the blured image and the mask, note the alpha goes in the mask
            integrated_images=integrated_images*np.stack([(1-blur_mask),(1-blur_mask),(1-blur_mask)], -1)+blur_masked_image[:,:,0:3]
            integrated_masks=integrated_masks*(1-blur_mask)+blur_mask
           
    #sanity check         

# =============================================================================
#     misc.imsave("out/integrated_masks_bellow_1_%f.png"%(d),((integrated_masks<0.8)*255).astype(np.uint8))
#     misc.imsave("out/integrated_masks%f.png"%(d),((integrated_masks/np.amax(integrated_masks))*255).astype(np.uint8))
# =============================================================================
    #misc.imsave("out/integrated_masks%f.png"%(d),((np.clip(integrated_masks,0,1))*255).astype(np.uint8))
     
    final_image =  integrated_images/np.stack([integrated_masks,integrated_masks,integrated_masks],-1)

    #normalise and return
    return final_image
    