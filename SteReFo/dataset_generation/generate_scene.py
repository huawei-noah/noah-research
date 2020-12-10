#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
"""
Script that generates a stereo dataset by inserting segmented objects into an existing rgb-d scene.
As of now uses a donwload colleciton of objecst from pixabay and the sceneflow dataset.
Mixing syhtetic and real data does not seem like a good idea tho but the lack of dataset with complte disparoty map is very problematic.
"""
#%% 
#FIXME: fix interpolation when doing the stereo translation
#TODO: Corrupt data?
#FIXME: find actual background rgb-d images?
#FIXME: Never let background occluded foreground?
import random
from PIL import Image
import numpy as np
from utils import readPFM, writePFM
from utils import pad_to_size, crop_to_size
from rendering_ops import gen_disp_map, clip_images, clip_disparities, refocus_noleak
#from scipy import misc

#%% Hard params, deal with it, too many for a command line interface, may consider a text config file
#how many scenes to generate
nb_scenes = 10000
#resume at a specific index
resume_idx=0
#nb objetcs per scene
max_nb_objects = 2
min_nb_objects = 1
#scale and rotation
max_scale_objects = 1
min_scale_objects = 0.5
max_object_rotation = 45
min_object_rotation = -45
#min max disp for the depth planes
# =============================================================================
# max_disp_objects = 300
# min_disp_objects = 150
# =============================================================================
max_disp_objects = 100
min_disp_objects = 30
#reative position from the center of the image
min_x_objects=-0.5
min_y_objects=-0.5
max_x_objects=0.5
max_y_objects=0.5
#disparity aplha threshold
disp_alpha = 0.5
#refocus params, focus on backgroud to generate the defocus dataset
# =============================================================================
# focus_plane = 100
# aperture = 0.3 
# =============================================================================
focus_plane = 20
aperture = 0.3 
#inner params
canvas_width = 2000
canvas_heigt = 2000
interp_mode=Image.BICUBIC

#%% Rendering function
def gen_scene_stereo_refocus(input_seg_obj_list, input_stereo_left_list, input_stereo_right_list,
                          input_stereo_left_disp_list, input_stereo_right_disp_list,
                          validity_test_function=None):
    """
    Generates a random scene and render lef right stereo views, disparity map, objects masks, and refocused image. No check on the final scene is performed.
    """
    assert(len(input_stereo_left_list)==len(input_stereo_right_list)==len(input_stereo_left_disp_list)==len(input_stereo_right_disp_list))
    #open background image and disparity
    background_idx = random.randint(0,len(input_stereo_left_list)-1)
    background_image_left = np.asarray(Image.open(input_stereo_left_list[background_idx]))
    background_image_right = np.asarray(Image.open(input_stereo_right_list[background_idx]))
    
    target_h = background_image_left.shape[0]
    target_w = background_image_right.shape[1]
    if background_image_left.shape[2] == 3:
        background_image_left = np.concatenate([background_image_left, 255*np.ones([target_h, target_w,1])], -1)
    if background_image_right.shape[2] == 3:
        background_image_right = np.concatenate([background_image_right, 255*np.ones([target_h, target_w,1])], -1)
        
    background_disparity_left = readPFM(input_stereo_left_disp_list[background_idx])
    background_disparity_right = readPFM(input_stereo_right_disp_list[background_idx])
    
    #generate randomness for objects
    nb_objects= random.randint(min_nb_objects,max_nb_objects)
    objects_scales = [random.uniform(min_scale_objects,max_scale_objects) for x in range(nb_objects) ]
    objects_idxs = [random.randint(0,len(input_seg_obj_list)-1) for x in range(nb_objects) ]
    objects_disparities = [random.uniform(min_disp_objects,max_disp_objects) for x in range(nb_objects) ]
    objects_dx = [round(target_w*random.uniform(min_x_objects,max_x_objects)) for x in range(nb_objects) ]
    objects_dy = [round(target_h*random.uniform(min_y_objects,max_y_objects)) for x in range(nb_objects) ]#FIXME: do integer translation?
    objects_rot = [random.randint(min_object_rotation,max_object_rotation) for x in range(nb_objects) ]

    #open and transform the images
    object_images_left = []
    object_disparity_map_left = []
    object_disparity_map_noaplha_left = []
    object_images_right = []
    object_disparity_map_right = []
    for i in range(nb_objects):
        object_image = Image.open(input_seg_obj_list[objects_idxs[i]])
        #object_image.save("out/ORIG_%d.png"%i)
        #Image.fromarray(np.asarray(object_image)[:,:,3]).save("alpha_%d.png"%i)
        #resize image 
        object_image = object_image.resize((int(object_image.size[0]*objects_scales[i]), 
                             int(object_image.size[1]*objects_scales[i])))
        #object_image.save("out/RESIZE_%d.png"%i)
        #pad to canvas image size so we dont have to worry about corpping accidentatluy things
        object_image = pad_to_size(object_image,(canvas_width, canvas_heigt))
        #object_image.save("out/CROPPAD_%d.png"%i)
        #rotate it 
        object_image = object_image.rotate(objects_rot[i], resample=interp_mode, expand=False)
        #object_image.save("out/ROTATE_%d.png"%i)
        #do the translation from the center of the image (do it separately for left and right to avoid interpolation twice)
        object_image_left = object_image.rotate(0, resample=interp_mode, translate=(objects_dx[i], objects_dy[i]))
        #object_image_left.save("out/object_image_left_%d.png"%i)
        #crop back to the sive of the image
        object_image_left = crop_to_size(object_image_left, (target_w, target_h), canvas_width, canvas_heigt)
        #FIXME: probably need to be corrupted with noise or something, qs can be deformed i guess
        object_images_left.append(np.asarray(object_image_left))
        #compute diaprity map
        object_disparity_map = gen_disp_map(object_image_left, objects_disparities[i])#TODO:can maybe generate random transforms for depth instead of constant values
        object_disparity_map_noalpha = gen_disp_map(object_image_left, objects_disparities[i], disp_alpha)
        object_disparity_map_noaplha_left.append(object_disparity_map_noalpha)
        object_disparity_map_left.append(object_disparity_map)
        #Image.fromarray((255*object_disparity_map/max_disp_objects).astype(np.uint8)).save("displ_tmp_%d.png"%i)
        #translate with disparity (assumes rectified images), not this crops the out of image values
        object_image_right = object_image.rotate(0, resample=interp_mode, translate=(objects_dx[i]+objects_disparities[i], objects_dy[i]))
        #crop back to the sive of the image
        object_image_right = crop_to_size(object_image_right, (target_w, target_h), canvas_width, canvas_heigt)
        object_images_right.append(np.asarray(object_image_right))
        #object_image_right.save("object_images_right_%d.png"%i)
        #compute diaprity map
        object_disparity_map = gen_disp_map(object_image_right, objects_disparities[i])#TODO: can maybe generate random transforms for depth instead of constant values
        object_disparity_map_right.append(object_disparity_map)
        #Image.fromarray((255*object_disparity_map/max_disp_objects).astype(np.uint8)).save("dispr_tmp_%d.png"%i)
        
    if validity_test_function is not None:
        return gen_scene_stereo_refocus(input_seg_obj_list, input_stereo_left_list, input_stereo_right_list,
              input_stereo_left_disp_list, input_stereo_right_disp_list,
               validity_test_function=None)
    else:
        #render by clipping
        synth_left_image, masks_left = clip_images(object_images_left+[background_image_left],   object_disparity_map_left+[background_disparity_left])
        synth_right_image, masks_right = clip_images(object_images_right+[background_image_right], object_disparity_map_right+[background_disparity_right])
        synth_left_disp = clip_disparities(object_disparity_map_noaplha_left+[background_disparity_left] )
        
        #refocus on the individual obhect in the right before the background?
        downsampling_trick_max_kernel_size = 11
        synth_left_refocus= refocus_noleak(object_images_left+[background_image_left], 
                                           object_disparity_map_left+[background_disparity_left], 
                                           aperture, focus_plane, 
                                           downsampling_trick_max_kernel_size)
        
        #return new disp, lef right images, eventually masks, and occluded masks and disparity (could be usefull to learn disocclusion)
        return synth_left_image, synth_right_image, synth_left_disp, synth_left_refocus, masks_left, masks_right
    
def gen_scene_refocus(input_seg_obj_list, input_background_list, input_background_disp_list):
    assert(len(input_stereo_left_list)==len(input_stereo_right_list)==len(input_stereo_left_disp_list)==len(input_stereo_right_disp_list))
    #open background image and disparity
    background_idx = random.randint(0,len(input_stereo_left_list)-1)
    background_image = np.asarray(Image.open(input_stereo_left_list[background_idx]))
    
    
    #resize
   # background_image = background_image[0:1000,:,:]
  
    target_h = background_image.shape[0]
    target_w = background_image.shape[1]
    if background_image.shape[2] == 3:
        background_image = np.concatenate([background_image, 255*np.ones([target_h, target_w,1])], -1)
    
    background_disparity = readPFM(input_stereo_left_disp_list[background_idx])
    
    #
    #background_disparity = background_disparity[0:1000,:]
     #rescale disparity 
    print(np.amax(background_disparity))
    background_disparity = 0.1*background_disparity
    print(np.amax(background_disparity))
    #pre process backgroud disparity
   
    #generate randomness for objects
    nb_objects= random.randint(min_nb_objects,max_nb_objects)
    objects_scales = [random.uniform(min_scale_objects,max_scale_objects) for x in range(nb_objects) ]
    objects_idxs = [random.randint(0,len(input_seg_obj_list)-1) for x in range(nb_objects) ]
    objects_disparities = [random.uniform(min_disp_objects,max_disp_objects) for x in range(nb_objects) ]
    objects_dx = [round(target_w*random.uniform(min_x_objects,max_x_objects)) for x in range(nb_objects) ]
    objects_dy = [round(target_h*random.uniform(min_y_objects,max_y_objects)) for x in range(nb_objects) ]#FIXME: do integer translation?
    objects_rot = [random.randint(min_object_rotation,max_object_rotation) for x in range(nb_objects) ]

    #open and transform the images
    object_images = []
    object_disparity = []
    object_disparity_map_noaplha = []
    for i in range(nb_objects):        
        object_image = Image.open(input_seg_obj_list[objects_idxs[i]])
        #resize image 
        object_image = object_image.resize((int(object_image.size[0]*objects_scales[i]), 
                             int(object_image.size[1]*objects_scales[i])))
        #pad to canvas image size so we dont have to worry about corpping accidentatluy things
        object_image = pad_to_size(object_image,(canvas_width, canvas_heigt))
        #object_image.save("out/CROPPAD_%d.png"%i)
        #rotate it 
        object_image = object_image.rotate(objects_rot[i], resample=interp_mode, expand=False)
        #object_image.save("out/ROTATE_%d.png"%i)
        #do the translation from the center of the image (do it separately for left and right to avoid interpolation twice)
        object_image = object_image.rotate(0, resample=interp_mode, translate=(objects_dx[i], objects_dy[i]))
        #object_image_left.save("out/object_image_left_%d.png"%i)
        #crop back to the sive of the image
        object_image = crop_to_size(object_image, (target_w, target_h), canvas_width, canvas_heigt)
        #FIXME: probably need to be corrupted with noise or something, qs can be deformed i guess
        object_images.append(np.asarray(object_image))
    
        #compute disparity map
        object_disparity_map = gen_disp_map(object_image, objects_disparities[i])#TODO:can maybe generate random transforms for depth instead of constant values
        object_disparity_map_noalpha_tmp = gen_disp_map(object_image, objects_disparities[i], disp_alpha)
        object_disparity_map_noaplha.append(object_disparity_map_noalpha_tmp)
        object_disparity.append(object_disparity_map)
    
    #render by clipping
    synth_image, masks = clip_images(object_images+[background_image], object_disparity+[background_disparity])
    synth_disp = clip_disparities(object_disparity_map_noaplha+[background_disparity] )
    
    #refocus on the individual obhect in the right before the background?
    downsampling_trick_max_kernel_size = 11
    synth_refocus = refocus_noleak(object_images+[background_image], 
                                  object_disparity+[background_disparity], 
                                  aperture, focus_plane, 
                                  downsampling_trick_max_kernel_size)
    
    #return new disp, left right images, eventually masks, and occluded masks and disparity (could be usefull to learn disocclusion)
    return synth_image, synth_disp, synth_refocus, masks

#%% Actual rendering script
if __name__ == "__main__":
    import os

    def listdir_fullpath(d):
        return sorted([os.path.join(d, f) for f in os.listdir(d)])
    
    #on aurora
    output_folder = "/home/matthieu/kgx_nfs2/data/internal/synthetic_blur/train"
    #pixabay dataset
    segmentation_root_path = "/home/matthieu/kgx_nfs2/data/external/pixabay/" #aurora
    #uses driving from sceneflow from now
    stereo_root_path = "/home/matthieu/kgx_nfs2/data/external/sceneflow"
    bullshit_path = "35mm_focallength/scene_forwards/slow"
    input_stereo_left_list = listdir_fullpath(stereo_root_path+"/frames_cleanpass/"+bullshit_path+"/left")
    input_stereo_right_list = listdir_fullpath(stereo_root_path+"/frames_cleanpass/"+bullshit_path+"/right")
    input_stereo_left_disp_list = listdir_fullpath(stereo_root_path+"/disparity/"+bullshit_path+"/left")
    input_stereo_right_disp_list = listdir_fullpath(stereo_root_path+"/disparity/"+bullshit_path+"/right")
    #remove comsecutive frames
    frame_step = 5
    n = len(input_stereo_left_list)
    input_stereo_left_list = input_stereo_left_list[0: n:frame_step]
    input_stereo_right_list = input_stereo_right_list[0: n:frame_step]
    input_stereo_left_disp_list = input_stereo_left_disp_list[0: n:frame_step]
    input_stereo_right_disp_list = input_stereo_right_disp_list[0: n:frame_step]
    print("%d stereo objects detecected"%len(input_stereo_left_list))
    
# =============================================================================
#     #test on local machines
#     output_folder = "./out"
#     segmentation_root_path = "D:/dataset_objects/" #local machine
# # =============================================================================
# #     input_stereo_left_list = [r"C:\Users\m00487515\Desktop\0500-left.png"]
# #     input_stereo_right_list = [r"C:\Users\m00487515\Desktop\0500-right.png"]
# #     input_stereo_left_disp_list = [r"C:\Users\m00487515\Desktop\0500-left.pfm"]
# #     input_stereo_right_disp_list = [r"C:\Users\m00487515\Desktop\0500-right.pfm"]
# # =============================================================================
#     input_stereo_left_list = [r"C:\Users\m00487515\Desktop\0800-left.png"]
#     input_stereo_right_list = [r"C:\Users\m00487515\Desktop\0800-right.png"]
#     input_stereo_left_disp_list = [r"C:\Users\m00487515\Desktop\0800-left.pfm"]
#     input_stereo_right_disp_list = [r"C:\Users\m00487515\Desktop\0800-right.pfm"]
# =============================================================================
    

    categories =  ["fashion", "nature", "backgrounds", "science", "education", 
                   "people", "feelings", "religion", "health", "places", "animals",
                   "industry", "food", "computer", "sports", "transportation", 
                   "travel", "buildings", "business", "music"]
    input_seg_obj_list = [listdir_fullpath( segmentation_root_path+"/"+c) for c in categories  ]
    #flatten the lists
    input_seg_obj_list = [y for x in input_seg_obj_list for y in x]
    print("%d segmented objects detecected"%len(input_seg_obj_list))
    
    scenes_idxs=range(resume_idx, nb_scenes)
    
# =============================================================================
# #stereo
#     if resume_idx == 0:
#         os.makedirs(output_folder+"/left")
#         os.makedirs(output_folder+"/right")
#         os.makedirs(output_folder+"/disparity")
#         os.makedirs(output_folder+"/refocus")
#         os.makedirs(output_folder+"/refocus_params")
#         os.makedirs(output_folder+"/masks_left")
#         os.makedirs(output_folder+"/masks_right")
# 
#     #wrapper
#     def do_gen(i):
#         print("Generation scene %d"%i)
#         synth_left_image, synth_right_image, synth_left_disp, synth_left_refocus, masks_left, masks_right  = gen_scene_stereo_refocus(input_seg_obj_list, input_stereo_left_list, 
#                                                                                                                               input_stereo_right_list, input_stereo_left_disp_list, 
#                                                                                                                               input_stereo_right_disp_list)
#         Image.fromarray(synth_left_image.astype(np.uint8)).save(output_folder+"/left/%05d.png"%i)
#         Image.fromarray(synth_right_image.astype(np.uint8)).save(output_folder+"/right/%05d.png"%i)
#         Image.fromarray(synth_left_refocus.astype(np.uint8)).save(output_folder+"/refocus/%05d.png"%i)
#         f= open(output_folder+"/refocus_params/%05d.csv"%i,"w+")
#         f.write("focus parameter:,%f\n"%focus_plane)
#         f.write("aperture,%f"%aperture)
#         f.close()
#         #writePFM(output_folder+ "/disparity/%05d.pfm"%i, synth_left_disp, scale=1)
#         np.save(output_folder+ "/disparity/%05d.npy"%i, synth_left_disp)
#         Image.fromarray((255*synth_left_disp/np.amax(synth_left_disp)).astype(np.uint8)).save(output_folder+"/disparity/%05d_preview.png"%i)
#         for o in range(masks_left.shape[2]):
#             Image.fromarray((255*masks_left[:,:,o]).astype(np.uint8)).save(output_folder+"/masks_left/%05d_%d.png"%(i,o))
#             Image.fromarray((255*masks_right[:,:,o]).astype(np.uint8)).save(output_folder+"/masks_right/%05d_%d.png"%(i,o))
# =============================================================================
            
    #refocus only
    if resume_idx == 0:
        os.makedirs(output_folder+"/image")
        os.makedirs(output_folder+"/disparity")
        os.makedirs(output_folder+"/refocus")
        os.makedirs(output_folder+"/refocus_params")
        os.makedirs(output_folder+"/masks")

    #wrapper
    def do_gen(i):
        print("Generation scene %d"%i)
        synth_image, synth_disp, synth_refocus, masks  = gen_scene_refocus(input_seg_obj_list, input_stereo_left_list, 
                                                                           input_stereo_left_disp_list)
        Image.fromarray(synth_image.astype(np.uint8)).save(output_folder+"/image/%05d.png"%i)
        Image.fromarray(synth_refocus.astype(np.uint8)).save(output_folder+"/refocus/%05d.png"%i)
        f= open(output_folder+"/refocus_params/%05d.csv"%i,"w+")
        f.write("focus parameter:,%f\n"%focus_plane)
        f.write("aperture,%f"%aperture)
        f.close()
        #writePFM(output_folder+ "/disparity/%05d.pfm"%i, synth_left_disp, scale=1)
        np.save(output_folder+ "/disparity/%05d.npy"%i, synth_disp)
        Image.fromarray((255*synth_disp/np.amax(synth_disp)).astype(np.uint8)).save(output_folder+"/disparity/%05d_preview.png"%i)
        for o in range(masks.shape[2]):
            Image.fromarray((255*masks[:,:,o]).astype(np.uint8)).save(output_folder+"/masks/%05d_%d.png"%(i,o))
                 
    #%%
    #actually generates the scenes
    for i in scenes_idxs:
        do_gen(i)
  
# =============================================================================
# #Multicore version      
#     from joblib import Parallel, delayed
#     import multiprocessing
#     num_cores = multiprocessing.cpu_count()  
#     _ = Parallel(n_jobs=num_cores)(delayed(do_gen)(i) for i in scenes_idxs)
# =============================================================================
