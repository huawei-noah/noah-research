#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
import numpy as np
import tensorflow as tf
import click 
import os
from PIL import Image
import time

from refocus_algorithms.layered_dof_tf import layered_bluring
  
#python demo.py /home/matthieu/kgx_nfs2/data/external/sceneflow/frames_cleanpass/35mm_focallength/scene_forwards/slow/left/0580.png  /home/matthieu/kgx_nfs2/data/external/sceneflow/frames_cleanpass/35mm_focallength/scene_forwards/slow/right/0580.png /home/matthieu/kgx_nfs2/ben/stereo_net/checkpoints/stereoNet_testrun_5.15377520732e-07_step_100000val_0.ckpt -vv -f 0 -i -s cost_volume

@click.command()

@click.argument('image')
@click.argument('disparity_map')

@click.option('--output_folder', '-o', default="./", help='Path to the output folder.')
@click.option('--focus_plane', '-f', multiple=True, default=[0.0], type=float, help='Focus parameter. Can be used multiple time to generate multiple images.')
@click.option('--aperture', '-a', default=0.1, help='Virtual aperture, ie blur intensity parameter.')
@click.option('--pyramidal_conv', '-p', default=11, type=(int), help='Defines a maximum kernel size such that the images will be downsampled before comvolution if the blur kernel is too big.')
@click.option('--disparity_range', '-d', nargs=2, type=float, default=[0,300],  help='Minimun and maximum disparity range.')

@click.option('--verbose', '-v', count=True, help='Sets the logging display level -v means warning -vv is info.')

def do_blur(image, disparity_map, output_folder, 
            focus_plane, aperture, 
            pyramidal_conv, disparity_range, 
            verbose):
    #defines verbose level
    if verbose>=2:
        tf.logging.set_verbosity(tf.logging.INFO)
    elif verbose>=1:
        tf.logging.set_verbosity(tf.logging.WARN)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)
    #remove verbose bits from tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   
   
    #Opening images
    tf.logging.info("Opening files")
    image = np.expand_dims(Image.open(image), 0).astype(np.float32)/255.0
    image = image[:,:,:,0:3]
    
    if disparity_map[-3:] == "npy":
        disp_map = np.expand_dims(np.expand_dims(np.load(disparity_map), 0),-1)
        
    elif disparity_map[-3:] == "png":
        tf.logging.warn("Loading disparty map from nor;alised png. Disparity values between 0 and 1")
        disp_map = np.expand_dims(Image.open(disparity_map), 0).astype(np.float32)/255.0
        disp_map = np.expand_dims(disp_map[:,:,:,0],-1)
    else:
        raise BaseException("Disparity map format unsupported yet")                 
           
    h,w = image.shape[1:3]
    assert(image.shape[1:3] == disp_map.shape[1:3], "Disparity map and  image of different size.")
    
    #making output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)      
    
    #Building the graph
    tf.logging.info("Making graph for %d focal planes"%len(focus_plane))
    img_ph = tf.placeholder(tf.float32,shape=[1, h,w,3])
    disp_ph = tf.placeholder(tf.float32,shape=[1, h,w,1])

    output_ph = []
    for f in focus_plane:
        output_ph.append(layered_bluring(img_ph, disp_ph, 
                        target_disparity=f, blur_magnitude=aperture,
                        min_disp = disparity_range[0], max_disp = disparity_range[1], 
                        downsampling_trick_max_kernel_size=pyramidal_conv,
                        differenciable=False))
           
    #Runs the thing
    with tf.Session() as sess:
  
      tf.logging.info("Runing refocusing")
      refocus_image =  sess.run(output_ph, feed_dict={img_ph:image, disp_ph:disp_map})
      tf.logging.info("Done")
        
      tf.logging.info("Saving to "+output_folder+"/refocused_image_[focus_plane].png")
      for i in range(len(focus_plane)):
          f=focus_plane[i]
          Image.fromarray((refocus_image[i][0,:,:,:]*255).astype(np.uint8)).save(output_folder+"/refocused_image_%f.png"%f)
   
if __name__ == '__main__':
    do_blur()
