#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
import numpy as np
import tensorflow as tf
import click 
import os
from PIL import Image
import time

from blur_baseline.model import refnet_blur_baseline
from tf_utils import optimistic_restore, time_up_to

    
#''

#python demo.py /home/matthieu/kgx_nfs2/data/external/sceneflow/frames_cleanpass/35mm_focallength/scene_forwards/slow/left/0580.png  /home/matthieu/kgx_nfs2/data/external/sceneflow/frames_cleanpass/35mm_focallength/scene_forwards/slow/right/0580.png /home/matthieu/kgx_nfs2/ben/stereo_net_refactored/checkpts/stereoNet_ScenFlow_driving_1.7888498861e-07_step_120000.ckpt -vv -f 0 -i -s cost_volume

#python demo.py /home/matthieu/kgx_nfs2/data/external/kitti/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/0000000022.png /home/matthieu/kgx_nfs2/data/external/kitti/2011_10_03/2011_10_03_drive_0027_sync/image_03/data/0000000022.png /home/matthieu/kgx_nfs2/ben/stereo_net_refactored/checkpts/kitti/stereoNet_kitti_full_resolution_refinement_1.42759967509e-08_step_134000.ckpt -vv -f 40 -i -t -d 0 80 -a 0.25 


@click.command()

#inputs
@click.argument('left_image')
@click.argument('rigth_image')
@click.argument('stereonet_checkpoint')
#outputs parms
@click.option('--output_folder', '-o', default="./", help='Path to the output folder.')
@click.option('--intermediate', '-i', is_flag=True, help='Saves the disparity or cost volume from stereonet.')
@click.option('--verbose', '-v', count=True, help='Sets the logging display level -v means warning -vv is info.')
@click.option('--timming', '-t', is_flag=True, help='Runs 10 times the refocusing to get accurate timming.')
@click.option('--deploy', '-y', is_flag=True, help='Saves the graph is a pb file, the graph is only for one foucs plane and one aperture tho.')
#refocusing parameters
@click.option('--focus_plane', '-f', multiple=True, default=[0.0], type=float, help='Focus parameter. Can be used multiple time to generate multiple images.')
@click.option('--aperture', '-a', default=0.1, help='Virtual aperture, ie blur intensity parameter.')
@click.option('--pyramidal_conv', '-p', default=11, type=(int), help='Defines a maximum kernel size such that the images will be downsampled before comvolution if the blur kernel is too big.')
@click.option('--disparity_range', '-d', nargs=2, type=float, default=[0,300],  help='Minimun and maximum disparity range (only needed for refocusing using disparity).')
@click.option('--from_stage', '-s', default='disparity_map', type=click.Choice(['disparity_map', 'cost_volume']), help='Uses the full resolution disparity or the cost volume to refocus.')

#TODO add option to outptut disparity, gt disp, gt blur, file check type=click.Path(exists=True)

def do_blur(left_image, rigth_image, stereonet_checkpoint,
            output_folder, intermediate, verbose, timming, deploy,
            focus_plane, aperture, pyramidal_conv, disparity_range, from_stage
            ):
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
    tf.logging.info("Opening Images")
    left_img = np.expand_dims(Image.open(left_image), 0).astype(np.float32)/255.0
    rigth_image = np.expand_dims(Image.open(rigth_image), 0).astype(np.float32)/255.0
    h,w = left_img.shape[1:3]
    #making output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)      
    
    #Building the graph
    tf.logging.info("Making graph for %d focal planes"%len(focus_plane))
    left_img_ph = tf.placeholder(tf.float32,shape=[1, h,w,3])
    rigth_img_ph = tf.placeholder(tf.float32,shape=[1, h,w,3])
    net_output_ph = refnet_blur_baseline(left_img_ph, 
                                     rigth_img_ph, 
                                     target_disparities=focus_plane, blur_magnitude=aperture,
                                     is_training=False, stop_grads = True,
                                     min_disp = disparity_range[0], max_disp = disparity_range[1], 
                                     downsampling_trick_max_kernel_size=pyramidal_conv,
                                     from_stage = from_stage
                                     )
    #Runs the thing
    with tf.Session() as sess:
      tf.logging.info("Loading stereonet weigts")
      rv = optimistic_restore(sess, stereonet_checkpoint)
      tf.logging.info("Restored %d vars"%rv)
    
      if deploy:
        tf.logging.info("Saving model and weigts for later deployement")
        refocus_image_ph = net_output_ph[0]
        print(refocus_image_ph)
        tf.saved_model.simple_save(sess, output_folder+"/deployable",
        inputs={"left_img": left_img_ph, "rigth_img": rigth_img_ph},
        outputs={"refocus_image": refocus_image_ph[0]})
        tf.logging.info("Done")

      tf.logging.info("Runing depth estimation and refocusing")
      refocus_image, intermediate_result, _ =  sess.run(net_output_ph, feed_dict={left_img_ph:left_img, rigth_img_ph:rigth_image})
      tf.logging.info("Done")
  
      if timming:
          tf.logging.info("Getting timing") 
          feed_dict={left_img_ph:left_img, rigth_img_ph:rigth_image}
          avg_timing  = time_up_to(sess, net_output_ph[0], feed_dict)  
          avg_timing_intermediate  = time_up_to(sess, net_output_ph[1], feed_dict)  
          tf.logging.info("Average timming for pipeline is %fs, including %fs to compute the disparity/cost volume and %fs for refocusing"%(avg_timing, avg_timing_intermediate, avg_timing-avg_timing_intermediate )) 
        
      tf.logging.info("Saving to "+output_folder+"/refocused_image_[focus_plane].png")
      for i in range(len(focus_plane)):
          f=focus_plane[i]
          Image.fromarray((refocus_image[i][0,:,:,:]*255).astype(np.uint8)).save(output_folder+"/refocused_image_%f.png"%f)
      if intermediate:
            tf.logging.info("Saving intermediate output")
            if from_stage == "disparity_map":
                disp=intermediate_result[0,:,:,:]
                disp=(disp-np.amin(disp))/(np.amax(disp)-np.amin(disp))
                disp=np.tile(disp, [1,1,3])
                Image.fromarray((disp*255).astype(np.uint8)).save(output_folder+"/disparity.png")
            else:
                disparity_range = np.arange(1,18+1)*8#FIXME: see ben for disp=0
                print(np.amin(intermediate_result))
                print(np.amax(intermediate_result))
                
                for d in range(intermediate_result.shape[-1]):
                    Image.fromarray((intermediate_result[0,:,:, d]*255).astype(np.uint8)).save(output_folder + "/conf_volume_%d.png"%disparity_range[d])
                
if __name__ == '__main__':
    do_blur()
