# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
import numpy as np
import tensorflow as tf
import click 
import os
from PIL import Image
import time

from blur_baseline.model import refnet_blur_baseline
from tf_utils import optimistic_restore, time_up_to

def parse_files_lines(fname):
    with open(fname) as file:
        return [line.rstrip('\n') for line in file]

@click.command()

#inputs
@click.argument('left_image_list')
@click.argument('rigth_image_list')
@click.argument('focus_list')
@click.argument('stereonet_checkpoint')
#outputs parms
@click.option('--output_folder', '-o', default="./", help='Path to the output folder.')
@click.option('--intermediate', '-i', is_flag=True, help='Saves the disparity or cost volume from stereonet.')
@click.option('--verbose', '-v', count=True, help='Sets the logging display level -v means warning -vv is info.')
#refocusing parameters
@click.option('--aperture', '-a', default=0.1, help='Virtual aperture, ie blur intensity parameter.')
@click.option('--pyramidal_conv', '-p', default=11, type=(int), help='Defines a maximum kernel size such that the images will be downsampled before comvolution if the blur kernel is too big.')
@click.option('--disparity_range', '-d', default=[0,300],  help='Minimun and maximum disparity range (only needed for refocusing using disparity).')
@click.option('--from_stage', '-s', default='disparity_map', type=click.Choice(['disparity_map', 'cost_volume']), help='Uses the full resolution disparity or the cost volume to refocus.')
#video params
@click.option('--resume', '-r', default=0, help='Resumes from a given frame index.')

def do_blur(left_image_list, rigth_image_list, focus_list, stereonet_checkpoint,
            output_folder, intermediate, verbose,
            aperture, pyramidal_conv, disparity_range, from_stage, 
            resume
            ):
    #defines verbose levels
    if verbose>=2:
        tf.logging.set_verbosity(tf.logging.INFO)
    elif verbose>=1:
        tf.logging.set_verbosity(tf.logging.WARN)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)
    #remove verbose bits from tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   
    
    #parse input files
    left_img_files = parse_files_lines(left_image_list)
    right_img_files = parse_files_lines(rigth_image_list)
    focus_planes = parse_files_lines(focus_list)
    assert(len(left_img_files)==len(right_img_files))
    assert(len(left_img_files)==len(focus_planes))
    inputs = zip(left_img_files, right_img_files, focus_planes)
    #remove the part already computed
    inputs = inputs[resume:]
    #useful
    sequence_length = len(inputs)
    tf.logging.info("Going to compute for %d frames (first %d ignored)"%(sequence_length, resume))
     
    #making output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 
    
    #loop on the frames
    prev_focus_plane =  float('nan')
    sess = tf.Session()
    for i in range(sequence_length):
        tf.logging.info("Processing frame %d"%(i+resume))
        focus_plane= float(inputs[i][2])
        left_img_file= inputs[i][0]
        right_img_file= inputs[i][1]
        #open files
        tf.logging.info("Opening Images")
        left_img = np.expand_dims(Image.open(left_img_file), 0).astype(np.float32)/255.0
        rigth_image = np.expand_dims(Image.open(right_img_file), 0).astype(np.float32)/255.0
        #file size
        h,w = left_img.shape[1:3]
        if prev_focus_plane != focus_plane:
            tf.logging.info("Change in focus plane detected, rebuilding graph")
            tf.reset_default_graph()
            #frame placeholders
            left_img_ph = tf.placeholder(tf.float32,shape=[1, h,w,3])
            rigth_img_ph = tf.placeholder(tf.float32,shape=[1, h,w,3])
            #build the graph for the first focus frame
            net_output_ph = refnet_blur_baseline(left_img_ph, 
                                                 rigth_img_ph, 
                                                 target_disparities=[focus_plane], blur_magnitude=aperture,
                                                 is_training=False, stop_grads = True,
                                                 min_disp = disparity_range[0], max_disp = disparity_range[1], 
                                                 downsampling_trick_max_kernel_size=pyramidal_conv,
                                                 from_stage = from_stage
                                                 )
            #need to reopen session because tf is shitty
            sess.close()
            sess = tf.Session()
            #Note: could avoid having to reload weigts here 
            tf.logging.info("Loading stereonet weigts")
            rv = optimistic_restore(sess, stereonet_checkpoint)
            tf.logging.info("Restored %d vars"%rv)
        #run the shit      
        tf.logging.info("Runing depth estimation and refocusing")
        refocus_image, intermediate_result, _ =  sess.run(net_output_ph, feed_dict={left_img_ph:left_img, rigth_img_ph:rigth_image})
        tf.logging.info("Done")

        tf.logging.info("Saving to "+output_folder+"/refocused_[left_image_file_name]s")

        Image.fromarray((refocus_image[0][0,:,:,:]*255).astype(np.uint8)).save(output_folder+"/refocused_%d.png"%(i+resume))
        if intermediate:
            tf.logging.info("Saving intermediate output")
            if from_stage == "disparity_map":
                disp=intermediate_result[0,:,:,:]
                disp=(disp-np.amin(disp))/(np.amax(disp)-np.amin(disp))
                disp=np.tile(disp, [1,1,3])
                Image.fromarray((disp*255).astype(np.uint8)).save(output_folder+"/disparity_%d.png"%(i+resume))
            else:
                disparity_range = np.arange(1,18+1)*8#FIXME: see ben for disp=0
                for d in range(intermediate_result.shape[-1]):
                        Image.fromarray((intermediate_result[0,:,:, d]*255).astype(np.uint8)).save(output_folder + "/conf_volume_%d_%d"%(disparity_range[d], i+resume))
                
if __name__ == '__main__':
    do_blur()
