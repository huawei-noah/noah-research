# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
import numpy as np
import tensorflow as tf
import click 
import os
from PIL import Image

from stereonet.model import stereonet
from tf_utils import optimistic_restore, time_up_to

#python demo.py /home/matthieu/kgx_nfs2/data/external/sceneflow/frames_cleanpass/35mm_focallength/scene_forwards/slow/left/0580.png  /home/matthieu/kgx_nfs2/data/external/sceneflow/frames_cleanpass/35mm_focallength/scene_forwards/slow/right/0580.png /home/matthieu/kgx_nfs2/ben/stereo_net/checkpoints/stereoNet_testrun_5.15377520732e-07_step_100000val_0.ckpt -vv -i -t

@click.command()

@click.argument('left_image')
@click.argument('rigth_image')
@click.argument('stereonet_checkpoint')

@click.option('--output_folder', '-o', default="./", help='Path to the output folder.')

@click.option('--intermediate', '-i', is_flag=True, help='Saves the disparity or cost volume from stereonet.')
@click.option('--verbose', '-v', count=True, help='Sets the logging display level -v means warning -vv is info.')
@click.option('--timming', '-t', is_flag=True, help='Runs 10 times the refocusing to get accurate timming.')

#TODO add option to outptut disparity, gt disp, gt blur, file check type=click.Path(exists=True)

def do_depth(left_image, rigth_image, stereonet_checkpoint,
            output_folder, 
            intermediate, verbose, timming):
    #defines verbose level
    if verbose>=2:
        tf.logging.set_verbosity(tf.logging.INFO)
    elif verbose>=1:
        tf.logging.set_verbosity(tf.logging.WARN)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)
    #remove verbose bits from tf
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   
   
    #Opening images
    tf.logging.info("Opening Images")
    left_img = np.expand_dims(Image.open(left_image), 0).astype(np.float32)/255.0
    rigth_image = np.expand_dims(Image.open(rigth_image), 0).astype(np.float32)/255.0
    #720p test
    #left_img = np.zeros([1,720/4,1280/4, 3])
    #rigth_image = np.zeros([1,720/4,1280/4,3])
  
    h,w = left_img.shape[1:3]
    tf.logging.info("Image size %d %d"%(h,w))
    
    
    #making output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)      
    
    #Building the graph 
    tf.logging.info("Making graph")
    left_img_ph = tf.placeholder(tf.float32,shape=[1, h,w,3])
    rigth_img_ph = tf.placeholder(tf.float32,shape=[1, h,w,3])
    net_output_ph = stereonet(left_img_ph, rigth_img_ph, is_training=True)
    
    #Runs the thing
    with tf.Session() as sess:
      tf.logging.info("Loading stereonet weigts")
      rv = optimistic_restore(sess, stereonet_checkpoint)
      tf.logging.info("Restored %d vars"%rv)
      tf.logging.info("Runing depth estimation")
      full_res_disparity_map, intermediate_results =  sess.run(net_output_ph, feed_dict={left_img_ph:left_img, rigth_img_ph:rigth_image})
      tf.logging.info("Done")
             
      if timming:
          tf.logging.info("Getting timing") 
            
          feed_dict={left_img_ph:left_img, rigth_img_ph:rigth_image}
            
          avg_timing  = time_up_to(sess, net_output_ph[0], feed_dict)
          avg_timing_input_left_raw = time_up_to(sess, net_output_ph[1]["input_left_raw"], feed_dict)
          avg_timing_input_left_preprocess = time_up_to(sess, net_output_ph[1]["input_left_preprocess"], feed_dict)
          avg_timing_tower_feature_right = time_up_to(sess, net_output_ph[1]["tower_feature_right"], feed_dict)
          avg_timing_tower_cost_volume_left_view_coarse=time_up_to(sess, net_output_ph[1]["cost_volume_left_view_coarse"], feed_dict)
          avg_timing_input_cost_volume_left_view = time_up_to(sess, net_output_ph[1]["cost_volume_left_view"], feed_dict)
          avg_timing_input_disparity_map_1_8 = time_up_to(sess, net_output_ph[1]["disparity_map_1_8"], feed_dict)
          avg_timing_input_disparity_map_1_4 = time_up_to(sess, net_output_ph[1]["disparity_map_1_4"], feed_dict)
          avg_timing_input_disparity_map_1_2 = time_up_to(sess, net_output_ph[1]["disparity_map_1_2"], feed_dict)
            
          
          tf.logging.info("Timeline:") 
          tf.logging.info("%fs to get to the raw input"%(avg_timing_input_left_raw)) 
          tf.logging.info("%fs to get to the processed input"%(avg_timing_input_left_preprocess))
          tf.logging.info("%fs to get to the image features"%(avg_timing_tower_feature_right))
          tf.logging.info("%fs to get to the coarse cost volume"%(avg_timing_tower_cost_volume_left_view_coarse)) 
          tf.logging.info("%fs to get to the cost volume"%(avg_timing_input_cost_volume_left_view)) 
          tf.logging.info("%fs to get to the disparity at 1/8"%(avg_timing_input_disparity_map_1_8))
          tf.logging.info("%fs to get to the disparity at 1/4"%(avg_timing_input_disparity_map_1_4))
          tf.logging.info("%fs to get to the disparity at 1/2"%(avg_timing_input_disparity_map_1_2)) 
          tf.logging.info("%fs to get to the disparity at 1/1"%(avg_timing)) 
        
          tf.logging.info("Differencial:") 
          tf.logging.info("%fs to get to the raw input"%(avg_timing_input_left_raw)) 
          tf.logging.info("%fs to get to the processed input"%(avg_timing_input_left_preprocess-avg_timing_input_left_raw))
          tf.logging.info("%fs to get to the image features"%(avg_timing_tower_feature_right-avg_timing_input_left_preprocess))
          tf.logging.info("%fs to get to the coarse cost volume"%(avg_timing_tower_cost_volume_left_view_coarse-avg_timing_tower_feature_right)) 
          tf.logging.info("%fs to get to the cost volume"%(avg_timing_input_cost_volume_left_view-avg_timing_tower_cost_volume_left_view_coarse)) 
          tf.logging.info("%fs to get to the disparity at 1/8"%(avg_timing_input_disparity_map_1_8-avg_timing_input_cost_volume_left_view))
          tf.logging.info("%fs to get to the disparity at 1/4"%(avg_timing_input_disparity_map_1_4-avg_timing_input_disparity_map_1_8))
          tf.logging.info("%fs to get to the disparity at 1/2"%(avg_timing_input_disparity_map_1_2-avg_timing_input_disparity_map_1_4)) 
          tf.logging.info("%fs to get to the disparity at 1/1"%(avg_timing-avg_timing_input_disparity_map_1_2)) 
        

      tf.logging.info("Saving to "+output_folder+"/disparity_image.png")
      def disp2im(disp):
        disp=disp[0,:,:,:]
        disp=(disp-np.amin(disp))/(np.amax(disp)-np.amin(disp))
        return (np.tile(disp, [1,1,3])*255).astype(np.uint8)
      Image.fromarray(disp2im(full_res_disparity_map)).save(output_folder+"/disparity.png")   
        
      if intermediate:
        #cost_volume_left_view=intermediate_results["cost_volume_left_view"]
        disparity_map_1_2=intermediate_results["disparity_map_1_2"]
        disparity_map_1_4=intermediate_results["disparity_map_1_4"]
        disparity_map_1_8=intermediate_results["disparity_map_1_8"]
        tf.logging.info("Saving intermediate output")
        Image.fromarray(disp2im(disparity_map_1_2)).save(output_folder+"/disparity_1_2.png")  
        Image.fromarray(disp2im(disparity_map_1_4)).save(output_folder+"/disparity_map_1_4.png")
        Image.fromarray(disp2im(disparity_map_1_8)).save(output_folder+"/disparity_map_1_8.png")
                
if __name__ == '__main__':
    do_depth()
