# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
import tensorflow as tf

from stereonet.utils import lcn_preprocess, conv2d, conv3d, resnet_block

def _tower_feature_net(img, is_training): 
    """
    Definition for the feature network.
    
    img: the input tensor for a batch of images
    
    Return a tensor of features.
    """
    
    with tf.variable_scope("feature_tower", reuse=tf.AUTO_REUSE):
        out = conv2d(inputs=img, filters=32, kernel_size=3, name='tower_conv1')
    
        out = resnet_block(inputs=out, filters=32, kernel_size=3, name='tower_res1')
        out = resnet_block(inputs=out, filters=32, kernel_size=3, name='tower_res2')
        out = resnet_block(inputs=out, filters=32, kernel_size=3, name='tower_res3')
    
        out = conv2d(inputs=out, filters=32, kernel_size=3, strides=2, name='tower_conv2')
        out = tf.layers.batch_normalization(inputs=out, training=is_training,name='tower_bn1')
    
        out = tf.nn.leaky_relu(features=out, name='tower_leaky1')
    
        out = conv2d(inputs=out, filters=32, kernel_size=3, strides=2, name='tower_conv3')
        out = tf.layers.batch_normalization(inputs=out, training=is_training, name='tower_bn2')
        out = tf.nn.leaky_relu(features=out, name='tower_leaky2')
    
        out = conv2d(inputs=out, filters=32, kernel_size=3, strides=2, name='tower_conv4')
        out = tf.layers.batch_normalization(inputs=out, training=is_training, name='tower_bn3')
        tower_feature = tf.nn.leaky_relu(features=out, name='tower_leaky3')
    
    return tower_feature

def _disparity_map_from_cost_volume(cost_volume, max_disp_lowres):
    """
    Do the softmax on the cost volume to get the depth map.
    
    cost_volume: the lowres cost volume with slices for disparity 0 to max_disp_lowres (in the lowres space)
    
    Return: the disparity map at lowres
    """

    with tf.variable_scope("disparity_map_from_cost_volume"):
        #FIXME: use tf softmax
        #FIXME: no beta parameter
        exp_disparity = tf.exp(-cost_volume)
        #FIXME:disparity starts at 1 here, should be 0 or arbitrary actually
        disparity_idx = tf.range(1,max_disp_lowres+1,dtype = tf.float32)
        numerator = tf.multiply(exp_disparity,disparity_idx)
        denominator = tf.reduce_sum(exp_disparity,axis=3,keep_dims=True)
        
        disparity_map = tf.reduce_sum(numerator/denominator,axis=3,keep_dims=True)
        
        return disparity_map

def _generate_cost_volume(tower_reference_view, tower_view_to_change, 
                         max_disp_lowres,
                         reference,
                         is_training):
    """
    Build the cost volume on a given view coordinate system.
    """
    cost_volume_w = int(tower_reference_view.shape[2])
    
    #FIXME: this step is abnormaly long
    with tf.variable_scope("cost_volume_building"):
    
        if reference == 'right':
            tower_view_to_change_padded = tf.pad(tower_view_to_change,[[0,0],[0,0],[0,max_disp_lowres],[0,0]],"CONSTANT")
        elif reference == 'left':
            tower_view_to_change_padded = tf.pad(tower_view_to_change,[[0,0],[0,0],[max_disp_lowres,0],[0,0]],"CONSTANT")
    
        cost_volume = None
    
        for each_disparity in range(max_disp_lowres):
            
            if reference == 'right':
                tower_view_to_change_offset = tower_view_to_change_padded[:,:,(each_disparity+1):cost_volume_w+(each_disparity+1),:]
            elif reference == 'left':
                tower_view_to_change_offset = tower_view_to_change_padded[:,:,(max_disp_lowres)-(each_disparity+1):cost_volume_w + (max_disp_lowres) - (each_disparity+1),:]
    
            each_feature_difference = tf.abs(tower_view_to_change_offset - tower_reference_view)
            each_disparity_cost = tf.expand_dims(each_feature_difference,3)
    
            if each_disparity == 0:
                cost_volume = each_disparity_cost
            else:
                cost_volume = tf.concat([cost_volume,each_disparity_cost],axis=3)
     
    #convenient placeholder
    cost_volume_coarse = cost_volume
                
    with tf.variable_scope("cost_volume_regularisation"):
        cost_volume = conv3d(input=cost_volume,num_outputs=32,kernel_size=3,name="cost_conv3d1")
        cost_volume = tf.layers.batch_normalization(inputs=cost_volume, training=is_training, name='cost_bn1')
        cost_volume = tf.nn.leaky_relu(features=cost_volume, name='cost_leaky1')
        cost_volume = conv3d(input=cost_volume,num_outputs=32,kernel_size=3,name="cost_conv3d2")
        cost_volume = tf.layers.batch_normalization(inputs=cost_volume, training=is_training, name='cost_bn2')
        cost_volume = tf.nn.leaky_relu(features=cost_volume, name='cost_leaky2')
        cost_volume = conv3d(input=cost_volume,num_outputs=32,kernel_size=3,name="cost_conv3d3")
        cost_volume = tf.layers.batch_normalization(inputs=cost_volume, training=is_training, name='cost_bn3')
        cost_volume = tf.nn.leaky_relu(features=cost_volume, name='cost_leaky3')
        cost_volume = conv3d(input=cost_volume,num_outputs=32,kernel_size=3,name="cost_conv3d4")
        cost_volume = tf.layers.batch_normalization(inputs=cost_volume, training=is_training, name='cost_bn4')
        cost_volume = tf.nn.leaky_relu(features=cost_volume, name='cost_leaky4')
        cost_volume = conv3d(input=cost_volume,num_outputs=1,kernel_size=3,name="cost_conv3d5")
        cost_volume = tf.squeeze(cost_volume,name='cost_squeeze',axis=4)
    
    return cost_volume, cost_volume_coarse


def _disparity_map_refinement_net(img, upsampled_disparity_map, scale_name, is_training):
    """
    Step to upsample and refine a disparity map, aided by the input image.
    """
    with tf.variable_scope("guided_upsampling_scale_"+scale_name):
        #
        disparity_out = conv2d(inputs=upsampled_disparity_map, filters=16, kernel_size=3, name = 'refine_disparity_conv1_' + scale_name )
        disparity_out = tf.layers.batch_normalization(inputs=disparity_out, training=is_training, name='refine_disparity_bn1_' + scale_name )
        disparity_out = tf.nn.leaky_relu(features=disparity_out, name='refine_disparity_leaky1_' + scale_name)
        disparity_out = resnet_block(inputs=disparity_out, filters=16, kernel_size=3, dilation_rate=1,name='refine_disparity_res1_' + scale_name)
        disparity_out = resnet_block(inputs=disparity_out, filters=16, kernel_size=3, dilation_rate=2, name='refine_disparity_res2_' + scale_name)
        #
        image_out = conv2d(inputs=img, filters=16, kernel_size=3, name = 'refine_image_conv1_' + scale_name)
        image_out = tf.layers.batch_normalization(inputs=image_out, training=is_training, name='refine_image_bn1_' + scale_name)
        image_out = tf.nn.leaky_relu(features=image_out, name='refine_image_leaky1_' + scale_name)
        image_out = resnet_block(inputs=image_out, filters=16, kernel_size=3, dilation_rate=1,name='refine_image_res1_' + scale_name)
        image_out = resnet_block(inputs=image_out, filters=16, kernel_size=3, dilation_rate=2, name='refine_image_res2_' + scale_name)
        #
        concat_out = tf.concat([image_out,disparity_out],axis=3,name='refine_concat_' + scale_name)
        #
        concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=4, name='refine_concat_res0_' + scale_name)
        concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=8, name='refine_concat_res1_' + scale_name)
        concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=1, name='refine_concat_res2_' + scale_name)
        concat_out = resnet_block(inputs=concat_out, filters=32, kernel_size=3, dilation_rate=1, name='refine_concat_res3_' + scale_name)
        #
        disparity_residual = conv2d(inputs=concat_out, filters=1, kernel_size=3, name = 'refine_concat_conv1_' + scale_name)
        #
        high_res_disparity_map = upsampled_disparity_map + disparity_residual
            
        return high_res_disparity_map
    
def _guided_upsampling(batch_left, disparity_map_1_8, is_training):
    """
    Performs bilinear upsampling and pass the result trougth a refinement net.
    """
    heigt=int(batch_left.shape[1])
    width=int(batch_left.shape[2])
    with tf.variable_scope("guided_upsampling"):
        #resize input image at different scales
        batch_left_1_4_raw = tf.image.resize_images(batch_left,size=(int(heigt/4),int(width/4)),align_corners=True)
        batch_left_1_2_raw = tf.image.resize_images(batch_left,size=(int(heigt/2),int(width/2)),align_corners=True)
        # bilinear upsample & refinement from scale /8 to /4
        disparity_map_1_4 = tf.image.resize_images(disparity_map_1_8,size=(int(heigt/4),int(width/4)),align_corners=True) *  2
        refined_disparity_map_1_4 = _disparity_map_refinement_net(batch_left_1_4_raw, disparity_map_1_4, scale_name='1_4', is_training=is_training)
        # bilinear upsample & refinement from scale /4 to /2
        disparity_map_1_2 = tf.image.resize_images(refined_disparity_map_1_4,size=(int(heigt/2),int(width/2)),align_corners=True) * 2
        refined_disparity_map_1_2 = _disparity_map_refinement_net(batch_left_1_2_raw, disparity_map_1_2, scale_name='1_2', is_training=is_training)
        # bilinear upsample & refinement from scale /2 to /1
        disparity_map_1_1 = tf.image.resize_images(refined_disparity_map_1_2,size=(int(heigt),int(width)),align_corners=True) * 2
        full_res_disparity_map = _disparity_map_refinement_net(batch_left,disparity_map_1_1,scale_name='1_1', is_training=is_training)
        return full_res_disparity_map, disparity_map_1_2, disparity_map_1_4

def stereonet(batch_left_raw, batch_right_raw, max_disp_lowres=18, is_training=False, do_preprocess=False):
    """
    Defines the complete pipeline for stereonet.
    
    batch_left, batch_right: the left and right images batch tensor
    max_disp_lowres: maximum value for disparity at low resolution, minimum is considered to be 0. Also the number of slices in the cost volume.
    is_training: flag to toogle batch nomrm
    
    Return the ful res disparity map tensor and the cost volume 
    """
    
    heigt, width = batch_left_raw.shape[1:3]

    #FIXME: this step is abdnomaly long
    if do_preprocess:
        batch_left = lcn_preprocess(batch_left_raw)
        batch_right = lcn_preprocess(batch_right_raw)
    else:
        batch_left = batch_left_raw
        batch_right = batch_right_raw

    with tf.variable_scope("stereonet"):
        #get features from r and l images
        tower_feature_left = _tower_feature_net(batch_left, is_training)
        tower_feature_right = _tower_feature_net(batch_right, is_training)
        #build the cost volume
        cost_volume_left_view, cost_volume_left_view_coarse = _generate_cost_volume(tower_feature_left,tower_feature_right, max_disp_lowres, reference='left', is_training=is_training)
        # get low resolution dispairty map 
        disparity_map_1_8 = _disparity_map_from_cost_volume(cost_volume_left_view, max_disp_lowres)
        #Do the guided upsampling step
        full_res_disparity_map, disparity_map_1_2, disparity_map_1_4 = _guided_upsampling(batch_left, disparity_map_1_8, is_training=is_training)
        
    intermediate_steps =  {
                          "input_left_raw":batch_left_raw,
                          "input_left_preprocess":batch_right,
                          "tower_feature_left":tower_feature_left,
                          "tower_feature_right":tower_feature_right,
                          "cost_volume_left_view_coarse": cost_volume_left_view_coarse,
                          "cost_volume_left_view": cost_volume_left_view,
                          "disparity_map_1_2": disparity_map_1_2,
                          "disparity_map_1_4": disparity_map_1_4,
                          "disparity_map_1_8":disparity_map_1_8
                          }
    
    return full_res_disparity_map, intermediate_steps
            

    
