#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
import tensorflow as tf
import time

def optimistic_restore(session, save_file):
    """
    Restores as many weigts as possible from a checkpoint filed, in contrast to tf
    function why returns an exeption if some weigts in the graph are not found in the checkpoint.
    Very convenient when you want to load pretrained part of your graph.
    
    session: session to restore the weigts to
    save_file: the checkpoint file
    
    returns the number of variables restored
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)
    return len(restore_vars)

def time_up_to(sess, placeholder,  feed_dict=None, run_n=11, ignore_first=True):
    """
    Evaluate run_n times the tensor placeholder on the session sess.
    Returns the average run time in seconds
    """
    avg_timing  = 0
    for t in range(run_n):
        start = time.time()
        _ =  sess.run(placeholder, feed_dict=feed_dict)
        if t!=0 or not ignore_first:
            avg_timing+= time.time()-start
    return avg_timing/run_n  

    
