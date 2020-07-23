import tensorflow as tf
import numpy as np 
import re

def conv2d(inputs, filters, kernel_size, name, strides=1, dilation_rate=1):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.zeros_initializer(),dilation_rate=dilation_rate,
                           name=name)

def conv3d(input,num_outputs,kernel_size,name):
    return tf.layers.conv3d(inputs=input,filters=num_outputs,kernel_size=kernel_size,kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=None,padding='same',name=name)

def resnet_block(inputs, filters, kernel_size, name, dilation_rate=1):
    out = conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, name=name + '_conv1', dilation_rate=dilation_rate)
    out = tf.nn.relu(out,name=name + '_relu1')
    out = conv2d(inputs=out, filters=filters, kernel_size=kernel_size, name=name + '_conv2', dilation_rate=dilation_rate)
    out = tf.add(out, inputs, name=name + '_add')
    out = tf.nn.relu(out,name=name + '_relu2')
    return out

def lcn_preprocess(input_tensor):
    """
    Returns the normalised and centered values of a tensor, along with its standard dev.
    """
    full_h = int(input_tensor.shape[1])
    full_w = int(input_tensor.shape[2])
    
    ##compute local averages
    ones = tf.ones_like(input_tensor)
    avg_filter = tf.ones([9,9,3,1],dtype=tf.float32,name='avg_filter')
    divide_weight = tf.nn.convolution(ones,filter=avg_filter,padding='SAME')
    input_tensor_avg = tf.nn.convolution(input_tensor,filter=avg_filter,padding='SAME') / divide_weight
    
    #compute local std dev
    padded_left = tf.pad(input_tensor,[[0,0],[4,4],[4,4],[0,0]])
    padded_ones = tf.pad(ones,[[0,0],[4,4],[4,4],[0,0]])
    input_tensor_std = tf.zeros_like(input_tensor)
    for x in range(9):
        for y in range(9):
            input_tensor_std += tf.square(padded_left[:,y:y+full_h,x:x+full_w,:] - input_tensor_avg) * padded_ones[:,y:y+full_h,x:x+full_w,:]
    const = 1e-10
    input_tensor_std = tf.sqrt((input_tensor_std + const) / divide_weight)
    
    #Center input around mean
    input_tensor = (input_tensor - input_tensor_avg) / (input_tensor_std + const)

    return input_tensor

def readPFM(file):
    '''
    This code is from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    '''
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (1,height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    
    return data

def writePFM(file, image, scale=1):
    '''
    This code is from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    '''
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)