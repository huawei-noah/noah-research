import numpy as np
import sys
from PIL import  ImageOps


def pad_to_size(im, new_size):
    """
    Central pad with 0 to size.
    """
    delta_w = new_size[0] - im.size[0]
    delta_h = new_size[1] - im.size[1]
    assert(delta_w>=0 and delta_h>=0)
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2),  delta_h-(delta_h//2))
    im = ImageOps.expand(im, padding)
    return im

def crop_to_size(im, new_size, canvas_width, canvas_heigt):
    """
    Central crop to size. taking a canvas into consideration.
    """
    delta_w =  im.size[0] - new_size[0]
    delta_h = im.size[1] - new_size[1] 
    assert(delta_w>=0 and delta_h>=0)
    cropping = (delta_w//2, delta_h//2, canvas_width-(delta_w//2),  canvas_heigt -(delta_h//2))
    im = im.crop(cropping)
    return im

def resize(array, size_arg):
    """
    Image resize using scipy.
    """
    from scipy.misc import imresize
    if len(array.shape) == 2:
        return imresize(array, size_arg, mode="F")
    elif len(array.shape) == 3:
        rarray = []
        for c in range(array.shape[2]):
            rarray.append(resize(array[:,:,c], size_arg))
        return np.stack(rarray, -1)
    

def readPFM(file):
    '''
    This code is from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    '''
    import re
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