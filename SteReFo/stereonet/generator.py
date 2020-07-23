import numpy as np
import random
from PIL import Image

from stereonet.utils import readPFM

class TrainingGeneratorStereoNet:
    """ Data generator class, used to open and build the train set for the depth estimation on the fly. """
    
    def __init__(self, list_files_left, list_files_right, list_files_disparity):
        """
        Takes as input two lists of tuples, each row containing the path to an input image and a GT sample.
        """
        self.list_files_left = list_files_left
        self.list_files_right = list_files_right
        self.list_files_disparity = list_files_disparity
        #dummy actuall pass the input train folder
        self.sample_num_left = len(list_files_left)
        self.sample_num_right = len(list_files_right)
        self.sample_num_disparity = len(list_files_disparity)
        self.counter = 0
    
    def __iter__(self):
        while True:
            #Compose the dataset opener that mixes data from dataset
            # open one element of each dataset
            
            number_of_imgs = min(self.sample_num_left, self.sample_num_right, self.sample_num_disparity)            
            
            data_indices = range(number_of_imgs)
            data_shuffled = random.sample(data_indices,number_of_imgs)
            
            for data in data_shuffled:
                #open lr images
                rgb_left = np.asarray(Image.open(self.list_files_left[data]))/255.0
                rgb_right = np.asarray(Image.open(self.list_files_right[data]))/255.0
                #open disparity and reshape dim
                disparity = readPFM(self.list_files_disparity[data])
                disparity = np.expand_dims(disparity,-1)
                #
                self.counter += 1
             
                # return rgb images and disparity
                yield (rgb_left, rgb_right, disparity)