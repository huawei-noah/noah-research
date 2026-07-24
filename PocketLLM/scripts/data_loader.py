from PIL import Image
import torch.utils.data as data
import numpy as np
import math
import os
from transformers import LlamaForCausalLM
from torch import nn
from .reshape import *

def normlizer(arr):
    maxer=arr.max()
    miner=arr.min()
    arr=(arr-miner)/(maxer-miner)
    return arr
    
class ImageDataset(data.Dataset):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    def __init__(self, data_dir="/home/ma-user/work/tianye/PocketLLM/llama2_7B_npy_v4/",\
        weight_type='q',weight_layer=3, codebook_len_index=12,cmd={}, **kwargs):
        
        self.data_dir = data_dir 
        self.input_embeddings=[]
        self.name_list=[]
        self.weight_type=weight_type
        self.weight_layer=weight_layer
        
        self.get_all_samples()
        
    def get_all_samples(self):
#         file_list=os.listdir(self.data_dir)
#         file_list.sort()
        if self.weight_type in ['gate','up','down']:
            file_name=f'model.layers.{str(self.weight_layer)}.mlp.{self.weight_type}_proj.npy'#file_list[0]
        else:
            file_name=f'model.layers.{str(self.weight_layer)}.self_attn.{self.weight_type}_proj.npy'#file_list[0]
        self.input_embeddings=np.load(self.data_dir+file_name)
        print('max:{:.11f},min:{:.11f}'.format(self.input_embeddings.max(),self.input_embeddings.min()))
        # self.input_embeddings=normlizer(self.input_embeddings)
        print("Load ",file_name," Done!")
        
        
    def __getitem__(self, index):
        # single=np.array([self.input_embeddings[index]]).astype('float32')
        single=np.array(self.input_embeddings[index]).astype('float32')
        
        return single

    def __len__(self):
        return len(self.input_embeddings)
