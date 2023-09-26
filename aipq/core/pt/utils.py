# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.         
#                                                                                
# This program is free software; you can redistribute it and/or modify it under  
# the terms of the MIT license.                                                  
#                                                                                
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.                      

import os
import torch
import shutil


def save_checkpoint(state, is_best, fdir='.', filename='checkpoint.pth.tar'):
    fpath = os.path.join(fdir, filename)

    torch.save(state, fpath)
    if is_best:
        fpath_best = os.path.join(fdir, 'model_best.pth.tar')
        shutil.copyfile(fpath, fpath_best)

