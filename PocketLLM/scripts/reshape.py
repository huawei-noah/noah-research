# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F
import math


def reshape_weight(weight):
    """
    C_out x C_in x k x k -> (C_in x k x k) x C_out.
    """
    
    if len(weight.size()) == 4:
        C_out, C_in, k, k = weight.size()
        return weight.view(C_out, C_in * k * k).t()
    else:
        return weight.t()


def reshape_back_weight(weight, k=3, conv=True):
    """
    (C_in x k x k) x C_out -> C_out x C_in x k x k.
    """
    
    if conv:
        C_in_, C_out = weight.size()
        C_in = C_in_ // (k * k)
        return weight.t().view(C_out, C_in, k, k)
    else:
        return weight.t()


def reshape_activations(activations, k=3, stride=(1, 1), padding=(1, 1), groups=1):
    """
    N x C_in x H x W -> (N x H x W) x C_in.
    """
    
    if len(activations.size()) == 4:
        # gather activations
        a_padded = F.pad(activations, (padding[1], padding[1], padding[0], padding[0]))
        N, C, H, W = a_padded.size()
        a_stacked = []
        
        for i in range(0, H - k + 1, stride[0]):
            for j in range(0, W - k + 1, stride[1]):
                a_stacked.append(a_padded[:, :, i:i + k, j:j + k])
        
        # reshape according to weight
        a_reshaped = reshape_weight(torch.cat(a_stacked, dim=0)).t()
        
        # group convolutions (e.g. depthwise convolutions)
        a_reshaped_groups = torch.cat(a_reshaped.chunk(groups, dim=1), dim=0)
        
        return a_reshaped_groups
    
    else:
        return activations


def reshape_activations_cin(activations, k=3, stride=(1, 1), padding=(1, 1), groups=1, ps_sample=4):
    """
    Args:
        - activations: N x C_in x H x W
        - k: kernel size
    """
    ratio = int(math.sqrt(ps_sample))
    
    if len(activations.size()) == 4:
        N, cin, _, _ = activations.size()
        a_reshaped = torch.zeros(N, cin, k, k)
        a_padded = F.pad(activations, (padding[1], padding[1], padding[0], padding[0]))
        _, _, H_padded, W_padded = a_padded.size()
        
        for i in range(0, k):
            for j in range(0, k):
                a_stacked = a_padded[:, :, i:H_padded - k + i + 1:stride[0], j:W_padded - k + j + 1:stride[1]]
                dim2_index = torch.randint(low=0, high=a_stacked.size(2), size=(a_stacked.size(2) // ratio,)).long()
                dim3_index = torch.randint(low=0, high=a_stacked.size(3), size=(a_stacked.size(3) // ratio,)).long()
                a_stacked_sample = a_stacked[:, :, dim2_index, :]
                a_stacked_sample = a_stacked_sample[:, :, :, dim3_index]
                a_reshaped[:, :, i, j] = torch.sum(torch.sum(a_stacked_sample, dim=3), dim=2)
        
        return a_reshaped  # N x C_in x k x k
    
    else:
        return activations


############ cin version
def reshape_weightlike_cin(weight, d):
    c_out, c_in = weight.size()  # [C_out x C_in]
    fc_unroll = torch.cat(weight.chunk(c_in // d, dim=1), dim=0)  # cat (C_in /d * [C_out, d], dim=0)
    return fc_unroll


def reshape_back_weight_cin(weight, d, c_in):
    """
    (C_out x C_in /d x k x k) x d -> C_out x C_in x k x k.
    """
    fc_rollback = torch.cat(weight.chunk(c_in // d, dim=0), dim=1)
    return fc_rollback


def reshape_einsum_cin(weight, d):
    result = torch.einsum('lk->l*2 k//2', weight)


########### cout version
def reshape_weightlike_cout(weight, d):
    c_out, c_in = weight.size()
    # fc_unroll=torch.cat(weight.chunk(c_in//d,dim=1),dim=0) # cat (C_out /d * [d, C_in], dim=1)
    fc_unroll = torch.cat(weight.chunk(c_out // d, dim=0), dim=1).transpose(0, 1)  # cat (C_out /d * [d, C_in], dim=1)
    return fc_unroll


def reshape_back_weight_cout(weight, d, c_out):
    # fc_rollback=torch.cat(weight.chunk(c_in//d,dim=0),dim=1)
    fc_rollback = torch.cat(torch.chunk(weight.transpose(0, 1), chunks=c_out // d, dim=1), dim=0)
    return fc_rollback


######## optimized for hwpe dataflow
def reshape_back_hwpe(unroll, h, w, cin, cout, blockh, blockw):
    # now support cin<blockh
    '''
      blw *(h*w*(cin/blh)*(cout/blw)*blh) -> cout*cin*h*w
    '''
    new_tensor = torch.zeros(cout, cin, h, w).cuda()
    index = 0
    for i in range(h):
        for j in range(w):
            for l in range(math.ceil(cin / blockh)):
                for m in range(cout // blockw):
                    for n in range(cin % blockh):
                        new_tensor[m * blockw:m * blockw + blockw, l * cin // blockh + n, i, j] = unroll[:, index]
                        index += 1
    return new_tensor


def reshape_hwpe(weight, h, w, cin, cout, blockh, blockw):
    # now support cin<blockh
    '''
      cout*cin*h*w -> blw *(h*w*(cin/blh)*(cout/blw)*blh)
    '''
    new_tensor = torch.zeros(blockw, h * w * cin * cout // blockh).cuda()
    index = 0
    for i in range(h):
        for j in range(w):
            for l in range(math.ceil(cin / blockh)):
                for m in range(cout // blockw):
                    for n in range(cin % blockh):
                        new_tensor[:, index] = weight[m * blockw:m * blockw + blockw, l * cin // blockh + n, i,
                                               j].squeeze()
                        index += 1
    return new_tensor
