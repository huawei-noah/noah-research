import os
import sys
from os.path import join
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import LlamaForCausalLM

def reshape_weightlike_cin(weight, d):
    c_out, c_in = weight.size()  # [C_out x C_in]
    fc_unroll = torch.cat(weight.chunk(c_in // d, dim=1), dim=0)  # cat (C_in /d * [C_out, d], dim=0)
    return fc_unroll

parent_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.abspath(parent_path))
os.chdir(parent_path)
print(f'>-------------> parent path {parent_path}')
print(f'>-------------> current work dir {os.getcwd()}')

from llm_codebook.LLMCodebook.utils.utils import codebook, codebook_level

torch_version = int(torch.__version__.split('.')[1])


def main():
    vector_len=4 #4
    llm_typer="llama3"
    if llm_typer=="llama1":
        base_model='/home/ma-user/work/tianye/alpaca-lora/LLaMA-7B_hf/' 
    elif llm_typer=="llama2":
        base_model='/home/ma-user/work/tianye/alpaca-lora/Llama-2-7b-hf/' 
    else:
        base_model='/home/ma-user/work/tianye/alpaca-lora/Llama-3-8B/'
        
    device = 'cuda' 
    ignor_layers = [0, 1, 2, 30, 31]
    Cmodules = ['up_proj', 'down_proj', 'gate_proj','q_proj', 'k_proj', 'v_proj', 'o_proj']#, 'up_proj', 'down_proj', 'gate_proj','q_proj', 'k_proj', 'v_proj', 'o_proj'
    
    
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            low_cpu_mem_usage=True if torch_version >= 1.9 else False
    )
    save_path=f'/home/ma-user/work/tianye/PocketLLM/{llm_typer}_7B_npy_v{str(vector_len)}/' #llama2_7B_npy_v
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    if device != "cpu":
        model.half()
    model.to(device)
    
    model.zero_grad()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and len(name.split('.')) > 3:
            layer_index = int(name.split('.')[2])
            layer_type = name.split('.')[4]
            if layer_index not in ignor_layers and layer_type in Cmodules:
                reshape_weight = reshape_weightlike_cin(module.weight.data.detach(), vector_len)
                print(module.weight.shape)
                np.save(f'/home/ma-user/work/tianye/PocketLLM/{llm_typer}_7B_npy_v{vector_len}/'+\
                        name+'.npy',reshape_weight.cpu().numpy())


if __name__ == '__main__':
    main()
