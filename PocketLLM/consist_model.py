import os
import torch
from transformers import LlamaForCausalLM
import shutil
llm_typer='llama3'

vector_length='4' 
if llm_typer=='llama1':
    data_path="/home/ma-user/work/tianye/PocketLLM/llm_codebook/rec_weight_ckpt_llama1/"
elif llm_typer=='llama2':
    data_path="/home/ma-user/work/tianye/PocketLLM/llm_codebook/rec_weight_ckpt_8_12/"
else:
    data_path="/home/ma-user/work/tianye/PocketLLM/llm_codebook/rec_weight_ckpt/"
    
weight_type=['k','q','v','o','gate','up','down']#,'k','q','v','o','gate','up','down'
rec_model_path="/home/ma-user/work/tianye/PocketLLM/"+llm_typer+"_7b_v"+vector_length+"_rec_all/"

weight_layer=[i for i in range(3,30)] #first 3 layers and the last layer

def rec_and_save( model, rec_dict):
    for name, module in model.named_modules():
        if name in rec_dict:
            module.weight.data=rec_dict[name]
            print("copying the weight ",name)
    return model.half()

def main():
    print("Start to load model!")
    if llm_typer=="llama1":
        llm_model = LlamaForCausalLM.from_pretrained('/home/ma-user/work/tianye/alpaca-lora/LLaMA-7B_hf/')
    elif llm_typer=='llama2':
        llm_model = LlamaForCausalLM.from_pretrained('/home/ma-user/work/tianye/alpaca-lora/Llama-2-7b-hf/') 
    else:
        llm_model = LlamaForCausalLM.from_pretrained('/home/ma-user/work/tianye/alpaca-lora/Llama-3-8B/') 
        
    # llm_model = LlamaForCausalLM.from_pretrained('/home/ma-user/work/tianye/PocketLLM/llama2_7b_v8_12_rec_alltemp2/') # compression ratio blend
    compressed_weight=[]
    
    # file_list=os.listdir(data_path)
    # file_list.sort()
    # for weight_name in file_list:
    #     save_weight=torch.load(data_path+weight_name) 
    #     compressed_weight.append(save_weight)
    #     print(f"Load the weight {weight_name}\n")

    for typer in weight_type:
        for layer in weight_layer:
            if typer in ['gate','up','down']:
                weight_name=f'model.layers.{str(layer)}.mlp.{typer}_proj.pth'
            else:
                weight_name=f'model.layers.{str(layer)}.self_attn.{typer}_proj.pth'
            save_weight=torch.load(data_path+weight_name) 
            compressed_weight.append(save_weight)
            print(f"Load the weight {weight_name}\n")
    
    for dicter in compressed_weight:
        llm_model=rec_and_save(llm_model,dicter) #
    llm_model.save_pretrained(rec_model_path)
    
    tokenizer_path=f"/home/ma-user/work/tianye/PocketLLM/tokenizer_{llm_typer}/"
    for t_file in os.listdir( tokenizer_path ):
        shutil.copy(tokenizer_path+t_file,rec_model_path+t_file)
        
    print("Reconstruct the float16 weight success. Save at ",rec_model_path)
    
if __name__ == "__main__":
    main()