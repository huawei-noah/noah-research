import os
import numpy as np
import shutil
# weight_path="/home/ma-user/work/tianye/PocketLLM/llama2_7B_npy_v4/"

data_path="/home/ma-user/work/tianye/PocketLLM/llm_codebook/modify_results/"
######show log#######
file_list=os.listdir(data_path)
file_list.sort()
for folder_name in file_list:
    if folder_name==".ipynb_checkpoints":
        continue
    read_path=data_path+folder_name+'/log.txt'
    with open(read_path, 'r') as file:
        lines=file.readlines()
        last_line = lines[-2]+lines[-1]
    print(folder_name,last_line)

#####save weight#####
# weight_type='o'
# for weight_layer in range(3,30):
#     folder_name=weight_type+'_'+str(weight_layer)
#     if folder_name==".ipynb_checkpoints":
#         continue
#     read_path=data_path+folder_name+'/checkpoints/model_best.pth'
#     shutil.copyfile(read_path,data_path+folder_name+'/checkpoints/model_best_32768.pth')

#####showing weight#######
# weight_type='o'
# for weight_layer in range(3,30):
#     file_name=f'model.layers.{str(weight_layer)}.self_attn.{weight_type}_proj.npy'#file_list[0]
#     input_embeddings=np.load(weight_path+file_name)
#     print(weight_type,weight_layer,'——max:{:.11f},min:{:.11f}'.format(input_embeddings.max(),input_embeddings.min())," magnitude:",input_embeddings.max()-input_embeddings.min())

#########showing the shape #########
# weight_type='v'
# for weight_layer in range(3,30):
#     folder_name=weight_type+'_'+str(weight_layer)
#     model_path=data_path+folder_name+'/checkpoints/model_best.pth'
#     states = torch.load(model_path)
#     model_dict={}
#     for k,v in states.items():
#         # model_dict[k] = v
#         print(k,v.shape)
#     # model.load_state_dict(model_dict, strict=True)