import matplotlib.pyplot as plt
import numpy as np
weight_layer=6
weight_type='v'
data_dir='/home/ma-user/work/tianye/PocketLLM/llama2_7B_npy_v4/'
file_name=f'model.layers.{str(weight_layer)}.self_attn.{weight_type}_proj.npy'
input_embeddings=np.load(data_dir+file_name)
print(input_embeddings.shape)

print(input_embeddings.min(),input_embeddings.max())
fudu=0.038
length=18
step=fudu/length
nums=4194304

x_plus=[0 for i in range (length)]
x_minus=[0 for i in range (length)]
x0=0
out_nums=0
for embeddings in input_embeddings[:nums]:
    for index in range(4):
        number=int(embeddings[index]/step)
        if number >=1:
            if number>length:
                out_nums+=1
                continue
            number=min(number,length)
            x_plus[number-1]+=1/(nums*4)
        elif  number <=-1:
            if number<-length:
                out_nums+=1
                continue
            number=max(number,-length)
            number*=-1
            x_minus[length-number]+=1/(nums*4)
            
        else:
            x0+=1/(nums*4)   
                
print(x_minus,x0,x_plus)

print("holds num: ",1-out_nums/(4*nums))
x_cords=[]
miner=-1*fudu
for i in range (2*length+1):
    x_cords.append(round(miner, 16))
    miner+=step
y_cords=x_minus
y_cords.append(x0)
y_cords.extend(x_plus)

# draw a bar chart
plt.bar(x_cords, y_cords,width=0.0012)

# Set the title and axis labels
plt.title('Value Distribution Bar')
plt.xlabel('Value')
plt.ylabel('Proportion')
 
# show
plt.show()
plt.savefig('single_wave.png')