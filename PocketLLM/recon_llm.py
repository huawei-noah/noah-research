import os
import sys
import time
import logging
import argparse

import torch.utils.data
from torch import optim
import torch.backends.cudnn as cudnn
from torchvision import  transforms
from torchvision.utils import save_image, make_grid

from scripts.util import setup_logging_from_args
from scripts.llm_encoder_rec import *
from scripts.data_loader import ImageDataset
from scripts.reshape import reshape_back_weight_cin
from thop import profile

dataset_train_args = {
    'llm': {'train': True, 'download': False},
}
dataset_test_args = {
    'llm': {'train': False, 'download': False},
}
nouse_arr=['ennorm0','ennorm3','en_addnorm0','en_addnorm1','denorm0','denorm3','de_addnorm0','de_addnorm1','enlinear',\
'delinear','mlp_encoder3','mlp_encoder4','mlp_encoder5','mlp_encoder6','mlp_encoder7','mlp_encoder8','mlp_encoder9',\
'mlp_decoder3','mlp_decoder4','mlp_decoder5','mlp_decoder6','mlp_decoder7','mlp_decoder8','mlp_decoder9',\
'en_addnorm3','en_addnorm4','en_addnorm5','en_addnorm6','en_addnorm7','en_addnorm8','en_addnorm9',\
'de_addnorm3','de_addnorm4','de_addnorm5','de_addnorm6','de_addnorm7','de_addnorm8','de_addnorm9']

def resume( model, path):
    if not os.path.exists(path):
        print("!!!!!!!!!!!!Checkpoint not found: " + path,"!!!!!!!!!!!!")
        return model
    states = torch.load(path)
    model_dict={}
    for k,v in states.items():
        if k.split('.')[0] in nouse_arr:
            continue
        print(k,v.shape)
        model_dict[k] = v
    model.load_state_dict(model_dict, strict=True)
    print("Resumed from " + path)
    return model

def main(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--ckptpath', default='vae',type=str,
                              help='checkpoint_path')
    model_parser.add_argument('--save_path', default='/home/ma-user/work/tianye/PocketLLM/llm_codebook/',type=str,
                              help='save_path of single weight')
    model_parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                              help='input batch size for training (default: 128)')
    model_parser.add_argument('--weight_layer', type=int, default=3,
                              help='the index of compressing layer')
    model_parser.add_argument('--weight_type', type=str, default='q',
                              help='the type of compressing layer')
    model_parser.add_argument('--hidden', type=int, metavar='N',
                              help='number of hidden channels')
    model_parser.add_argument('--inputer', type=int, metavar='N',default=4,
                              help='number of input channels')
    model_parser.add_argument('--rec_lenth', type=int, metavar='N',default=4096,
                              help='number of input channels')
    model_parser.add_argument('--k', '--dict-size', type=int, dest='k', metavar='K',
                              help='number of atoms in dictionary')
    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='llm',help='dataset to use llm')
    training_parser.add_argument('--data-dir', default='/media/ssd/Datasets',
                                 help='directory containing the dataset')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='0',
                                 help='gpus used for training - e.g 0,1,3')
    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=2000, metavar='N',
                                help='how many batches to wait before logging training status')
    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    dataset_dir_name = args.dataset 
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
        
    if args.weight_type =='down':
        if "llama3" in args.data_dir:
            args.rec_lenth=14336
            model = Mlp_VAE(in_features=args.inputer,hidden_features=args.hidden,k=args.k,expender=14336//args.inputer)#.half() #drop=0.1
        else:
            model = Mlp_VAE(in_features=args.inputer,hidden_features=args.hidden,k=args.k,expender=11008//args.inputer)#.half() #drop=0.1
            args.rec_lenth=11008
            
    # elif args.weight_type =='k' or args.weight_type =='v':
    #     if "llama3" in args.data_dir:
    #         args.rec_lenth=1024
    #     else:
    #         args.rec_lenth=4096
    #     model = Mlp_VAE(in_features=args.inputer,hidden_features=args.hidden,k=args.k,expender=4096//args.inputer)#.half()
    else:
        args.rec_lenth=4096
        model = Mlp_VAE(in_features=args.inputer,hidden_features=args.hidden,k=args.k,expender=4096//args.inputer)#.half()
        
    model=resume(model,args.ckptpath)
    if args.cuda:
        model.cuda()
    model.eval()
    
    ###mac and parameters
#     input_sentence=torch.randn(1024,4).cuda()
#     flops,params=profile(model,inputs=(input_sentence,))
#     print('Flops = '+str(flops/1000**3)+'G\n','Params = '+str(params/1000**2)+'M')
    
#     total = sum([param.nelement() for param in model.parameters()])
#     print('Number of parameters: %.4fM\n' % (total / 1e6))

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    dataset_test_dir = args.data_dir
    
    test_dataset=ImageDataset(dataset_test_dir,weight_type=args.weight_type,weight_layer=args.weight_layer)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False, **kwargs)
    
    y_name='y_'+args.weight_type+'_'+str(args.weight_layer)
    #获取权重列表
    s_time=time.time()
    rec_weight,y_weight,index_weight= test_net(1, model, test_loader, args.cuda,y_name,  args)
    print(time.time()-s_time)
    if args.weight_type in ['gate','up','down']:
        weight_name=f'model.layers.{str(args.weight_layer)}.mlp.{args.weight_type}_proj'
    else:
        weight_name=f'model.layers.{str(args.weight_layer)}.self_attn.{args.weight_type}_proj'
    
    ######save y weight#####
    # print("Save y done!")
#     y_weight=reshape_back_weight_cin(y_weight, args.inputer, args.rec_lenth).half()
#     np.save(f'/home/ma-user/work/tianye/PocketLLM/llama2_7B_npy_savey/'+weight_name+'.npy',y_weight.cpu().numpy())
    ######save index #####
    # print("Save index done!")
    # np.save(f'/home/ma-user/work/tianye/PocketLLM/llama2_7B_npy_saveindex/'+weight_name+'.npy',index_weight.cpu().numpy())

    # os.makedirs(os.path.join(args.save_path, 'rec_weight_ckpt'), exist_ok=True)
    #reshape weight
    save_weight=reshape_back_weight_cin(rec_weight, args.inputer, args.rec_lenth).half()
    print("from: ",rec_weight.shape," to: ",save_weight.shape)

    checkpoint_path = os.path.join(args.save_path, 'rec_weight_ckpt', f'{weight_name}.pth')
    torch.save({weight_name:save_weight},checkpoint_path) 
    print(f"Save the weight at {checkpoint_path}\n")

def test_net(epoch, model, test_loader, cuda, y_name,args):
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    
    re_weight=None
    y_weight=None
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if cuda:
                data = data.cuda()
            outputs = model(data)
            
            if i==0:
                re_weight=outputs[0]
                y_weight=outputs[4]
                index_weight=outputs[3]
            else:
                re_weight=torch.concat((re_weight,outputs[0]),dim=0)
                y_weight=torch.concat((y_weight,outputs[4]),dim=0)
                index_weight=torch.concat((index_weight,outputs[3]),dim=0)
                
                    
            model.loss_function(data, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
    # save_reconstructed_images(data, epoch, outputs[0], '/home/ma-user/work/tianye/PocketLLM/llm_codebook/modify_results/','reconstruction_train')
                
    for key in losses:
        losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
    loss_string = ' '.join(['{}: {:.12f}'.format(k, v) for k, v in losses.items()])
    # logging.info('====> Test set losses: {}'.format(loss_string))
    print('====> Test set losses: {}'.format(loss_string))
    
    return re_weight.squeeze(1),y_weight.squeeze(1),index_weight#[bs,4]

def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    # comparison = torch.cat([data[:n],
                            # outputs.view(batch_size, size[1], size[2])[:n]]).unsqueeze(1)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1])[:n]]).unsqueeze(1).unsqueeze(1)
    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n, normalize=True)

if __name__ == "__main__":
    main(sys.argv[1:])
