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
from scripts.llm_encoder import *
from scripts.data_loader import ImageDataset

logging.basicConfig(level=logging.INFO)
dataset_train_args = {
    'llm': {'train': True, 'download': False},
}
dataset_test_args = {
    'llm': {'train': False, 'download': False},
}

dataset_transforms = {
    'llm': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}

def main(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    model_parser = parser.add_argument_group('Model Parameters')
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
    model_parser.add_argument('--k', '--dict-size', type=int, dest='k', metavar='K',
                              help='number of atoms in dictionary')
    model_parser.add_argument('--lr', type=float, default=None,
                              help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=None,
                              help='vq coefficient in loss')
    model_parser.add_argument('--commit_coef', type=float, default=None,
                              help='commitment coefficient in loss')
    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='llm',help='dataset to use llm')
    training_parser.add_argument('--data-dir', default='/media/ssd/Datasets',
                                 help='directory containing the dataset')
    training_parser.add_argument('--epochs', type=int, default=20, metavar='N',
                                 help='number of epochs to train (default: 10)')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='0',
                                 help='gpus used for training - e.g 0,1,3')
    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=16384, metavar='N',
                                help='how many batches to wait before logging training status')
    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    dataset_dir_name = args.dataset 
    if "llama1" in args.data_dir:
        save_path = f'./modify_results_llama1/{args.weight_type}_{args.weight_layer}'#setup_logging_from_args(args)
    else:
        save_path = f'./modify_results/{args.weight_type}_{args.weight_layer}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
    if args.weight_type=='down':
        if "llama3" in args.data_dir:
            model = Mlp_VAE(in_features=args.inputer,hidden_features=args.hidden,k=args.k,expender=14336//args.inputer)#.half() #drop=0.1
        else:
            model = Mlp_VAE(in_features=args.inputer,hidden_features=args.hidden,k=args.k,expender=11008//args.inputer)#.half() #drop=0.1
    else:  
        model = Mlp_VAE(in_features=args.inputer,hidden_features=args.hidden,k=args.k,expender=4096//args.inputer)#.half() #drop=0.1

    if args.cuda:
        model.cuda()
    
    # print('lr:',args.lr,'batch_size:',args.batch_size,'codebook_len:',args.k,"hidden_size:",args.hidden)
    logging.info(' lr: '+str(args.lr)+' batch_size: '+str(args.batch_size)+' codebook_len: '+str(args.k)+" hidden_size: "+str(args.hidden))
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    dataset_train_dir = args.data_dir#os.path.join(args.data_dir, dataset_dir_name)
    if args.dataset in ['imagenet', 'custom']:
        dataset_train_dir = os.path.join(dataset_train_dir, 'train')
        dataset_test_dir = os.path.join(dataset_test_dir, 'val')
    train_dataset=ImageDataset(dataset_train_dir,weight_type=args.weight_type,weight_layer=args.weight_layer)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=False, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.1,)#3 0.8 4 0.8
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=int(train_dataset.__len__()/args.batch_size)*args.epochs)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35,eta_min=args.lr*0.1)

    log_lines=""
    best_loger="No best epoch!!!"
    mini_loss=3e-3

    
    for epoch in range(1, args.epochs + 1):
        train_losses,log_line = train(epoch, model, train_loader, optimizer, args.cuda,
                             args.log_interval, save_path,args)#,best_log
        if float(train_losses['mse_train'])<mini_loss :#
            mini_loss=train_losses['mse_train']
            best_loger="Best epoch is "+str(epoch)+". The minimal mse_loss is "+str(mini_loss)+".\n"
            save_checkpoint(model, epoch, save_path)
        
        log_lines+=log_line+"\n"
        for k in train_losses.keys():
            name = k.replace('_train', '')
            train_name = k

        scheduler.step()
    with open( os.path.join(save_path,'log.txt'),'w') as filer:
        filer.write(log_lines+best_loger)
        

def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path,args):#, writer):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    best_log=""
    # if epoch%15==0:
    #     model.vq_coef*=10
    for batch_idx, data in enumerate(train_loader):
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        
        loss = model.loss_function(data, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])
        if batch_idx % log_interval == 0 and batch_idx>=1:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.12f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(data), total_batch=len(train_loader) * len(data),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()
            for key in latest_losses:
                losses[key + '_train'] = 0
        # if batch_idx == (len(train_loader) - 1):
        #     save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_train')

            # write_images(data, outputs, writer, 'train')

      #   if args.dataset in ['imagenet', 'custom'] and batch_idx * len(data) > args.max_epoch_samples:
      #       break
    for key in epoch_losses:
      epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    
    loss_string = '\t'.join(['{}: {:.12f}'.format(k, v) for k, v in epoch_losses.items()])
    log_line='====> Epoch: {} {}'.format(epoch, loss_string)
    logging.info(log_line)
    return epoch_losses,log_line#,best_log


def test_net(epoch, model, test_loader, cuda, save_path, args):#, writer):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if cuda:
                data = data.cuda()
            outputs = model(data)
            model.loss_function(data, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i == 0:
                # write_images(data, outputs, writer, 'test')
                save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_test')
                save_checkpoint(model, epoch, save_path)
            if args.dataset == 'imagenet' and i * len(data) > 1000:
                break

    for key in losses:
        if args.dataset not in ['imagenet', 'custom']:
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (i * len(data))
    loss_string = ' '.join(['{}: {:.12f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


# def write_images(data, outputs, writer, suffix):
#     original = data.mul(0.5).add(0.5) 
#     original_grid = make_grid(original[:6])
#     writer.add_image(f'original/{suffix}', original_grid)
#     reconstructed = outputs[0].mul(0.5).add(0.5)
#     reconstructed_grid = make_grid(reconstructed[:6])
#     writer.add_image(f'reconstructed/{suffix}', reconstructed_grid)


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


def save_checkpoint(model, epoch, save_path):
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(save_path, 'checkpoints', 'model_best.pth') #f'model_{epoch}.pth'
    torch.save(model.state_dict(), checkpoint_path) 


if __name__ == "__main__":
    main(sys.argv[1:])
