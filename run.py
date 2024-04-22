import os
import argparse
import heapq
from tqdm import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="/home/qilang/PythonProjects/ICME/Ppromo/config/smarthome-cs/train.yaml", type=str)
parser.add_argument("--save", default="/home/qilang/PythonProjects/ICME/Ppromo/weights/", type=str)
args = parser.parse_args()

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
from torchvision import datasets, transforms
from VideoDataset import Dataset

from net.Ppromo_fmw import ppromo
from net.ctrgcn_att import Model
# from dataset import *
from net.utils.Meter import *

from tensorboardX import SummaryWriter


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  

def run(model_name):


    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    cur_time = model_name + cur_time
    logdir = os.path.join('./runs', cur_time)
    writer = SummaryWriter(log_dir=logdir)

    setup_seed(3471)

    f = open(args.config)
    yaml_args = yaml.load(f,Loader = yaml.FullLoader)

    train_dataset = Dataset(**yaml_args['train_dataset_args'])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=yaml_args['batch_size'], shuffle=True, num_workers=0,pin_memory=True)    
    test_dataset = Dataset(**yaml_args['test_dataset_args'])
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=yaml_args['batch_size'], shuffle=True, num_workers=12,pin_memory=True)

    model = ppromo(**yaml_args['model_args']).cuda()

    lr = yaml_args['lr']
    optimizer = optim.SGD(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)

    criterion=nn.CrossEntropyLoss().cuda()

    # train it
    e = 0
    
    for epoch in range(int(e), 30):

        train_top1 = AverageMeter()
        train_top5 = AverageMeter()
        train_loss = AverageMeter()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        print("--------------------------EPOCH:{}--------------------------------".format(epoch))
        print("--------------------------TRAIN--------------------------------")
        model.train()
        with tqdm(total=len(dataloader), desc="Train") as pbar:
            for step, (frame_indices, inputs, skl, labels) in enumerate(dataloader):
                inputs = Variable(inputs.cuda())
                frame_indices = Variable(frame_indices.cuda(),requires_grad=False)
                skl = Variable(skl.mean(-1).unsqueeze(-1).cuda(),requires_grad=False)
                labels = Variable(labels.cuda(),requires_grad=False)
                        
                per_frame_logits = model(inputs,skl,frame_indices)
                loss = criterion(per_frame_logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                prec1_ac, prec5_ac = accuracy(per_frame_logits.data, labels, topk=(1, 5))

                train_loss.update(loss.item(), inputs.size(0))
                train_top1.update(prec1_ac.item(), inputs.size(0))
                train_top5.update(prec5_ac.item(), inputs.size(0))
                

                pbar.set_postfix({'loss' : '{:.4f}'.format(train_loss.val),'avg_loss' : '{:.4f}'.format(train_loss.avg),'top-1 && top-5' : '{:.1f} ,{:.1f}'.format(train_top1.val,train_top5.val)})
                pbar.update(1)

        scheduler.step()

        print ('TRAIN:EPOCH : {}, Tot Loss: {:.4f}, Top-1: {:.4f}, Top-5: {:.4f}'.format(epoch,train_loss.avg,train_top1.avg,train_top5.avg) )
        # save model
        torch.save({'epoch':epoch, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}, os.path.join(args.save,model_name))

        writer.add_scalar('train_loss_epoch', train_loss.avg, epoch)
        writer.add_scalar('train_top1_acc_epoch', train_top1.avg, epoch)
        writer.add_scalar('train_top5_acc_epoch', train_top5.avg, epoch)


        print("--------------------------VAL--------------------------------")
        # i3d.eval()
        model.eval()
        with tqdm(total=len(val_dataloader), desc="Test") as pbar:
            with torch.no_grad():
                for step, (frame_indices, inputs, skl, labels) in enumerate(val_dataloader):
                    inputs = Variable(inputs.cuda())
                    frame_indices = Variable(frame_indices.cuda(),requires_grad=False)
                    skl = Variable(skl.mean(-1).unsqueeze(-1).cuda(),requires_grad=False)
                    labels = labels.cuda()
                    pred = model(inputs,skl,frame_indices)

                    loss = criterion(pred, labels)

                    prec1, prec5 = accuracy(pred.data, labels, topk=(1, 5))
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

                    pbar.set_postfix({'loss' : '{:.4f}'.format(losses.val),'avg_loss' : '{:.4f}'.format(losses.avg),'top-1 && top-5' : '{:.1f} ,{:.1f}'.format(top1.val,top5.val)})
                    pbar.update(1)
        print ('TEST:EPOCH : {}, Tot Loss: {:.4f},  Top-1: {:.4f}, Top-5: {:.4f}'.format(epoch,losses.avg, top1.avg,top5.avg) )

        writer.add_scalar('val_loss_epoch', losses.avg, epoch)
        writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
        writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)


if __name__ == '__main__':
    run(model_name='ppromo_1')