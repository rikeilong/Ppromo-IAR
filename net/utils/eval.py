import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'
import argparse
import heapq
from tqdm import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument("--frames-path", default="../frames/", type=str)
parser.add_argument("--csv-path", default="./Labels/custom_split_7_3/", type=str)

parser.add_argument('--mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('--save_model', default='weights/', type=str)
# parser.add_argument('--root', default='', type=str)
parser.add_argument('--protocol', default='CS', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
from torchvision import datasets, transforms
from VideoDataset import Dataset

from net.ctrgcn_att import Model
# from dataset import *
from Meter import *

from tensorboardX import SummaryWriter


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速

def run(init_lr=0.01,  batch_size=16, save_model='/weights/', protocol='CS'):

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    cur_time = 'i3d_fused(0.2,bigger)_ctrgcn_att_crop_cs_' + cur_time
    logdir = os.path.join('./runs', cur_time)
    writer = SummaryWriter(log_dir=logdir)

    setup_seed(3471)

    num_classes = 31

    # train_dataset = Dataset('/mnt/sda/smarthome_res18_14x14/', "/home/qilang/PythonProjects/ICME/smarthome/xsub/train_data_joint.npy",
    #                         "/home/qilang/PythonProjects/ICME/smarthome/xsub/train_label.pkl", 
    #                         '/home/qilang/PythonProjects/ICME/frames/')
    # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,pin_memory=True)
    test_dataset = Dataset('/mnt/sda/smarthome_res18_14x14/', "/home/qilang/PythonProjects/ICME/smarthome/xsub/val_data_joint.npy",
                            "/home/qilang/PythonProjects/ICME/smarthome/xsub/val_label.pkl", 
                            '/home/qilang/PythonProjects/ICME/frames/')

    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,   num_workers=12,pin_memory=True)    


    graph_args = {'layout': 'smarthome17','strategy': 'spatial'}
    gcn = Model(31,17,1,'smarthome.Graph',graph_args,2,t_att=True).cuda()
    weights = torch.load('/home/qilang/PythonProjects/ICME/Ppromo/smarthome_cs_ctrgcn_joint-47-64080.pt')
    gcn.load_state_dict(weights)
    gcn = nn.DataParallel(gcn,device_ids=[0,1])



    lr = init_lr
    # optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9)
    # optimizer_1 = optim.Adam(i3d.parameters(),lr=lr)
    # optimizer_2 = optim.SGD(agcn.parameters(),lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)
    # scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, 5, gamma=0.1, last_epoch=-1)
    # scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, 5, gamma=0.1, last_epoch=-1)


    # lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
    criterion=nn.CrossEntropyLoss().cuda()

    soft_loss = nn.KLDivLoss(reduction="batchmean")

    # train it
    # e = 0
    
    # for epoch in range(int(e), 30):

    #     train_top1 = AverageMeter()
    #     train_top5 = AverageMeter()
    #     train_loss = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    #     print("--------------------------EPOCH:{}--------------------------------".format(epoch))
    #     print("--------------------------TRAIN--------------------------------")
    #     i3d.train()
    #     # agcn.train()
    #     with tqdm(total=len(dataloader), desc="训练集") as pbar:
    #         for step, (_,frame_indices, inputs, skl, fused, labels) in enumerate(dataloader):
    #     # while steps < max_steps:
    #     #     print ('Step {}/{}'.format(steps, max_steps))
    #     #     print ('-' * 10)

    #             # optimizer_1.zero_grad()
    #             # wrap them in Variable
                
    #             # inputs = Variable(inputs.cuda())
    #             fused = Variable(fused.cuda())
    #             frame_indices = Variable(frame_indices.cuda(),requires_grad=False)
    #             skl = Variable(skl.cuda(),requires_grad=False)
    #             labels = Variable(labels.cuda(),requires_grad=False)

    #             #gcn
    #             # pred,t_extract,t_weights = agcn(skl)
    #             # pred,t_weighted = agcn(skl,frame_indices)
    #             # for b in range(t_extract.shape[0]):

    #             #     frames = inputs[b].size(2)
    #             #     times = t_extract[b][0:frames].tolist()
    #             #     max_number = heapq.nlargest(32, times)
    #             #     max_index = []
    #             #     for n in max_number:
    #             #         max_index.append(times.index(n))
    #             #     max_index.sort()
    #             #     frame_indices = max_index
                        
    #             per_frame_logits = i3d(fused,skl,frame_indices)

        
    #             loss = criterion(per_frame_logits, labels)
                
    #             # loss = criterion(per_frame_logits,labels)
    #             optimizer_1.zero_grad()
    #             loss.backward()
    #             # optimizer_1.step()
    #             optimizer_1.step()

    #             # prec1, prec3 = accuracy(per_frame_logits.data, labels, topk=(1, 3))
    #             prec1_ac, prec5_ac = accuracy(per_frame_logits.data, labels, topk=(1, 5))

    #             train_loss.update(loss.item(), inputs.size(0))
    #             train_top1.update(prec1_ac.item(), inputs.size(0))
    #             train_top5.update(prec5_ac.item(), inputs.size(0))

    #             # _, indices = torch.topk(per_frame_logits, k=5, dim=1, largest=True, sorted=True)
                

    #             pbar.set_postfix({'loss' : '{:.4f}'.format(train_loss.val),'avg_loss' : '{:.4f}'.format(train_loss.avg),'top-1 && top-5' : '{:.1f} ,{:.1f}'.format(train_top1.val,train_top5.val)})
    #             pbar.update(1)

    #     # scheduler.step(train_total_loss) #若loss一直不变，则自动降低学习率
    #     scheduler_1.step()

    #     print ('TRAIN:EPOCH : {}, Tot Loss: {:.4f}, Top-1: {:.4f}, Top-5: {:.4f}'.format(epoch,train_loss.avg,train_top1.avg,train_top5.avg) )
    #     # save model
    #     # torch.save({'epoch':epoch, 'state_dict':i3d.module.state_dict(), 'optimizer':optimizer.state_dict()}, f'./weights/i3d_fused(0.2,bigger)_t_att_cs.pth')
    #     torch.save({'epoch':epoch, 'state_dict':i3d.module.state_dict(), 'optimizer':optimizer_1.state_dict()}, f'./weights/i3d_fused(0.2,bigger)_s_att_crop_cs.pth')

    #     writer.add_scalar('train_loss_epoch', train_loss.avg, epoch)
    #     writer.add_scalar('train_top1_acc_epoch', train_top1.avg, epoch)
    #     writer.add_scalar('train_top5_acc_epoch', train_top5.avg, epoch)


    print("--------------------------VAL--------------------------------")
    # i3d.eval()
    gcn.eval()
    with tqdm(total=len(val_dataloader), desc="验证集") as pbar:
        with torch.no_grad():
            for step, (frame_indices, inputs, skl, labels) in enumerate(val_dataloader):
                skl = skl.mean(-1).unsqueeze(-1).cuda()
                # inputs = inputs.cuda()
                # frame_indices = frame_indices.cuda()
                labels = labels.cuda()
                # pred = i3d(fused)
                # pred,t_extract,t_weights = agcn(skl)
                # _,t_weighted = agcn(skl,frame_indices)
                    
                pred = gcn(skl)

                loss = criterion(pred, labels)

                prec1, prec5 = accuracy(pred.data, labels, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                pbar.set_postfix({'loss' : '{:.4f}'.format(losses.val),'avg_loss' : '{:.4f}'.format(losses.avg),'top-1 && top-5' : '{:.1f} ,{:.1f}'.format(top1.val,top5.val)})
                pbar.update(1)
    epoch = 1
    print ('TEST:EPOCH : {}, Tot Loss: {:.4f},  Top-1: {:.4f}, Top-5: {:.4f}'.format(epoch,losses.avg, top1.avg,top5.avg) )

    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)


if __name__ == '__main__':
    # need to add argparse
    run(batch_size=16, save_model=args.save_model, protocol=args.protocol)