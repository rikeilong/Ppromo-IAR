import os
import argparse
import heapq
from tqdm import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="/home/qilang/PythonProjects/ICME/Ppromo/config/smarthome-cs/train.yaml", type=str)
parser.add_argument("--model", default="/home/qilang/PythonProjects/ICME/Ppromo/weights/ppromo_cs.pth", type=str)
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


def val():

    f = open(args.config)
    yaml_args = yaml.load(f,Loader = yaml.FullLoader)

    test_dataset = Dataset(**yaml_args['test_dataset_args'])
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=yaml_args['batch_size'], shuffle=True, num_workers=12,pin_memory=True)

    model = ppromo(**yaml_args['model_args']).cuda()
    weights = torch.load(args.model)
    model.load_state_dict(weights['state_dict'])
    lr = yaml_args['lr']
    

    print("--------------------------VAL--------------------------------")
    model.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()

    with tqdm(total=len(val_dataloader), desc="Test") as pbar:
        with torch.no_grad():
            for step, (frame_indices, inputs, skl, labels) in enumerate(val_dataloader):
                inputs = Variable(inputs.cuda())
                frame_indices = Variable(frame_indices.cuda(),requires_grad=False)
                skl = Variable(skl.mean(-1).unsqueeze(-1).cuda(),requires_grad=False)
                labels = labels.cuda()
                pred = model(inputs,skl,frame_indices)

                prec1, prec5 = accuracy(pred.data, labels, topk=(1, 5))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
                pbar.set_postfix({'top-1 && top-5' : '{:.1f} ,{:.1f}'.format(top1.val,top5.val)})
                pbar.update(1)

    print ('Top-1: {:.4f}, Top-5: {:.4f}'.format(top1.avg,top5.avg) )

if __name__ == '__main__':
    val()