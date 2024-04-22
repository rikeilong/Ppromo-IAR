import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict

from net.ctrgcn_att import *

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, L = 2, drop_p = 0.1):
        super().__init__(
            nn.Linear(emb_size, L * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(L * emb_size, emb_size),
        )

class SA_block(nn.Sequential):
    def __init__(self, emb_size = 512, drop_p = 0.1, heads_attention = 4, forward_expansion = 2,
                forward_drop_p = 0.1):
        
        super(SA_block, self).__init__()
        self.form_q = nn.Linear(512,512)
        self.form_k = nn.Linear(512,512)
        self.form_v = nn.Linear(512,512)

        self.norm1 = nn.LayerNorm(emb_size)
        self.self_att = nn.MultiheadAttention(emb_size,heads_attention,drop_p)

        self.block2 = nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, L=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
        
    def forward(self, x):
        q,k,v = self.form_q(x),self.form_k(x),self.form_v(x)
        res = self.self_att(q,k,v)[0]
        res2 = self.block2(res + x)
        x = x + res2

        return x

class CA_block(nn.Sequential):
    def __init__(self, emb_size = 512, drop_p = 0.1, heads_attention = 4, forward_expansion = 2,
                forward_drop_p = 0.1):
        
        super(CA_block, self).__init__()
        self.attn_p2v = nn.MultiheadAttention(emb_size,heads_attention,drop_p,batch_first=False)
        self.attn_v2p = nn.MultiheadAttention(emb_size,heads_attention,drop_p,batch_first=False)

        self.norm1 = nn.LayerNorm(emb_size)
        self.fc_fusion = nn.Linear(2 * emb_size, emb_size)
    
    def forward(self, visual,skl):

        visual_feat_att = self.attn_p2v(skl, visual, visual, attn_mask=None, key_padding_mask=None)[0]
        pose_feat_att = self.attn_v2p(visual, skl, skl, attn_mask=None, key_padding_mask=None)[0]

        return visual_feat_att, pose_feat_att

class ppromo(nn.Module):

    def __init__(self, num_class,num_point,num_person,pose_graph,graph_args,pose_dir,in_channels,t_att):


        super(ppromo, self).__init__()
        self._num_classes = num_class

        gcn = Model(num_class,num_point,num_person,pose_graph,graph_args,in_channels,t_att=t_att).cuda()
        weights = torch.load(pose_dir)
        gcn.load_state_dict(weights)
        self.gcn = gcn

        # for name, parameter in self.gcn.named_parameters():
        #     parameter.requires_grad = False

        self.fc_skl = nn.Linear(256, 512)

        planes = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes * 4, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.attn_t = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.attn_s = nn.MultiheadAttention(512, 4, dropout=0.1)

        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear11 = nn.Linear(512, 512)
        self.linear22 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout11 = nn.Dropout(0.1)
        self.dropout22 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

        self.sa_visual = SA_block()
        self.sa_skl = SA_block()

        self.ca = CA_block()

        self.fc_fusion = nn.Linear(1024, 512)
        self.norm_fusion = nn.LayerNorm(512)
        self.dropout_fusion = nn.Dropout(0.1)

        self.classifier = nn.Linear(512, num_class)
    
    def forward(self, visual, skl, frame_indices):

        B,T,C,H,W=visual.size()
        #TGM
        t_weighted,g_skl = self.gcn.extract_temporal(skl,frame_indices)
        t_weighted,g_skl = t_weighted.cuda(),g_skl.cuda()
        visual = torch.mul(t_weighted,visual) #B,T,512,14,14

        visual = visual.view(B,C,T,H,W).type(torch.FloatTensor).cuda()

        out = self.conv1(visual)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        visual = visual + out

        temp_visual=visual.view(B*T,C,H,W)
        v_feat=self.avgpool(temp_visual)
        visual_feat=v_feat.squeeze(-1).squeeze(-1) # B T 512
        visual_feat = visual_feat.view(B,T,C)

        k = visual.mean(1).mean(1)[0]

        #B, 256
        g_skl = F.relu(self.fc_skl(g_skl))
        #SGM
        visual_feat = visual_feat.permute(1, 0, 2) # T B 512
        query_skl = g_skl.unsqueeze(0)
        visual_feat_att_t = self.attn_t(query_skl,visual_feat,visual_feat, attn_mask=None,key_padding_mask=None)[0].squeeze(0)
        src = self.linear1(self.dropout1(F.relu(self.linear11(visual_feat_att_t)))) #B 512
        visual_feat_att_t = visual_feat_att_t + self.dropout11(src) #B 512
        visual_feat_att_t = self.norm1(visual_feat_att_t) #B 512

        #Inter-learn
        visual_feat_global = self.sa_visual(visual_feat_att_t)
        skl_feat_global = self.sa_skl(g_skl)

        ca_v, ca_p= self.ca(visual_feat_att_t,g_skl)

        final_v = visual_feat_global + ca_v
        final_p = skl_feat_global + ca_p

        final_feat = torch.cat((final_v,final_p),dim=1)
        final_feat = F.tanh(final_feat)
        final_feat = self.norm1(self.fc_fusion(final_feat)) #B 512

        final_feat = self.classifier(final_feat)

        return final_feat
    

# if __name__ == "__main__":
#     num_classes = 31
#     input_tensor = torch.autograd.Variable(torch.rand(1, 3, 5, 224, 224))
#     skl = torch.autograd.Variable(torch.rand(1, 2, 4000, 17, 1))
#     frame_indices = torch.tensor([[[1,3,5,7,9]]]).cuda()
#     graph_args = {'layout': 'smarthome17','strategy': 'spatial'}
#     agcn = Model(31,17,1,'smarthome.Graph',graph_args,2,attention=True,t_att=True)
#     weights = torch.load('/mnt/disk72/yql/i3d_smarthome-main/weights/smarthome_cs_aagcn_joint-25-23140.pt')
#     agcn.load_state_dict(weights)
#     agcn.updata_spatial()
#     # input_tensor2 = torch.autograd.Variable(torch.rand(1, 3, 1, 224, 224))
#     i3d = InceptionI3d(400, in_channels=3)

#     i3d.load_state_dict(torch.load('./models/rgb_imagenet.pt'))
#     i3d.replace_logits(num_classes,agcn=agcn)

#     # output = i3d(input_tensor,skl,frame_indices)
#     # print(output.size())

#     t = get_parameter_number(i3d)
#     print(t)