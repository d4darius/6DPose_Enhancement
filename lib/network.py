import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)
        
        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        # print('img', img.size())
        # print('choose', choose.size())
        out_img = self.cnn(img)
        # print('out_img', out_img.size())
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        # print('emb', emb.size())
        choose = choose.long().to(emb.device)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        # print('choose', choose.size())
        emb = torch.gather(emb, 2, choose).contiguous()
        #print(choose[0, 1, :])
        x = x.transpose(2, 1).contiguous()
        # print('emb', emb.size())
        # print('x', x.size())
        ap_x = self.feat(x, emb)
        # print('ap_x', ap_x.size())

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))      

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)
        # print('rx', rx.size())
        # print('tx', tx.size())
        # print('cx', cx.size())
        out_rx = []
        out_tx = []
        out_cx = []

        for b in range(bs):
            b_rx = torch.index_select(rx[b], 0, obj[b])
            b_tx = torch.index_select(tx[b], 0, obj[b])
            b_cx = torch.index_select(cx[b], 0, obj[b])
            
            # Match original tensor transformation
            b_rx = b_rx.contiguous().transpose(2, 1).contiguous()
            b_cx = b_cx.contiguous().transpose(2, 1).contiguous()
            b_tx = b_tx.contiguous().transpose(2, 1).contiguous()
            
            out_rx.append(b_rx)
            out_tx.append(b_tx)
            out_cx.append(b_cx)

        # Stack back into batched tensors
        out_rx = torch.stack(out_rx)
        out_tx = torch.stack(out_tx)
        out_cx = torch.stack(out_cx)
        out_rx = out_rx.squeeze(1)
        out_tx = out_tx.squeeze(1)
        out_cx = out_cx.squeeze(1)
        
        return out_rx, out_tx, out_cx, emb.detach()
    
class GATFeat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATFeat, self).__init__()
        self.g_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.c_conv1 = torch.nn.Conv1d(32, 64, 1)


        self.gnn_conv1 = GATv2Conv(128, 256, heads=1, concat=True)
        self.gnn_conv2 = GATv2Conv(256, 512, heads=1, concat=True)
        self.gnn_conv3 = GATv2Conv(512, 1024, heads=1, concat=True)
        self.gnn_conv4 = GATv2Conv(1024, 1024, heads=1, concat=True)
        

    def forward(self, x, emb, graph_data):
        # We apply pointnet
        x = F.relu(self.g_conv1(x))
        emb = F.relu(self.c_conv1(emb))

        # 2-LAYER GNN (with skip connections)
        fused = torch.cat((x, emb), dim=1)        # (B, 128, N)
        fused = fused.permute(0, 2, 1).contiguous() # (B, N, 128)
        fused = fused.view(-1, 128)                 # (B*N, 128)
        graph_data.x = fused
        feat, edge_index = graph_data.x, graph_data.edge_index
        feat_1 = F.relu(self.gnn_conv1(feat, edge_index))
        feat_2 = F.relu(self.gnn_conv2(feat_1, edge_index))
        feat_3 = F.relu(self.gnn_conv3(feat_2, edge_index))
        feat_4 = self.gnn_conv4(feat_3, edge_index)
        
        return torch.cat([feat_1, feat_2, feat_4], dim=1) # (bs, 256+512+1024, 500)

class GNNPoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(GNNPoseNet, self).__init__()
        self.cnn = ModifiedResnet()
        self.feat = GATFeat()
        
        self.conv1_r = torch.nn.Conv1d(1792, 896, 1)
        self.conv1_t = torch.nn.Conv1d(1792, 896, 1)
        self.conv1_c = torch.nn.Conv1d(1792, 896, 1)

        self.conv2_r = torch.nn.Conv1d(896, 448, 1)
        self.conv2_t = torch.nn.Conv1d(896, 448, 1)
        self.conv2_c = torch.nn.Conv1d(896, 448, 1)

        self.conv3_r = torch.nn.Conv1d(448, 224, 1)
        self.conv3_t = torch.nn.Conv1d(448, 224, 1)
        self.conv3_c = torch.nn.Conv1d(448, 224, 1)

        self.conv4_r = torch.nn.Conv1d(224, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(224, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(224, num_obj*1, 1) #confidence

        self.num_points = num_points
        self.num_obj = num_obj

    def forward(self, img, x, choose, graph_data, obj):
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.long().to(emb.device)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        x = x.transpose(2, 1).contiguous()

        gnn_fusfeat = self.feat(x, emb, graph_data)
        gnn_fusfeat = gnn_fusfeat.view(bs, self.num_points, 1792).permute(0, 2, 1).contiguous()  # (bs, 768, num_points)

        rx = F.relu(self.conv1_r(gnn_fusfeat))
        tx = F.relu(self.conv1_t(gnn_fusfeat))
        cx = F.relu(self.conv1_c(gnn_fusfeat))      

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        out_rx = []
        out_tx = []
        out_cx = []

        for b in range(bs):
            b_rx = torch.index_select(rx[b], 0, obj[b])
            b_tx = torch.index_select(tx[b], 0, obj[b])
            b_cx = torch.index_select(cx[b], 0, obj[b])
            
            # Match original tensor transformation
            b_rx = b_rx.contiguous().transpose(2, 1).contiguous()
            b_cx = b_cx.contiguous().transpose(2, 1).contiguous()
            b_tx = b_tx.contiguous().transpose(2, 1).contiguous()
            
            out_rx.append(b_rx)
            out_tx.append(b_tx)
            out_cx.append(b_cx)

        # Stack back into batched tensors
        out_rx = torch.stack(out_rx)
        out_tx = torch.stack(out_tx)
        out_cx = torch.stack(out_cx)
        out_rx = out_rx.squeeze(1)
        out_tx = out_tx.squeeze(1)
        out_cx = out_cx.squeeze(1)
        
        return out_rx, out_tx, out_cx, emb.detach()


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))   

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        # b = 0
        # out_rx = torch.index_select(rx[b], 0, obj[b])
        # out_tx = torch.index_select(tx[b], 0, obj[b])
        out_rx = []
        out_tx = []
        
        for b in range(bs):
            b_rx = torch.index_select(rx[b], 0, obj[b])
            b_tx = torch.index_select(tx[b], 0, obj[b])
            
            out_rx.append(b_rx)
            out_tx.append(b_tx)

        # Stack back into batched tensors
        out_rx = torch.stack(out_rx)
        out_tx = torch.stack(out_tx)
        # out_rx = out_rx.squeeze(1)
        # out_tx = out_tx.squeeze(1)

        return out_rx, out_tx
