import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pspnet import PSPNet
from torch_geometric.nn import GINConv

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
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.long().to(emb.device)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

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
    
class GNNFeat(nn.Module):
    def __init__(self):
        super(GNNFeat, self).__init__()
        self.g_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.c_conv1 = torch.nn.Conv1d(32, 64, 1)

        self.mlp1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.gnn_conv1 = GINConv(self.mlp1)
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.gnn_conv2 = GINConv(self.mlp2)

        self.mlp3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )
        self.gnn_conv3 = GINConv(self.mlp3)

        self.fc = nn.Linear(64, 256)


    def forward(self, x, emb, graph_data, e):
        x = F.relu(self.g_conv1(x))
        emb = F.relu(self.c_conv1(emb))

        if e == "color":
            feat = emb
        else:
            feat = x
        B, C, N = feat.shape  # B=batch, C=64, N=num_points
        feat = feat.permute(0, 2, 1).contiguous().view(-1, 64)  # (B*N, 64)
        edge_index = graph_data.edge_index

        feat_1 = F.relu(self.gnn_conv1(feat, edge_index))  # (B*N, 128)
        feat_2 = F.relu(self.gnn_conv2(feat_1, edge_index))  # (B*N, 256)

        # Reshape back to (B, C, N)
        feat_1 = feat_1.view(B, N, 128).permute(0, 2, 1).contiguous()  # (B, 128, N)
        feat_2 = feat_2.view(B, N, 256).permute(0, 2, 1).contiguous()  # (B, 256, N)

        if e == "color":
            x_t = x.permute(0, 2, 1)  # (B, N, 64)
            oth_feat = F.relu(self.fc(x_t))  # (B, N, 256)
        else:
            emb_t = emb.permute(0, 2, 1)  # (B, N, 64)
            oth_feat = F.relu(self.fc(emb_t))  # (B, N, 256)
        oth_feat = oth_feat.permute(0, 2, 1).contiguous()  # (B, 256, N)

        fused = torch.cat([feat_2, oth_feat], dim=1)  # (B, 512, N)
        fused_flat = fused.permute(0, 2, 1).contiguous().view(-1, 512)  # (B*N, 512)
        feat_3 = self.gnn_conv3(fused_flat, edge_index)  # (B*N, 1024)
        feat_3 = feat_3.view(B, N, 1024).permute(0, 2, 1).contiguous()  # (B, 1024, N)

        return torch.cat([feat_1, feat_2, feat_3], dim=1)  # (B, 128+256+1024, N)

class GNNPoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(GNNPoseNet, self).__init__()
        self.cnn = ModifiedResnet()
        self.feat = GNNFeat()

        in_layers = 1408
        
        self.conv1_r = torch.nn.Conv1d(in_layers, 704, 1)
        self.conv1_t = torch.nn.Conv1d(in_layers, 704, 1)
        self.conv1_c = torch.nn.Conv1d(in_layers, 704, 1)

        self.conv2_r = torch.nn.Conv1d(704, 352, 1)
        self.conv2_t = torch.nn.Conv1d(704, 352, 1)
        self.conv2_c = torch.nn.Conv1d(704, 352, 1)

        self.conv3_r = torch.nn.Conv1d(352, 176, 1)
        self.conv3_t = torch.nn.Conv1d(352, 176, 1)
        self.conv3_c = torch.nn.Conv1d(352, 176, 1)

        self.conv4_r = torch.nn.Conv1d(176, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(176, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(176, num_obj*1, 1) #confidence

        self.num_points = num_points
        self.num_obj = num_obj

    def forward(self, img, x, choose, graph_data, obj, e):
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.long().to(emb.device)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        x = x.transpose(2, 1).contiguous()

        gnn_fusfeat = self.feat(x, emb, graph_data, e)  # (bs, 1408, num_points)
        # No need to reshape, just permute if needed for Conv1d
        gnn_fusfeat = gnn_fusfeat.contiguous()

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

class PosePredMLP(nn.Module):
    def __init__(self, input_dim=6, num_points=500):
        super(PosePredMLP, self).__init__()
        self.flattened_dim = input_dim * num_points  # 6 × 500 = 3000

        self.fc1 = nn.Linear(self.flattened_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 7)  # output: [quat (4) + trans (3)]
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
