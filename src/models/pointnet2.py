
# 来自pyg官方样例：
# examples/pointnet2_classification.py

"""
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)


def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                    'data/ModelNet10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
        train(epoch)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}')
    

"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MLP, PointNetConv, fps, radius, knn_interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x_interpolated = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x_interpolated = torch.cat([x_interpolated, x_skip], dim=1)
        x_out = self.nn(x_interpolated)
        return x_out

class DeepPointNet2(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        # Encoder
        self.sa1_module = SAModule(0.25, 0.2, MLP([in_channels + 3, 256, 256, 512]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([512 + 3, 256, 256, 512]))
        self.sa3_module = SAModule(0.25, 0.8, MLP([512 + 3, 256, 256, 512]))
        # Decoder
        self.fp3_module = FPModule(1, MLP([512 + 512, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 512, 256, 256]))
        self.fp1_module = FPModule(3, MLP([256 + in_channels, 256, 256, out_channels]))
        
        # Classification Head: 128 embedding -> num_classes logits
        #self.classifier = torch.nn.Linear(128, out_channels)

    def forward(self, data):
        sa0_pos, sa0_batch, sa0_x = data.pos, data.batch, data.x
        
        # Encoder
        sa1_x, sa1_pos, sa1_batch = self.sa1_module(sa0_x, sa0_pos, sa0_batch)
        sa2_x, sa2_pos, sa2_batch = self.sa2_module(sa1_x, sa1_pos, sa1_batch)
        sa3_x, sa3_pos, sa3_batch = self.sa3_module(sa2_x, sa2_pos, sa2_batch)

        # Decoder
        fp3_x = self.fp3_module(sa3_x, sa3_pos, sa3_batch, sa2_x, sa2_pos, sa2_batch)
        fp2_x = self.fp2_module(fp3_x, sa2_pos, sa2_batch, sa1_x, sa1_pos, sa1_batch)
        fp1_x = self.fp1_module(fp2_x, sa1_pos, sa1_batch, sa0_x, sa0_pos, sa0_batch)

        # Segmentation Logits
        #logits = self.classifier(fp1_x)
        return fp1_x  # , logits
