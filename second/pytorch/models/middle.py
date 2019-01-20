import time
from enum import Enum
from functools import reduce

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import spconv
import torchplus
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from second.pytorch.models.resnet import SparseBasicBlock, SparseBottleneck

class SparseMiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape
        middle_layers = []

        num_filters = [num_input_features] + num_filters_down1
        # num_filters = [64] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]

        for i, o in filters_pairs_d1:
            middle_layers.append(spconv.SubMConv3d(i, o, 3, bias=False, indice_key="subm0"))
            middle_layers.append(BatchNorm1d(o))
            middle_layers.append(nn.ReLU())
        middle_layers.append(
            spconv.SparseConv3d(
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        middle_layers.append(BatchNorm1d(num_filters[-1]))
        middle_layers.append(nn.ReLU())
        # assert len(num_filters_down2) > 0
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d2:
            middle_layers.append(spconv.SubMConv3d(i, o, 3, bias=False, indice_key="subm1"))
            middle_layers.append(BatchNorm1d(o))
            middle_layers.append(nn.ReLU())
        middle_layers.append(
            spconv.SparseConv3d(
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        middle_layers.append(BatchNorm1d(num_filters[-1]))
        middle_layers.append(nn.ReLU())
        self.middle_conv = spconv.SparseSequential(*middle_layers)

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

class SpMiddleD4HD(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleD4HD'):
        super(SpMiddleD4HD, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # num_input_features = 4
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2, padding=1), # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

class SpResNetD4HD(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpResNetD4HD'):
        super(SpResNetD4HD, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # num_input_features = 4
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, 3, indice_key="res0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SparseBasicBlock(32, 32, indice_key="res0"),
            SparseBasicBlock(32, 32, indice_key="res0"),
            SpConv3d(32, 64, 3, 2, padding=1), # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SparseBasicBlock(64, 64, indice_key="res1"),
            SparseBasicBlock(64, 64, indice_key="res1"),
            SpConv3d(64, 64, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SparseBasicBlock(64, 64, indice_key="res2"),
            SparseBasicBlock(64, 64, indice_key="res2"),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleD4HDLite(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleD4HDLite'):
        super(SpMiddleD4HDLite, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # num_input_features = 4
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2, padding=1), # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleD8HD(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleD8HD'):
        super(SpMiddleD8HD, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2, padding=1), # [800, 600, 41] -> [400, 300, 21]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2, padding=1),  # [400, 300, 21] -> [200, 150, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2, padding=[0, 1, 1]),  # [200, 150, 11] -> [100, 75, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),  # [100, 75, 5] -> [100, 75, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleFHD(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHD'):
        super(SpMiddleFHD, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2, padding=1), # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2, padding=1), # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )
        self.max_batch_size = 6
        # self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32).cuda()

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

class SpMiddleFHDV2(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHDV2'):
        super(SpMiddleFHDV2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2, padding=1), # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2, padding=1), # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseMaxPool3d([2, 1, 1]),
        )
        self.max_batch_size = 6
        self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32).cuda()

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size, self.grid)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

class SpMiddle2K(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        super(SpMiddle2K, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 8, 3, indice_key="subm0"), # [3200, 2400, 81] -> [1600, 1200, 41]
            BatchNorm1d(8),
            nn.ReLU(),
            SubMConv3d(8, 8, 3, indice_key="subm0"),
            BatchNorm1d(8),
            nn.ReLU(),
            SpConv3d(8, 16, 3, 2, padding=1), # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm1"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm1"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2, padding=1), # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm2"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm2"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2, padding=1), # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm4"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )
        self.max_batch_size = 3
        self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32).cuda()

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size, self.grid)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

class SpMiddleFHDLite(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHDLite'):
        super(SpMiddleFHDLite, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SpConv3d(num_input_features, 16, 3, 2, padding=1), # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2, padding=1), # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleResNetFHD(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleResNetFHD'):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="res0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SparseBasicBlock(16, 16, indice_key="res0"),
            SparseBasicBlock(16, 16, indice_key="res0"),
            SpConv3d(16, 32, 3, 2, padding=1), # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            nn.ReLU(),
            SparseBasicBlock(32, 32, indice_key="res1"),
            SparseBasicBlock(32, 32, indice_key="res1"),
            SpConv3d(32, 64, 3, 2, padding=1), # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SparseBasicBlock(64, 64, indice_key="res2"),
            SparseBasicBlock(64, 64, indice_key="res2"),
            SpConv3d(64, 128, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(128),
            nn.ReLU(),
            SparseBasicBlock(128, 128, indice_key="res3"),
            SparseBasicBlock(128, 128, indice_key="res3"),
            SpConv3d(128, 128, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleFHDLarge(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHDLarge'):
        super(SpMiddleFHDLarge, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2, padding=1), # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2, padding=1), # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 128, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(128, 128, 3, indice_key="subm3"),
            BatchNorm1d(128),
            nn.ReLU(),
            SubMConv3d(128, 128, 3, indice_key="subm3"),
            BatchNorm1d(128),
            nn.ReLU(),
            SubMConv3d(128, 128, 3, indice_key="subm3"),
            BatchNorm1d(128),
            nn.ReLU(),
            SpConv3d(128, 128, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()
        
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret
