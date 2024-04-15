import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .util import *


class Model(nn.Module):
    def __init__(self, points=1024, maxpool=True, embed_dim=32, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2]):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.points = points
        self.embedding1 = ConvBNReLU1D(3, embed_dim // 2, bias=bias, activation=activation)
        # self.embedding2 = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        self.embedding2 = ConvBNReLU1D(3, embed_dim // 2, bias=bias, activation=activation)
        # self.embedding2 = ConvBNReLU1D(embed_dim * 2, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.group_blocks_list = nn.ModuleList()
        self.sampling_blocks_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * 2
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_blocks_list
            local_grouper = GrouperBlock(channel=last_channel, groups=anchor_points, kneighbors=kneighbor,
                                         use_xyz=use_xyz, normalize=normalize)  # [b,g,k,d]
            self.group_blocks_list.append(local_grouper)
            # append sampling_blocks
            sampling_block = SamplingBlock(channel=last_channel, maxpool=maxpool, act=activation)
            self.sampling_blocks_list.append(sampling_block)
            # append pre_blocks_list
            pre_block_module = PreBlock(last_channel, out_channel, pre_block_num, groups=groups,
                                        res_expansion=res_expansion, maxpool=maxpool,
                                        bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_blocks_list
            pos_block_module = PosBlock(out_channel, pos_block_num, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )
        # self.normal_aligner = nn.Sequential(
        #     ConvBNReLU1D(in_channels=last_channel, out_channels=128, kernel_size=1, bias=False),
        #     ConvBNReLU1D(in_channels=128, out_channels=64, kernel_size=1, bias=False),
        #     ConvBNReLU1D(in_channels=64, out_channels=32, kernel_size=1, bias=False),
        #     ConvBNReLU1D(in_channels=32, out_channels=16, kernel_size=1, bias=False),
        #     ConvBNReLU1D(in_channels=16, out_channels=3, kernel_size=1, bias=False),
        # )

    def forward(self, pts, normals):
        x = pts.permute(0, 2, 1)
        normals = normals.permute(0, 2, 1)
        point_features = self.embedding1(x)  # B,D,N
        normal_features = self.embedding2(normals)
        # data = self.embedding2(normals)
        # data = self.embedding2(torch.concat([point_features, normal_features], dim=-2))
        data = torch.concat([point_features, normal_features], dim=-2)
        for i in range(self.stages):
            pts, data = self.group_blocks_list[i](pts, data.permute(0, 2, 1))
            # data = self.sampling_blocks_list[i](data)
            data = self.pre_blocks_list[i](data)
            data = self.pos_blocks_list[i](data)

        # data = self.normal_aligner(data)
        # data = F.adaptive_max_pool1d(data, 1).squeeze(dim=-1)
        # return data

        # data = F.adaptive_max_pool1d(data, 1).squeeze(dim=-1)
        # data = self.classifier(data)
        b, f, n = data.shape
        data = self.classifier(data.transpose(1,2).reshape(-1, f))
        # data = F.adaptive_max_pool1d(data.reshape(2,-1,3).transpose(1,2), 1).squeeze(-1)
        # data = F.adaptive_max_pool1d(data.reshape(2,-1,3).transpose(1,2), 1).squeeze(-1)
        data = F.adaptive_avg_pool1d(data.reshape(b, n, 3).transpose(1,2), 1).squeeze(-1)
        return data


class GrouperBlockOld(nn.Module):
    def __init__(self, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(GrouperBlockOld, self).__init__()
        self.groups = kwargs.get('groups')
        self.kneighbors = kwargs.get('kneighbors')
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, kwargs.get('channel') + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, kwargs.get('channel') + add_channel]))

    def get_params(self, inputs):
        return inputs['xyz'], inputs['normal_xyz'], inputs['normals'], inputs['points']

    def forward(self, **kwargs):
        xyz, normal_xyz, normals, points = self.get_params(kwargs)
        B, N, C = xyz.shape

        xyz = xyz.contiguous()  # xyz [batch, points, xyz]

        fps_idx = farthest_point_sample(xyz, self.groups).long()
        # 对每个批次的1024个点进行采样，输出512个点
        # 1024->512
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]

        new_points = index_points(points, fps_idx)  # [B, npoint, d]
        new_normals = index_points(normals, fps_idx)  # [B, npoint, d]

        # 找到每个点对应的采样的512个点中最近的24个点
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_points = index_points(points, idx)  # [B, npoint, k, 3]
        # grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        grouped_normals = index_points(normals, idx)
        if self.use_xyz:
            grouped_normals = torch.cat([grouped_normals, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_normals, dim=2, keepdim=True)
            if self.normalize == "anchor":
                pts_mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                nms_mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_normals
                # mean = torch.cat([normals, points],dim=-1) if self.use_xyz else normals
                pts_mean = pts_mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
                nms_mean = nms_mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            pts_std = torch.std((grouped_points - pts_mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(
                dim=-1).unsqueeze(dim=-1)
            nms_std = torch.std((grouped_normals - nms_mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(
                dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - pts_mean) / (pts_std + 1e-5)
            grouped_normals = (grouped_normals - nms_mean) / (nms_std + 1e-5)
            # grouped_normals = self.affine_alpha*grouped_normals + self.affine_beta

        # knn的12个点的feature与中心点的feature concat在一起返回
        # new_features = torch.cat([grouped_points, points.unsqueeze(dim=-2).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        new_points = torch.cat([grouped_points, new_points.unsqueeze(dim=-2).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        new_normals = torch.cat([grouped_normals, new_normals.unsqueeze(dim=-2).repeat(1, 1, self.kneighbors, 1)],
                                dim=-1)
        new_features = torch.cat([new_points, new_normals], dim=-1)  # 128 features
        # b, p, g, f = new_features.size()
        # new_features = self.conv1(new_features.view(-1,f,g))
        # new_features = F.relu(new_features)
        # xyz是原始点
        return new_features


class GrouperBlock(nn.Module):
    def __init__(self, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(GrouperBlock, self).__init__()
        self.groups = kwargs.get('groups')
        self.kneighbors = kwargs.get('kneighbors')
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, kwargs.get('channel') + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, kwargs.get('channel') + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [batch, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        # 对每个批次的1024个点进行采样，输出512个点
        fps_idx = farthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        # 1024->512
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        #
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        # 找到每个点对应的采样的512个点中最近的24个点
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points

    def save_pc(self, points):
        np.savetxt('/hz/code/pointmlp/PointCloud_hz/checkpoints/data1.txt', points)


class SamplingBlock(nn.Module):
    def __init__(self, channel=128, maxpool=True, act='relu'):
        super(SamplingBlock, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        # channel, channel//2, channel//8, channel//32
        # channels = [128, 64, 16, 4]
        channel *= 2
        channels = [channel, channel // 2, channel // 8, channel // 16]
        # self.pool = nn.MaxPool1d(channel // 16) if maxpool else nn.AvgPool1d(channel // 16)

        down_list = [ConvBNReLU1D(channels[0], channels[0], activation=act)]
        for i in range(len(channels) - 1):
            down_list.append(ConvBNReLU1D(channels[i], channels[i + 1], activation=act))
        self.down_sample = nn.Sequential(*down_list)

        up_list = [ConvTransposeBNReLU1D(channels[-1], channels[-1], activation=act)]
        # up_list = [ConvTransposeBNReLU1D(1, channels[-1], activation=act)]
        for i in range(len(channels) - 1, 0, -1):
            up_list.append(ConvTransposeBNReLU1D(channels[i], channels[i - 1], activation=act))
        self.up_sample = nn.Sequential(*up_list)

    def forward(self, x):
        b, n, k, f = x.size()
        x = x.view(-1, f, k)
        x = self.down_sample(x)
        # x = self.pool(x.permute(0, 2, 1))
        # x = self.up_sample(x.permute(0, 2, 1))
        x = self.up_sample(x)
        x = x.reshape(b, n, f, k).transpose(-1, -2)
        # B N K F(batch, N points, Knn, Features)
        return x


class PreBlock(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1.0, bias=True,
                 activation='relu', use_xyz=True, maxpool=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreBlock, self).__init__()
        # in_channels = 3+2*channels if use_xyz else 2*channels
        in_channels = 3 + 2 * channels if use_xyz else (channels * 2)
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)
        self.maxpool = maxpool

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        if self.maxpool:
            x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        else:
            x = torch.mean(x, 2, keepdim=True).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosBlock(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosBlock, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


def pointMLP_sampling(points, **kwargs) -> Model:
    model = Model(points=points, maxpool=False, embed_dim=32, groups=1, res_expansion=1.0,
                  activation="tanh", bias=False, use_xyz=False, normalize="anchor",
                  dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                  k_neighbors=[12, 12, 12, 12], reducers=[2, 2, 2, 2])
    return model


def pointMLP_elite_sampling(points, **kwargs) -> Model:
    model = Model(points=points, maxpool=False, embed_dim=32, groups=1, res_expansion=1.0,
                  activation="tanh", bias=False, use_xyz=False, normalize="anchor",
                  dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                  k_neighbors=[12, 12, 12, 12], reducers=[2, 2, 2, 2])
    return model


def pointMLP_small_sampling(points, **kwargs) -> Model:
    model = Model(points=points, maxpool=False, embed_dim=32, groups=1, res_expansion=1.0,
                  activation="tanh", bias=False, use_xyz=False, normalize="anchor",
                  dim_expansion=[2, 2], pre_blocks=[1, 2], pos_blocks=[1, 2],
                  k_neighbors=[12, 12], reducers=[2, 2])
    return model

def pointMLP_medium_sampling(points, **kwargs) -> Model:
    model = Model(points=points, maxpool=False, embed_dim=64, groups=1, res_expansion=1.0,
                  activation="tanh", bias=False, use_xyz=False, normalize="anchor",
                  dim_expansion=[2, 2], pre_blocks=[1, 2], pos_blocks=[1, 2],
                  k_neighbors=[24, 24], reducers=[2, 2])
    return model

def test_grouper():
    grouper = GrouperBlock(channel=32, groups=512, kneighbors=24, use_xyz=False, normalize="anchor")
    xyz = torch.rand(2, 4096, 3)
    normal_xyz = torch.rand(2, 4096, 3)
    normals = torch.rand(2, 4096, 32)
    points = torch.rand(2, 4096, 32)
    features = {}
    features['grouper'] = grouper
    p = grouper(xyz=xyz, normal_xyz=normal_xyz, normals=normals, points=points)
    print(1)


def test_sampling_block():
    block = SamplingBlock(channel=64, maxpool=False, act='relu')
    features = torch.rand(2, 4096, 24, 128)

    ret = block(features)
    print(1)


def test_posblock():
    block = SamplingBlock(maxpool=False, act='relu')
    features = torch.rand(2, 4096, 24, 128)

    ret = block(features)
    print(1)


def test_model():
    model = Model(points=1024, embed_dim=32, groups=1, res_expansion=1.0,
                  activation="relu", bias=False, use_xyz=False, normalize="anchor",
                  dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                  k_neighbors=[12, 12, 12, 12], reducers=[2, 2, 2, 2])
    pts = torch.rand(2, 4096, 3)
    normals = torch.rand(2, 4096, 3)
    res = model(pts, normals)


def test_out():
    class Mod(nn.Module):
        def __init__(self):
            super(Mod, self).__init__()
            self.conv1 = ConvBNReLU1D(in_channels=64, out_channels=32, kernel_size=1, bias=False)
            self.conv2 = ConvBNReLU1D(in_channels=32, out_channels=16, kernel_size=1, bias=False)
            self.conv3 = ConvBNReLU1D(in_channels=16, out_channels=8, kernel_size=1, bias=False)
            self.conv4 = ConvBNReLU1D(in_channels=8, out_channels=3, kernel_size=1, bias=False)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            F.adaptive_max_pool1d(x, 1).squeeze()
            return x

    data = torch.rand(2, 512, 64)
    m = Mod()
    res = m(data.permute(0, 2, 1))
    print(1)


if __name__ == '__main__':
    # test_grouper()
    # test_sampling_block()
    test_model()
    # test_out()
    print(1)
