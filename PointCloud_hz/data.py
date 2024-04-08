import os
import glob
import h5py
import numpy as np
import torch
from os import path as osp
from stl import mesh

from torch.utils.data import Dataset
from pointnet2_ops import pointnet2_utils
import random

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class TeethPointCloudData(Dataset):
    def __init__(self, args, partition='train', device='cuda', split_factor=0.9):
        self.partition = partition
        self.split_factor = split_factor
        self.split_data(args.data_path)
        self.sample_groups = args.get('sample_groups', 2048)
        self.device = device
        self.path = args.data_path
        self.data, self.labels = self.load_teethpc()

    def split_data(self, path):
        list_files = os.listdir(osp.join(path, 'meshes'))
        random.shuffle(list_files)
        split_index = int(len(list_files) * self.split_factor)
        self.train_data = list_files[:split_index]
        self.test_data = list_files[split_index:]


    def load_teethpc(self):
        list_files = self.train_data if self.partition == 'train' else self.test_data
        all_data = []
        all_labels = []
        for i in list_files:
            teeth_mesh = mesh.Mesh.from_file(osp.join(self.path, 'meshes', i))
            xyz = torch.from_numpy(teeth_mesh.centroids[np.newaxis, :, :]).to(self.device)
            if 'cpu' == self.device:
                fps_idx = self.farthest_point_sample(xyz, self.sample_groups)
            else:
                fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.sample_groups).long()
            xyz = self.index_points(xyz, fps_idx).squeeze(0).detach().cpu().numpy()
            normals = self.index_points(torch.from_numpy(teeth_mesh.normals[np.newaxis, :, :].copy()), fps_idx).squeeze(
                0).detach().cpu().numpy()
            # self.save_pc(xyz)
            # coord_vertices, coord_normals, _ = Read_stl(osp.join(path, i))
            # normals = teeth_mesh.normals
            # vectors = teeth_mesh.vectors
            # data = np.concatenate([vectors, normals[:, :, np.newaxis]], axis=-1)
            all_data.append((xyz, normals))
            label = np.loadtxt(osp.join(self.path, 'labels', i.split('.')[0] + '.txt'), dtype=np.float32)
            all_labels.append(label)
        return all_data, all_labels
    def save_pc(self, points):
        np.savetxt('/hz/code/pointmlp/PointCloud_hz/checkpoints/data1.txt', points)

    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        points = points[batch_indices, idx, :]
        return points

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
        # centroid.detach().cpu()
        # distance.detach().cpu()
        # farthest.detach().cpu()
        # batch_indices.detach().cpu()
        # dist.detach().cpu()
        # farthest.detach().cpu()
        return centroids

    def __getitem__(self, item):
        data = self.data[item]
        # normals = pc.normals
        # vectors = pc.vectors
        # data = np.concatenate([vectors, normals[:, :, np.newaxis]], axis=-1)
        label = self.labels[item]
        # if self.partition == 'train':
        # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
        # pointcloud = translate_pointcloud(pointcloud)
        # np.random.shuffle(data)
        return data, label

    def __len__(self):
        return len(self.data)


def compute_normal_vector(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal_vector = np.cross(v1, v2)
    return normal_vector


def calculate_center(points):
    x_sum = 0
    y_sum = 0
    z_sum = 0

    n = len(points)
    for p in points:
        x_sum += p[0]
        y_sum += p[1]
        z_sum += p[2]

    cx = x_sum / n
    cy = y_sum / n
    cz = z_sum / n
    return cx, cy, cz


if __name__ == '__main__':
    data = TeethPointCloudData("/hz/data/pointcloud/train/")
    print(1)
