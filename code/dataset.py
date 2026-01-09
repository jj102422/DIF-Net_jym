import json
import cv2
import yaml
import scipy
import os
import pickle
from copy import deepcopy
import h5py

import numpy as np
import scipy
import torch
import yaml
from torch.utils.data import Dataset
from utils import read_nifti


class Geometry(object):
    def __init__(self, config):
        self.v_res = np.array(config['nVoxel'])     # [X, Y, Z]
        self.p_res = config['nDetector'][0]         # projections
        self.v_spacing = np.array(config['dVoxel']) # [sx, sy, sz]
        self.p_spacing = np.array(config['dDetector'])[0]
        self.DSO = config['DSO'] # mm
        self.DSD = config['DSD'] # mm

    def project(self, points, angle, scale_tensor=None):
        # points: [N, 3] ranging from [0, 1]
        # d_points: [N, 2] ranging from [-1, 1]

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] =  -(points[:, 2]- max(points[:, 2])/2) # [-z/2, z/2]
        # points *= self.v_res * self.v_spacing # mm
        physical_size = self.v_res * self.v_spacing
        if scale_tensor is not None:
             physical_size = physical_size * scale_tensor
        
        points *= physical_size

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        points = points @ rot_M.T

        d1 = self.DSO
        d2 = self.DSD
        
        coeff = (d2) / (d1 - points[:, 0]) # N,
        d_points = points[:, [2, 1]] * coeff[:, None] # [N, 2] float
        d_points /= (self.p_res * self.p_spacing)
        d_points *= 2 # NOTE: some points may fall outside [-1, 1]

        return d_points


class Mixed_CBCT_dataset(Dataset):
    def __init__(self, dst_list, **kwargs) -> None:
        super().__init__()
        print('mixed_dataset:', dst_list)
        self.name_list = dst_list
        self.datasets = []
        self.xray_root = '/root/aicp-data/data-HDF5-512_ct512_plastimatch_xray/'
        for dst_name in self.name_list:
            self.datasets.append(CBCT_dataset(dst_name, **kwargs))
    
    def __len__(self):
        dst_len = [len(d) for d in self.datasets]
        return np.sum(dst_len)
    
    @property
    def num_dst(self):
        return len(self.datasets)
    
    def find_dst(self, index):
        for i, dst in enumerate(self.datasets):
            if index >= len(dst):
                index -= len(dst)
            else:
                return i, index

    def __getitem__(self, index):
        dst_idx, index = self.find_dst(index)

        dst_vec = np.zeros(self.num_dst)
        dst_vec[dst_idx] = 1.

        data_dict = self.datasets[dst_idx][index]
        data_dict['dst_vec'] = dst_vec
        data_dict['dst_name'] = self.name_list[dst_idx]
        return data_dict


class CBCT_dataset(Dataset):
    def __init__(
            self,
            dst_name,
            split='train',
            num_views=10,
            npoint=5000,
            out_res=256,
            random_views=False,
            view_offset=0
        ):
        super().__init__()
        dst_root = './data'
        self.xray_root = '/root/aicp-data/data-HDF5-512_ct512_plastimatch_xray/'
        self.ct_root = '/home/public/CTSpine1K/data/ct512/'
        
        # load dataset info
        if dst_name in ['knee_cbct']:
            data_root = os.path.join(dst_root, dst_name)
            with open(os.path.join(data_root, 'info.json'), 'r') as f:
                cfg = json.load(f)
                name_list = sorted(cfg[split])
                print('CBCT_dataset, name: {}, split: {}, len: {}.'.format(dst_name, split, len(name_list)))
        else:
            raise ValueError(dst_name)

        # load projection config
        with open(os.path.join(data_root, cfg['projection_config']), 'r') as f:
            proj_cfg = yaml.safe_load(f)
            self.geo = Geometry(proj_cfg)

        # prepare points
        if split == 'train':
            # load blocks' coordinates [train only]
            # self.blocks = np.load(os.path.join(data_root, cfg['blocks']))
            # load blocks_z from z_length.json
            with open(os.path.join(data_root, 'z_length.json'), 'r') as f:
                self.z_length = json.load(f)
        else:
            # prepare sampling points
            points = np.mgrid[:out_res, :out_res, :out_res]
            points = points.astype(float) / (out_res - 1)
            points = points.reshape(3, -1)
            self.points = points.transpose(1, 0) # N, 3
        
        # other parameters
        self.out_res = out_res
        self.data_root = data_root
        self.cfg = cfg

        self.name_list = name_list
        self.npoint = npoint
        self.is_train = (split == 'train')
        self.num_views = num_views
        self.random_views = random_views
        self.view_offset = view_offset

    def __len__(self):
        return len(self.name_list)
    
    def sample_projections(self, name):
        # 构建文件路径
        path_pa = os.path.join(self.xray_root, f"{name}_xray1.pfm")
        path_lat = os.path.join(self.xray_root, f"{name}_xray2.pfm")
        
        # 检查文件是否存在
        if not os.path.exists(path_pa) or not os.path.exists(path_lat):
             raise FileNotFoundError(f"Missing X-ray files for {name}")

        img_pa = cv2.imread(path_pa, -1)
        img_lat = cv2.imread(path_lat, -1)
        
        projs = np.stack([img_pa, img_lat], axis=0)
        
        # 几何修正 (H, W 互换)
        projs = np.swapaxes(projs, -1, -2) 
        
        projs = projs.astype(np.float32)
        projs = projs[:, None, ...] # [2, 1, W, H]

        angles = np.array([-np.pi / 2, 0.0])
        return projs, angles
    # def sample_projections(self, name):
    #     # -- load projections
    #     with open(os.path.join(self.data_root, self.cfg['projections'].format(name)), 'rb') as f:
    #         data = pickle.load(f)
    #         projs = data['projections'] # K, 1, res^2
    #         angles = data['angles']     # K,

    #     # -- sample projections
    #     views = np.linspace(0, len(projs), self.num_views, endpoint=False).astype(int)
    #     offset = np.random.randint(len(projs) - views[-1]) if self.random_views else self.view_offset
    #     views += offset
    #     projs = projs[views].astype(float) / 255.
    #     projs = projs[:, None, ...]
    #     angles = angles[views]

        # -- de-normalization [required for mixed dataset]
        # projs = projs * self.cfg['projection_norm'] / 0.2
        
        # return projs, angles
    
    def normalize_hu(self, data):
        min_val = -1000.0  # 骨窗下限
        max_val = 3000.0   # 骨窗上限
        data = np.clip(data, min_val, max_val)
        data = (data - min_val) / (max_val - min_val)
        return data
    
    def load_ct(self, name):
        path = os.path.join(self.ct_root, name, 'ct_xray_data.h5')
        try:
           with h5py.File(path, 'r') as f:
                image = f['ct'][:] 
        except Exception as e:
            raise FileNotFoundError(f"无法读取 H5 文件: {path}. 错误: {e}")
        image = read_nifti(os.path.join(self.data_root, self.cfg['image'].format(name)))
        image = image.astype(np.float32)
        image = self.normalize_hu(image)
        image = np.transpose(image, (2, 1, 0))
        if self.out_res == 128:
            image = scipy.ndimage.zoom(image, 0.5, order=3, prefilter=False)
        elif self.out_res != 256:
            # 允许 512 或其他分辨率
            pass
        return image
    
    def load_block(self, name, b_idx):
        path = os.path.join(self.data_root, self.cfg['image_block'].format(name, b_idx))
        return np.load(path)['arr_0']
    
    def sample_points(self, points, values=None):
        choice = np.random.choice(len(points), size=self.npoint, replace=False)
        points = points[choice]
        if values is not None:
            values = values[choice]
            values = values.astype(float)
            values = self.normalize_hu(values)
            return points, values
        else: return points

    def get_block_coords(self, name, b_idx):
        """
        生成对应 Block 的 3D 坐标
        """
        # 【修正 1】形状必须是 (512, 512, z_len)
        # 对应你之前代码中的 np.transpose(image, (2, 1, 0)) -> (X, Y, Z)
        z_len = self.z_length[name]
        shape = (512, 512, z_len) 

        # 计算 offset (对应 generate_blocks 里的 x, y, z 循环顺序)
        ox = b_idx // 16          
        oy = (b_idx % 16) // 4
        oz = b_idx % 4
        offset = np.array([ox, oy, oz])

        # 【修正 2】生成网格
        # torch.meshgrid 配合 indexing='ij' 能够模拟 np.mgrid 的行为
        # 形状对应 shape: (X, Y, Z)
        r_x = torch.arange(shape[0] // 4) * 4
        r_y = torch.arange(shape[1] // 4) * 4
        r_z = torch.arange(shape[2] // 4) * 4
        
        grid = torch.meshgrid(r_x, r_y, r_z, indexing='ij')
        
        # 堆叠并展开 -> [3, N]
        base = torch.stack(grid, dim=0).reshape(3, -1)
        coords = base + torch.from_numpy(offset).view(3, 1)

        # 归一化到 [0, 1]
        points = coords.float().t() # [N, 3]
        points[:, 0] /= shape[0]
        points[:, 1] /= shape[1]
        points[:, 2] /= shape[2]

        return points.numpy()
    
    def __getitem__(self, index):
        name = self.name_list[index]

        # -- load projections
        projs, angles = self.sample_projections(name)

        # -- load sampling points
        if not self.is_train:
            points = self.points
            image = self.load_ct(name)
            p_gt = np.zeros(len(points))
        else:
            b_idx = np.random.randint(64) 
            block_coords = self.get_block_coords(name,b_idx)
            block_values = self.load_block(name, b_idx)
            points, p_gt = self.sample_points(block_coords, block_values)
            # 【修正 3】计算 Z 轴缩放比例
            # Config 中的 nVoxel[2] 是 512，但实际 Z 是 z_len
            # 我们需要告诉 project 函数，这个物体的 Z 轴只有 z_len 那么长
            config_z = self.geo.v_res[2] # 512
            real_z = self.z_length[name]
            z_scale = real_z / config_z

        # -- project points and view direction
        # 构造缩放向量 [1, 1, z_scale]
        scale_vec = np.array([1.0, 1.0, z_scale]) if self.is_train else None
        proj_points = []
        for a in angles:
            p = self.geo.project(points, a, scale_tensor=scale_vec)
            proj_points.append(p)
        proj_points = np.stack(proj_points, axis=0) # M, N, 2
        
        # -- normalize points
        points = deepcopy(points) # ~[0, 1]
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] =  -(points[:, 2]- max(points[:, 2])/2) # [-z/2, z/2]
        points *= 2 # => [-1, 1]

        # -- normalize viewing angles
        angles = np.array(angles, dtype=float) # ~[0, +pi]
        angles = angles / np.pi * 2 - 1 # => [-1, 1]

        # -- collect data
        ret_dict = {
            'name': name,
            'points': points,           # 3D points
            'angles': angles[:, None],  # angles
            'proj_points': proj_points, # projected points
            'projs': projs,             # 2D projections
            'p_gt': p_gt[None, :]       # labels
        }
        if not self.is_train:
            ret_dict['image'] = image

        return ret_dict


if __name__ == '__main__':
    dst = CBCT_dataset(dst_name='knee_zhao', random_views=True, num_views=10)
    item = dst[0]
    import pdb; pdb.set_trace()
    print("Points Shape:", item['points'].shape)
    print("Projections Shape:", item['projs'].shape)
    print("Projected Points Shape:", item['proj_points'].shape)