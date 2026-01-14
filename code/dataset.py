import json
import cv2
import h5py
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

# 运行指令：bash scripts/train.sh

class Geometry(object):
    def __init__(self, config):
        self.v_res = np.array(config['nVoxel'])     # [X, Y, Z]
        self.p_res = config['nDetector'][0]         # projections
        self.v_spacing = np.array(config['dVoxel']) # [sx, sy, sz]
        self.p_spacing = np.array(config['dDetector'])[0]
        self.DSO = config['DSO'] # mm
        self.DSD = config['DSD'] # mm

    def project(self, points, angle, scale_tensor=None, max_z=None):
        # points: [N, 3] ranging from [0, 1]
        # d_points: [N, 2] ranging from [-1, 1]

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = -(points[:, 2] - max_z/2)# [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

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
        dst_root = '/root/aicp-data/DIF-Net_jym/data'
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
        
        # 【修改 1】总是加载 z_length.json
        # 无论 split 是 train 还是 eval，都需要知道真实的 z_len
        z_len_path = os.path.join(data_root, 'z_length.json')
        if not os.path.exists(z_len_path):
            z_len_path = './z_length.json' # 备用路径
        
        with open(z_len_path, 'r') as f:
            self.z_lengths = json.load(f)
        print(f"Loaded Z-lengths for {len(self.z_lengths)} patients.")


        # 【修改 2】不再预先生成 self.points
        # 因为现在 Z 轴长度不固定，必须在 __getitem__ 里针对每个病人单独生成

        self.out_res = out_res
        self.data_root = data_root
        self.cfg = cfg
        self.name_list = name_list
        self.npoint = npoint
        self.is_train = (split == 'train')
        self.num_views = num_views
        self.random_views = random_views
        self.view_offset = view_offset

    # CT 归一化 (Min-Max: [0, 2500] -> [0, 1])
    def normalize_hu(self, data):
        # 范围设定：[0, 2500]
        # 注意：源数据中有 -258 的值，会被 clip 为 0
        min_val = 0.0
        max_val = 2500.0
        data = np.clip(data, min_val, None)
        data = (data - min_val) / (max_val - min_val)
        
        return data

    # X-ray 归一化 (ImageNet Standardization)
    def normalize_xray(self, data):
        # ImageNet 统计量 (针对单通道的近似处理)
        # Mean: 0.485, Std: 0.229
        mean = 0.485
        std = 0.229
        
        # 执行标准化 (x - mean) / std
        data = (data - mean) / std
        
        return data

    def __len__(self):
        return len(self.name_list)
    
    def sample_projections(self, name):
        # 硬编码读取路径
        xray_root = "/root/aicp-data/data-HDF5-512_ct512_plastimatch_xray/"
        path1 = os.path.join(xray_root, f"{name}_xray1.pfm") # PA
        path2 = os.path.join(xray_root, f"{name}_xray2.pfm") # Lateral
        p1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
        p2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)
        if p1 is None or p2 is None:
            raise FileNotFoundError(f"无法读取 X-ray PFM: {path1} 或 {path2}")

        # 强制双视角堆叠 [2, H, W]
        projs = np.stack([p1, p2])
        # 确保类型 float32
        projs = projs.astype(np.float32)

        # 归一化 (ImageNet Mean/Std)，数据已经是 0-1，直接标准化
        projs = self.normalize_xray(projs)

        # 增加通道维度 -> [2, 1, H, W]
        projs = projs[:, None, ...]
        
        # 固定角度: 0度 (PA) 和 90度 (Lateral)，对应弧度 [0, pi/2]
        angles = np.array([np.pi/2, 0.0], dtype=np.float32)
        
        return projs, angles
    
    def load_ct(self, name):
        # 验证集路径 (硬编码)
        val_root = "/root/aicp-data/val_data/"
        path = os.path.join(val_root, name, 'ct_xray_data.h5')
        
        try:
            with h5py.File(path, 'r') as f:
                image = f['ct'][:] 
        except Exception as e:
            raise FileNotFoundError(f"无法读取 H5 文件: {path}. 错误: {e}")
        
        image = image.astype(np.float32)
        
        # 应用 CT 归一化 [0, 2500] -> [0, 1]
        image = self.normalize_hu(image)
        
        # 转置适配 (X, Y, Z)
        image = np.transpose(image, (2, 1, 0))
        
        # # 【核心修改】强制三维缩放 (Force 3D Resize)
        # # 确保输出形状严格等于 (out_res, out_res, out_res)
        # target_shape = np.array([self.out_res, self.out_res, self.out_res])
        # current_shape = np.array(image.shape)
        
        # # 计算每个维度的缩放比例
        # # 例如: 512/512=1.0, 512/512=1.0, 512/450=1.137
        # zoom_factors = target_shape / current_shape
        
        # # 只要有任何一个维度不匹配，就进行缩放
        # if np.any(np.abs(zoom_factors - 1.0) > 1e-3):
        #     # order=1 (线性插值) 速度较快，order=3 (三次样条) 质量更好但慢
        #     # 考虑到 512^3 计算量巨大，建议先用 order=1 跑通，不影响代码运行
        #     image = scipy.ndimage.zoom(image, zoom_factors, order=1, prefilter=False)
            
        return image
    
    def load_block(self, name, b_idx):
        """只加载数值 (Pixel Values)"""
        block_root = "/root/aicp-data/block/" 
        path = os.path.join(block_root, name, f"block_{b_idx}.npz")
        
        # 读取一维数组 (N,)
        data = np.load(path)['arr_0'].astype(np.float32)
        
        # 归一化 HU 值
        data = self.normalize_hu(data)
        
        return data

    def sample_points(self, points, values=None):
        choice = np.random.choice(len(points), size=self.npoint, replace=False)
        points = points[choice]
        if values is not None:
            values = values[choice]
            # values 已经在 load_block 里归一化过了，不需要额外处理
            return points, values
        else: return points

    def get_coords_and_values(self, name, b_idx, block_values):
        """
        根据 b_idx 和 z_length 动态计算坐标，并与 block_values 匹配
        """
        z_len = self.z_lengths[name]
        
        # 确定形状，transpose(2,1,0)，所以是 (X, Y, Z) = (512, 512, z_len)
        shape = (512, 512, z_len)
        
        # 基础网格大小 (Base Grid Shape)，相当于 np.mgrid 的范围
        dx = shape[0] // 4 # 128
        dy = shape[1] // 4 # 128
        dz = shape[2] // 4 # 动态
        
        # 计算偏移量 offset (ox, oy, oz)，b_idx = x*16 + y*4 + z
        ox = b_idx // 16          # X offset
        oy = (b_idx % 16) // 4    # Y offset
        oz = b_idx % 4            # Z offset
        
        # 优化随机采样索引 (Indices)，不生成全量网格，直接从总数中选索引
        total_points = len(block_values) # Should be dx * dy * dz
        choice_indices = np.random.choice(total_points, size=self.npoint, replace=False)
        sampled_values = block_values[choice_indices]
        
        # 索引转坐标 (Index -> Coordinate)，将一维索引还原为 3D 网格索引 (u, v, w)
        u, v, w = np.unravel_index(choice_indices, (dx, dy, dz))
        
        # 计算绝对坐标，公式: coord = grid_index * 4 + offset，对应原逻辑: r_x + offset
        x_abs = u * 4 + ox
        y_abs = v * 4 + oy
        z_abs = w * 4 + oz
        
        # 堆叠成 [N, 3]
        points = np.stack([x_abs, y_abs, z_abs], axis=1).astype(np.float32)
        
        # 归一化到 [0, 1]
        points[:, 0] /= (self.out_res - 1)
        points[:, 1] /= (self.out_res - 1)
        points[:, 2] /= (self.out_res - 1)

        return points, sampled_values

    def __getitem__(self, index):
        name = self.name_list[index]

        # -- load projections
        projs, angles = self.sample_projections(name)
        # 在这里获取 Z 轴长度，因为 scale_vec 计算需要它
        real_z = self.z_lengths[name]
        scale_vec = np.array([1.0, 1.0, real_z / self.out_res], dtype=np.float32)

        # -- load sampling points
        if not self.is_train:
            # Eval 模式：动态生成网格，获取该病人的真实 Z 轴长度
            real_z = self.z_lengths[name]
            
            # 生成网格 (512, 512, real_z)
            points = np.mgrid[:self.out_res, :self.out_res, :real_z]

            # 归一化到 [0, 1]
            points = points.astype(np.float32)
            points[0] /= (self.out_res - 1) # X
            points[1] /= (self.out_res - 1) # Y
            points[2] /= (self.out_res - 1) # Z 
            
            # 变形为 (N, 3)
            points = points.reshape(3, -1).transpose(1, 0)
            
            # 加载真实图像用于计算 PSNR (形状是 512x512xReal_Z)
            image = self.load_ct(name)
            p_gt = np.zeros(len(points)) # eval 时不需要 gt，占位即可
            
        else:
            b_idx = np.random.randint(64) 
            block_values = self.load_block(name, b_idx)
            points, p_gt = self.get_coords_and_values(name, b_idx, block_values)
        max_z = (self.z_lengths[name] - 1) / (self.out_res - 1)

        # -- project points
        proj_points = []
        for a in angles:
            p = self.geo.project(points, a, scale_tensor=scale_vec, max_z=max_z)
            proj_points.append(p)
        proj_points = np.stack(proj_points, axis=0) 
        points = deepcopy(points) 
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = -(points[:, 2] - max_z/2)  
        points *= 2 # => [-1, 1]

        angles = np.array(angles, dtype=float) 
        angles = angles / np.pi * 2 - 1 
    
        ret_dict = {
            'name': name,
            'points': points,           
            'angles': angles[:, None],  
            'proj_points': proj_points, 
            'projs': projs,             
            'p_gt': p_gt[None, :]       
        }

        if not self.is_train:
            ret_dict['image'] = image


        return ret_dict

if __name__ == '__main__':
    dst = CBCT_dataset(dst_name='knee_cbct', random_views=True, num_views=10)
    item = dst[0]
    import pdb; pdb.set_trace()
    print("Points Shape:", item['points'].shape)
    print("Projections Shape:", item['projs'].shape)
    print("Projected Points Shape:", item['proj_points'].shape)
    