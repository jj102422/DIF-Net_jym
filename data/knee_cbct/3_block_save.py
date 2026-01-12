import json
import os
import glob
import numpy as np
import h5py
import SimpleITK as sitk
from tqdm import tqdm

def read_h5_ct(path):
    with h5py.File(path, 'r') as f:
        # 假设ct数据在'ct'键下
        image = f['ct'][:]
    return image

def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_img)
    return image

def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)

def generate_blocks(shape):
    block_list = []
    base = np.mgrid[: shape[0] // 4, : shape[1] // 4, : shape[2] // 4] * 4  # 3, 64 ^ 3
    base = np.mgrid[: shape[0] // 4, : shape[1] // 4, : shape[2] // 4] * 4  # 3, 64 ^ 3
    base = np.mgrid[: shape[0] // 4, : shape[1] // 4, : shape[2] // 4] * 4  # 3, 64 ^ 3
    base = base.reshape(3, -1)
    for x in range(4):
        for y in range(4):
            for z in range(4):
                offset = np.array([x, y, z])
                block = base + offset[:, None]
                block_list.append(block)
    return block_list

if __name__ == "__main__":
    # 配置路径
    files = glob.glob("/home/public/CTSpine1K/data/ct512/**/ct_xray_data.h5", recursive=True)
    save_root = "/home/public/CTSpine1K/data/block/"
    json_path = "./z_length.json"
    
    z_length = {}
    print(f"Found {len(files)} files.")

    for file in tqdm(files, ncols=80):
        try:
            # 1. 获取名字
            name = os.path.basename(os.path.dirname(file))
            
            # 2. 读取数据
            image = read_h5_ct(file)
            
            # 3. 转置 (原始 Z,Y,X -> 转置后 X,Y,Z)
            # 您确认需要保留此操作，则 Z 轴变成了第 3 维 (index 2)
            image = np.transpose(image, (2, 1, 0)) 
            
            # 4. 记录 Z 轴长度
            # 因为转置了，Z轴现在是 image.shape[2]
            z_len = image.shape[2] 
            z_length[name] = z_len 
            
            # 5. 创建保存路径
            save_dir = os.path.join(save_root, name)
            os.makedirs(save_dir, exist_ok=True)
            
            # 6. 生成并保存 Block
            block_list = generate_blocks(image.shape)
            
            for k, block in enumerate(block_list):
                # 调整坐标形状 (3, N) -> (N, 3)
                block = block.reshape(3, -1).transpose(1, 0)
                
                # 【重点检查】索引提取
                # image 是 (X, Y, Z)，block 里的坐标也是对应的 (x, y, z)
                # 所以这里提取是正确的
                image_block = image[block[:, 0], block[:, 1], block[:, 2]]
                
                # 保存 (确保这一行在循环内，并且没有被注释掉)
                np.savez(os.path.join(save_dir, f"block_{k}.npz"), image_block)
        
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # 7. 循环结束后保存 z_length.json
    with open(json_path, "w") as f:
        json.dump(z_length, f, indent=4)
    
    print(f"Done! Z-length info saved to {json_path}")