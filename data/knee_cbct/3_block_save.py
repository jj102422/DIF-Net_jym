import os
import numpy as np
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import h5py

def read_h5_ct(path):
    with h5py.File(path, 'r') as f:
        # 假设ct数据在'ct'键下，如有不同请修改
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
    base = base.reshape(3, -1)
    for x in range(4):
        for y in range(4):
            for z in range(4):
                offset = np.array([x, y, z])
                block = base + offset[:, None]
                block_list.append(block)
    return block_list


if __name__ == "__main__":

    files = glob("/home/public/CTSpine1K/data/ct512/**/ct_xray_data.h5", recursive=True)
    for file in tqdm(files, ncols=50):
        # 以父文件夹名作为name
        name = os.path.basename(os.path.dirname(file))
        data_path = file
        image = read_h5_ct(data_path)
        image = np.transpose(image, (2, 1, 0))

        save_dir = f"/home/public/CTSpine1K/data/block/{name}/"
        os.makedirs(save_dir, exist_ok=True)

        block_list = generate_blocks(image.shape)
        blocks = np.stack(block_list, axis=0)  # K, 3, N^3
        blocks = blocks.transpose(0, 2, 1).astype(float) / 511  # K, N^3, 3
        # np.savez(os.path.join(save_dir, f"blocks.npz"), blocks=blocks)
        for k, block in enumerate(block_list):
            block = block.reshape(3, -1).transpose(1, 0)
            image_block = image[block[:, 0], block[:, 1], block[:, 2]]
            np.savez(os.path.join(save_dir, f"block_{k}.npz"), image_block)
