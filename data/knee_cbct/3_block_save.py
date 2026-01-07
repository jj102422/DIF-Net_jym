import json
import os
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


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

    files = glob(f"processed/*.nii.gz")
    json_path = "./z_length.json"
    z_length = {}
    for file in tqdm(files, ncols=50):
        name = file.split("/")[-1].split(".")[0]
        data_path = f"./processed/{name}.nii.gz"
        image = read_nifti(data_path)
        z_length[name] = image.shape[0]
        save_dir = f"./blocks/{name}/"
        os.makedirs(save_dir, exist_ok=True)

        block_list = generate_blocks(image.shape)
        for k, block in enumerate(block_list):
            block = block.reshape(3, -1).transpose(1, 0)
            image_block = image[block[:, 0], block[:, 1], block[:, 2]]
            np.savez(os.path.join(save_dir, f"block_{k}.npz"), image_block)

    # make z_length.json
    with open(json_path, "w") as f:
        json.dump(z_length, f)
