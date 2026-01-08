import os
import glob

block_root = "/home/public/CTSpine1K/data/block/"
ct_root = "/home/public/CTSpine1K/data/ct512/"

# 统计ct512下有多少个子文件夹（病人）
ct_subfolders = [f for f in os.listdir(ct_root) if os.path.isdir(os.path.join(ct_root, f))]
num_folders = len(ct_subfolders)
print(f"ct512下共有子文件夹（病人）：{num_folders} 个")

# 统计block下所有npz文件的总大小
total_size = 0
block_folders = [os.path.join(block_root, f) for f in os.listdir(block_root) if os.path.isdir(os.path.join(block_root, f))]
for folder in block_folders:
    npz_files = glob.glob(os.path.join(folder, "*.npz"))
    folder_size = sum(os.path.getsize(f) for f in npz_files)
    total_size += folder_size

print(f"block下所有npz文件总大小：{total_size / 1024 / 1024:.2f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")

if len(block_folders) > 0:
    avg_size = total_size / len(block_folders)
    estimated_total = avg_size * num_folders
    print(f"预计全部{num_folders}个病人处理完后总大小约：{estimated_total / 1024 / 1024:.2f} MB ({estimated_total / 1024 / 1024 / 1024:.2f} GB)")