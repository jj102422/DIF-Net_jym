import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# 添加 code 目录到路径以便导入 dataset
sys.path.append(os.path.join(os.path.dirname(__file__), "../code"))

from dataset import CBCT_dataset


def vector_splatting(projs_gt, proj_coords, voxels, out_res=(512, 512)):
    """
    向量化 Splatting 投影验证
    Args:
        projs_gt: (M, 1, H, W) 真实的投影图像
        proj_coords: (M, N, 2) 归一化的投影坐标 [-1, 1]
        voxels: (N,) 展平的 3D 体素值 (CT值)
        out_res: (H, W) 输出分辨率
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Running validation on {device}...")

    M, N, _ = proj_coords.shape
    H, W = out_res

    # 1. 转 Tensor 并移至 GPU
    if not isinstance(proj_coords, torch.Tensor):
        proj_coords = torch.from_numpy(proj_coords).float().to(device)
    if not isinstance(voxels, torch.Tensor):
        voxels = torch.from_numpy(voxels).float().to(device)

    generated_projs = []

    for m in range(M):
        print(f"Processing view {m+1}/{M}...")
        # 2. 坐标映射: [-1, 1] -> [0, W-1] / [0, H-1]
        # dataset.py 中 d_points[:, 0] 对应 Z, d_points[:, 1] 对应 Y
        # 这通常对应图像的 u (宽) 和 v (高)
        # 注意: 如果图像看起来旋转了90度，可能需要交换 x, y 或者对结果做 transpose

        coord_x = (proj_coords[m, :, 0] + 1) / 2 * (W - 1)
        coord_y = (proj_coords[m, :, 1] + 1) / 2 * (H - 1)

        # 取整
        ix = torch.round(coord_x).long()
        iy = torch.round(coord_y).long()

        # 3. 筛选落在图像范围内的点
        valid_mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)

        valid_ix = ix[valid_mask]
        valid_iy = iy[valid_mask]
        valid_val = voxels[valid_mask]

        # 4. 创建画布并累加
        # 注意：这里模拟的是线积分
        canvas = torch.zeros((H * W), device=device, dtype=torch.float32)

        # 计算扁平化索引: idx = y * W + x
        flat_indices = valid_iy * W + valid_ix

        # 核心操作：累加
        canvas.index_add_(0, flat_indices, valid_val)

        # 恢复形状
        canvas = canvas.view(H, W)

        # 归一化以便显示 (Min-Max Nomalization)
        canvas_min = canvas.min()
        canvas_max = canvas.max()
        if canvas_max > canvas_min:
            canvas = (canvas - canvas_min) / (canvas_max - canvas_min)

        generated_projs.append(canvas.cpu().numpy())

    return generated_projs


def visualize_comparison(gt_projs, gen_projs, save_path="verify_result.png"):
    M = len(gen_projs)
    fig, axes = plt.subplots(2, M, figsize=(4 * M, 8))

    if M == 1:
        axes = axes[:, None]  # 保持二维结构

    for m in range(M):
        # Ground Truth
        # gt_projs shape: (M, 1, H, W)
        gt_img = gt_projs[m, 0]
        # 归一化 GT
        gt_img = (gt_img - np.min(gt_img)) / (np.max(gt_img) - np.min(gt_img) + 1e-8)

        axes[0, m].imshow(gt_img, cmap="gray")
        axes[0, m].set_title(f"Real X-ray (GT) View {m}")
        axes[0, m].axis("off")

        # Generated Splatting
        gen_img = gen_projs[m]

        # 注意：这里可能需要根据实际几何情况对生成图像进行 旋转 或 翻转
        # 例如: axes[1, m].imshow(np.rot90(gen_img), cmap='gray')
        axes[1, m].imshow(gen_img, cmap="gray")
        axes[1, m].set_title(f"Reprojected (Splatting) View {m}")
        axes[1, m].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Comparison saved to {save_path}")


if __name__ == "__main__":
    # 1. 加载数据
    print("Loading dataset...")
    # NOTE: 这里假设您能够成功创建一个 eval split 的 dataset
    # out_res 设置较小一点(如128)可以让调试跑得更快，但为了对齐 projs (512x512)，建议 eval 也用较高的分辨率，
    # 或者注意 voxels 的数量会随 out_res 变化。
    # 真实场景检验最好使用 out_res=256 或 512，即便慢一些。

    try:
        # 尝试创建一个样本
        # 注意：split='eval' 可能会加载大量数据，请确保内存充足
        # 如果只想测试几何，可以用 out_res=128 快速验证
        dst = CBCT_dataset(dst_name="knee_cbct", split="eval", out_res=256)
        idx = 0
        if len(dst) == 0:
            print("Dataset is empty. Please check info.json or paths.")
            exit()

        item = dst[idx]
        print(f"Loaded sample: {item['name']}")

        projs_gt = item["projs"]  # (M, 1, H, W)
        proj_points = item["proj_points"]  # (M, N, 2)
        image_3d = item["image"]  # (X, Y, Z) CT Volume

        # 2. 准备体素数据
        # 展平 image 作为 voxels
        # 关键: image_3d flatten 的顺序必须和 points 生成的顺序一致
        voxels = image_3d.flatten()

        print(f"Voxel count: {voxels.shape[0]}")
        print(f"Projection points: {proj_points.shape}")

        # 3. 执行投影
        # 注意: projs_gt 的分辨率是 (512, 512)，所以我们的画布也要是 (512, 512)
        gen_projs = vector_splatting(
            projs_gt, proj_coords=proj_points, voxels=voxels, out_res=(512, 512)
        )

        # 4. 可视化
        visualize_comparison(
            projs_gt,
            gen_projs,
            save_path="/root/aicp-data/DIF-Net_jym/scripts/verify_reprojection.png",
        )

    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback

        traceback.print_exc()
