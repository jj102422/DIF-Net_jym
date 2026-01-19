import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import csv
import wandb

import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import Mixed_CBCT_dataset
from models.model import DIF_Net
from utils import convert_cuda, add_argument, save_nifti


# 新增一个辅助函数：提取3个方向中心切片并拼图
def get_center_slices(volume_gt, volume_pred):
    """
    输入形状: (W, H, D) 或 (X, Y, Z)
    输出: 一个包含 3个方向对比图的 wandb.Image 对象
    """
    # 确保数据在 0-1 之间并转为 uint8
    def to_uint8(vol):
        vol = np.clip(vol, 0, 1)
        return (vol * 255).astype(np.uint8)

    gt = to_uint8(volume_gt)
    pred = to_uint8(volume_pred)
    
    shape = gt.shape
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2

    # --- 1. Axial (XY平面, 取Z中心) ---
    # 形状: (X, Y)
    ax_gt = gt[:, :, cz]
    ax_pred = pred[:, :, cz]
    
    # --- 2. Coronal (XZ平面, 取Y中心) ---
    # 形状: (X, Z) -> 转置以便显示: (Z, X) 或者保持原样，视习惯而定
    cor_gt = np.rot90(gt[:, cy, :]) 
    cor_pred = np.rot90(pred[:, cy, :])

    # --- 3. Sagittal (YZ平面, 取X中心) ---
    # 形状: (Y, Z) -> 转置
    sag_gt = np.rot90(gt[cx, :, :])
    sag_pred = np.rot90(pred[cx, :, :])

    # 拼接图片 (左边是 GT, 右边是 Pred)
    # 使用 np.concatenate 拼接
    # 注意：如果尺寸不一致 (比如 Z轴长度不同)，需要 resize，这里假设 out_res=512 是各向同性的或者你接受拉伸
    
    img_list = []
    
    # Axial 对比图
    axial_combine = np.concatenate([ax_gt, ax_pred], axis=1) # 左右拼接
    img_list.append(wandb.Image(axial_combine, caption="Axial: GT vs Pred"))

    # Coronal 对比图
    cor_combine = np.concatenate([cor_gt, cor_pred], axis=1)
    img_list.append(wandb.Image(cor_combine, caption="Coronal: GT vs Pred"))

    # Sagittal 对比图
    sag_combine = np.concatenate([sag_gt, sag_pred], axis=1)
    img_list.append(wandb.Image(sag_combine, caption="Sagittal: GT vs Pred"))

    return img_list

def eval_one_epoch(model, loader, npoint=50000, save_dir=None, ignore_msg=True, use_tqdm=False, return_vis=False):
    model.eval()
    results = {}
    metrics = {}
    metrics_tmp = {key:[] for key in ['psnr', 'ssim']} # , 'rmse', 'mse', 'mae']}
    if use_tqdm:
        loader = tqdm(loader, ncols=50)
    
    vis_images = None # 用于存储可视化图片
    with torch.no_grad():
        for i, item in enumerate(loader):
            item = convert_cuda(item)

            dst_name = item['dst_name'][0]
            name = item['name'][0]
            image = item['image'].cpu().numpy()
            image = image[0] # W, H, D

            output = model(item, is_eval=True, eval_npoint=npoint) # B, 1, N
            output = output[0, 0].data.cpu().numpy()
            output = output.reshape(image.shape)
            # 强制将 GT (image) 和 预测值 (output) 限制在 [0, 1]
            image = np.clip(image, 0., 1.)
            output = np.clip(output, 0., 1.)

            # 提取第一个病人的切片用于 WandB
            if return_vis and i == 0:
                vis_images = get_center_slices(image, output)

            # 显式指定 data_range=1.0
            psnr = peak_signal_noise_ratio(image, output, data_range=1.0)
            ssim = structural_similarity(image, output, data_range=1.0)

            if not ignore_msg:
                print('{}, PSNR: {:.4}, SSIM: {:.4}'.format(
                    name, psnr, ssim
                ))

            dst_res = results.get(dst_name, [])
            dst_met = metrics.get(dst_name, deepcopy(metrics_tmp))

            dst_res.append({
                'name': name, 
                'psnr': psnr,
                'ssim': ssim,
            })
            for key in dst_met.keys():
                dst_met[key].append(dst_res[-1][key])
            
            results[dst_name] = dst_res
            metrics[dst_name] = dst_met

            if save_dir is not None:
                output = np.clip(output, 0, 1)
                output *= 255.
                output = output.astype(np.uint8)
                save_path = os.path.join(save_dir, f'{name}.nii.gz')
                save_nifti(output, save_path)

    for dst_name in metrics.keys():
        dst_met = metrics[dst_name]
        m = {key:np.mean(val) for key, val in dst_met.items()}
        metrics[dst_name] = m
    
    if return_vis:
        return metrics, results, vis_images
    else:
        return metrics, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')
    parser = add_argument(parser, train=False)
    args = parser.parse_args()
    print(args)

    # -- dataloader
    eval_loader = DataLoader(
        Mixed_CBCT_dataset(
            dst_list=args.dst_list.split('+'),
            split=args.split, 
            num_views=args.num_views,
            out_res=args.out_res,
            view_offset=args.view_offset,
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )

    # -- model, load ckpt
    ckpt_path = f'./logs/{args.name}/ep_{args.epoch}.pth'
    ckpt = torch.load(ckpt_path)
    print('load ckpt from', ckpt_path)
    
    model = DIF_Net(
        num_views=args.num_views,
        combine=args.combine
    )
    model.load_state_dict(ckpt)
    model = model.cuda()

    # -- output dir
    save_dir = None
    if args.visualize:
        save_dir = f'./logs/{args.name}/results/ep_{args.epoch}/predictions'
        os.makedirs(save_dir, exist_ok=True)

    # -- evaluate
    metrics, results = eval_one_epoch(
        model, 
        eval_loader, 
        args.eval_npoint,
        save_dir=save_dir,
        ignore_msg=False,
        use_tqdm=False
    )
    print(metrics)

    # -- save results
    pred_dir = f'./logs/{args.name}/results/ep_{args.epoch}'
    os.makedirs(pred_dir, exist_ok=True)

    csv_file = open(os.path.join(pred_dir, 'results.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['dataset', 'obj_id', 'psnr', 'ssim'])

    for dst_name in results.keys():
        dst_res = results[dst_name]
        for res in dst_res:
            csv_writer.writerow([dst_name, res['name'], res['psnr'], res['ssim']])

        dst_avg = metrics[dst_name]
        csv_writer.writerow([dst_name, 'average', dst_avg['psnr'], dst_avg['ssim']])
    
    csv_file.close()

    with open(os.path.join(pred_dir, 'args.json'), 'w') as f:
        args = vars(args)
        json.dump(args, f, indent=4)