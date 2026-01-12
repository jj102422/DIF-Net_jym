import os
import glob
import numpy as np
import re
import cv2  # OpenCV 通常支持读取 PFM

def read_pfm_manual(file_path):
    """
    备用方案：如果 cv2 读失败，使用手动解析 PFM 头
    """
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header not in ['PF', 'Pf']:
            raise Exception("不是 PFM 文件")
        
        dim_match = f.readline().decode('utf-8').rstrip()
        width, height = map(int, dim_match.split())
        
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if header == 'PF' else (height, width)
        
        data = np.reshape(data, shape)
        data = np.flipud(data) # PFM 通常是倒置的
        return data

def check_xray():
    # 您的 X-ray 根目录
    xray_root = "/root/aicp-data/data-HDF5-512_ct512_plastimatch_xray/"
    
    print(f"正在搜索 PFM 文件: {xray_root}")
    # 递归搜索所有 .pfm 文件
    files = glob.glob(os.path.join(xray_root, "**", "*.pfm"), recursive=True)
    
    if not files:
        print("错误：没找到任何 .pfm 文件，请检查路径是否正确！")
        return

    # 检查前 3 个文件
    for i, file_path in enumerate(files[:3]):
        print("-" * 40)
        print(f"检查文件 [{i+1}]: {file_path}")
        
        try:
            # 优先尝试 OpenCV 读取
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print("cv2 读取失败，尝试手动解析...")
                image = read_pfm_manual(file_path)
            
            print(f"  Shape: {image.shape}")
            print(f"  Type : {image.dtype}")
            print(f"  Min  : {np.min(image)}")
            print(f"  Max  : {np.max(image)}")
            print(f"  Mean : {np.mean(image)}")
            
            # 判断逻辑
            mx = np.max(image)
            if mx > 255:
                print("  => 结论：【高动态范围 Float】(可能是累积衰减值)。")
                print("     建议：不要除以 255，直接使用 ImageNet 标准化 (Normalize)。")
            elif mx > 1.1:
                print("  => 结论：【0-255 范围】。")
                print("     建议：Dataset 中需要先 / 255.0 转为 0-1，再做 ImageNet 标准化。")
            else:
                print("  => 结论：【0-1 范围】。")
                print("     建议：Dataset 中直接做 ImageNet 标准化。")
                
        except Exception as e:
            print(f"读取出错: {e}")

if __name__ == "__main__":
    check_xray()