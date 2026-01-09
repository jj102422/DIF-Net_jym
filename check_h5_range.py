import h5py
import numpy as np
import glob
import os

# 设置您的 H5 文件搜索路径
# 也可以直接指定某一个具体文件的路径，例如: file_path = "/path/to/your/file.h5"
search_path = "/home/public/CTSpine1K/data/ct512/**/ct_xray_data.h5"

# 获取第一个找到的文件进行检查
files = glob.glob(search_path, recursive=True)

if not files:
    print("未找到任何 H5 文件，请检查路径！")
else:
    # 只检查第一个文件通常就够了，或者您可以循环检查前几个
    file_path = files[0]
    print(f"正在检查文件: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 打印文件里所有的 keys，防止键名不是 'ct'
            print(f"Keys in H5 file: {list(f.keys())}")
            
            if 'ct' in f.keys():
                data = f['ct'][:]
                
                print("-" * 30)
                print(f"数据形状 (Shape): {data.shape}")
                print(f"数据类型 (Dtype): {data.dtype}")
                print(f"最小值 (Min): {data.min()}")
                print(f"最大值 (Max): {data.max()}")
                print(f"平均值 (Mean): {data.mean():.2f}")
                print("-" * 30)
                
                # 判断建议
                if data.min() >= 0 and data.max() <= 255:
                    print("结论: 数据看起来是 uint8 或 0-255 范围。")
                    print("建议: 代码中保留 / 255.0 是正确的。")
                elif data.min() < 0 or data.max() > 1000:
                    print("结论: 数据看起来是原始 CT 值 (HU)。")
                    print("建议: 代码中必须删除 / 255.0，并添加窗宽窗位归一化函数。")
                else:
                    print("结论: 数据范围不常见，请根据上述 Min/Max 自行判断。")
            else:
                print("错误: 找不到 'ct' 这个键，请检查上面的 Keys 列表。")
                
    except Exception as e:
        print(f"读取文件出错: {e}")
    