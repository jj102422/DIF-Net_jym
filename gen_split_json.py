import json
import os

# ================= 配置区域 =================
# 请在这里填入你之前划分好的切片列表文件的路径
# 如果你没有文件，只有变量，也可以把路径列表直接赋值给 parse_patient_ids 函数
train_txt_path = "/root/aicp-data/train.txt"  # 修改为你的训练集txt路径
val_txt_path   = "/root/aicp-data/val.txt"    # 修改为你的验证集txt路径
test_txt_path  = "/root/aicp-data/test.txt"   # 修改为你的测试集txt路径

# 输出的 info.json 路径 (建议先存个临时的，确认没问题再覆盖)
output_json_path = "/root/aicp-data/DIF-Net_jym/data/knee_cbct/info.json"
# ===========================================

def parse_patient_ids(file_path):
    """
    读取切片路径列表，提取唯一的病人ID
    """
    patient_ids = set()
    
    if not os.path.exists(file_path):
        print(f"Warning: 文件不存在 {file_path}")
        return []

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    print(f"正在处理 {os.path.basename(file_path)}，共 {len(lines)} 行...")
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 示例路径: /root/aicp-data/.../volume-covid19-A-0215_ct/ct/axial_297.npz
        # 使用 / 分割
        parts = line.split('/')
        
        # 逻辑判断：提取病人 ID
        # 根据你提供的示例，病人ID在倒数第3级 (index -3)
        # parts[-1] = axial_297.npz
        # parts[-2] = ct
        # parts[-3] = volume-covid19-A-0215_ct  <-- 我们要这个
        
        # 为了更稳健，我们可以查找 'ct' 文件夹的前一级
        if 'ct' in parts:
            idx = parts.index('ct')
            patient_id = parts[idx - 1]
        else:
            # 如果路径结构不一致，回退到倒数第3个
            patient_id = parts[-3]
            
        patient_ids.add(patient_id)
    
    # 转为排序后的列表
    unique_ids = sorted(list(patient_ids))
    print(f" -> 提取到 {len(unique_ids)} 个唯一病人ID")
    return unique_ids

def main():
    # 1. 提取 ID
    train_ids = parse_patient_ids(train_txt_path)
    val_ids   = parse_patient_ids(val_txt_path)
    test_ids  = parse_patient_ids(test_txt_path)

    # 2. 检查是否有重叠 (可选，但推荐)
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    if not train_set.isdisjoint(val_set) or not train_set.isdisjoint(test_set):
        print("\n[警告] 训练集与验证/测试集存在重叠病人！请检查原始切片划分。")
        intersection = train_set.intersection(val_set).union(train_set.intersection(test_set))
        print(f"重叠病人: {intersection}")

    # 3. 构建 info.json 内容
    # 注意：前面的 image/blocks 等路径仅作为模板，实际加载逻辑在 dataset.py 里已经改写
    info_content = {
        "image": "ignored",
        "blocks": "ignored",
        "image_block": "ignored",
        "projections": "ignored",
        "projection_config": "./config.yaml",
        "projection_norm": 0.10,
        "train": train_ids,
        "eval": val_ids,
        "test": test_ids
    }

    # 4. 保存
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(info_content, f, indent=4)
    
    print(f"\n成功生成 info.json！保存路径: {output_json_path}")
    print(f"Train: {len(train_ids)}, Eval: {len(val_ids)}, Test: {len(test_ids)}")

if __name__ == "__main__":
    main()