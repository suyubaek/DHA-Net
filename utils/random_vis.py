import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def create_vis_from_val(num_samples=12, seed=42):
    """
    从验证集中随机选择图像-掩码对，并将它们复制到可视化集目录中。

    Args:
        num_samples (int): 要选择的样本数量。
        seed (int): 用于可复现性的随机种子。
    """
    # 1. 定义路径
    base_dir = Path("/mnt/data1/suyubaek/dataset/S1_Water_256")
    val_dir = base_dir / "val"
    vis_dir = base_dir / "vis"

    # 源目录
    val_img_dir = val_dir / "img"
    val_mask_dir = val_dir / "mask"

    # 目标目录
    vis_img_dir = vis_dir / "img"
    vis_mask_dir = vis_dir / "mask"

    print("--- 正在从验证集创建可视化样本 ---")
    print(f"源目录 (val): {val_dir}")
    print(f"目标目录 (vis): {vis_dir}")

    # 2. 确保目标目录存在
    vis_img_dir.mkdir(parents=True, exist_ok=True)
    vis_mask_dir.mkdir(parents=True, exist_ok=True)
    print("目标目录已确认或创建。")

    # 3. 从 val/mask 获取所有样本文件名
    try:
        all_mask_files = [p.name for p in val_mask_dir.glob("*.tif")]
        if not all_mask_files:
            print(f"[错误] 在 {val_mask_dir} 中未找到任何 .tif 文件。")
            return
        print(f"在源掩膜目录中找到 {len(all_mask_files)} 个文件。")
    except FileNotFoundError:
        print(f"[错误] 源掩膜目录不存在: {val_mask_dir}")
        return

    # 4. 设置随机种子并随机抽样
    random.seed(seed)
    if len(all_mask_files) < num_samples:
        print(f"[警告] 样本总数 ({len(all_mask_files)}) 小于期望抽样数 ({num_samples})。将使用所有可用样本。")
        num_samples = len(all_mask_files)
        
    selected_files = random.sample(all_mask_files, num_samples)
    print(f"已随机选择 {len(selected_files)} 个样本 (种子: {seed})。")

    # 5. 复制文件
    print("\n--- 开始复制文件 ---")
    copied_count = 0
    for filename in tqdm(selected_files, desc="复制样本"):
        src_img_path = val_img_dir / filename
        dest_img_path = vis_img_dir / filename
        
        src_mask_path = val_mask_dir / filename
        dest_mask_path = vis_mask_dir / filename

        # 检查源文件是否存在
        if src_img_path.exists() and src_mask_path.exists():
            shutil.copy(src_img_path, dest_img_path)
            shutil.copy(src_mask_path, dest_mask_path)
            copied_count += 1
        else:
            print(f"\n[警告] 找不到对应的图像或掩膜，跳过: {filename}")

    print(f"\n--- 操作完成 ---")
    print(f"成功复制了 {copied_count} / {num_samples} 个图像-掩码对到 {vis_dir}")

if __name__ == "__main__":
    create_vis_from_val(num_samples=12, seed=42)
