import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm
import sys
import warnings

# --- 1. 全局配置 ---

# 您的数据集根目录
DATASET_DIR = Path("/mnt/data1/rove/dataset/S1_Water_512")
# 保存可视化结果的目录
QC_OUTPUT_DIR = Path("/home/rove/ThesisExp2026/utils/dataset_qc_visuals")
# 预期的图块大小 (在此处修改以适应不同的数据集)
EXPECTED_SIZE = 512
# 每个子集 (train/val/test) 随机抽查多少个?
N_SAMPLES_PER_SPLIT = 10

# --- 2. Check 3 & 4: 全量验证函数 ---

def validate_dataset(base_dir):
    """
    遍历所有图块，执行 Check 3 (NoData, Shape) 和 Check 4 (Distribution)
    """
    
    all_stats = []
    
    print("开始全量验证... 这可能需要一些时间。")

    for split in ["train", "val", "test"]:
        mask_dir = base_dir / split / "mask"
        img_dir = base_dir / split / "img"
        
        if not mask_dir.is_dir():
            print(f"\n[警告] {split} 目录 {mask_dir} 不存在，已跳过。")
            continue
            
        mask_files = list(mask_dir.glob("*.tif"))
        if not mask_files:
            print(f"\n[警告] {split} 目录 {mask_dir} 中没有 .tif 文件。")
            continue
            
        print(f"\n--- 正在验证 {split} 子集 ({len(mask_files)} 个图块) ---")
        
        for mask_path in tqdm(mask_files, desc=f"Validating {split}"):
            img_path = img_dir / mask_path.name
            
            file_stats = {
                "split": split,
                "filename": mask_path.name,
                "errors": [],
                "water_percent": -1.0
            }

            try:
                # --- Check 3: Shape (Mask) ---
                with rasterio.open(mask_path) as src_mask:
                    # 使用变量 EXPECTED_SIZE 替代硬编码的 256
                    if src_mask.shape != (EXPECTED_SIZE, EXPECTED_SIZE):
                        file_stats["errors"].append(f"MaskShapeError: {src_mask.shape} != {EXPECTED_SIZE}")
                    
                    # --- Check 4: Distribution ---
                    mask_data = src_mask.read(1)
                    water_pixels = np.sum(mask_data == 1)
                    total_pixels = mask_data.size
                    file_stats["water_percent"] = water_pixels / total_pixels

                if not img_path.exists():
                    file_stats["errors"].append("ImgMissing")
                    all_stats.append(file_stats)
                    continue
                    
                with rasterio.open(img_path) as src_img:
                    # --- Check 3: Shape (Image) ---
                    # 假设 [C, H, W] 格式
                    if src_img.height != EXPECTED_SIZE or src_img.width != EXPECTED_SIZE:
                        file_stats["errors"].append(f"ImgShapeError: {src_img.shape} != {EXPECTED_SIZE}")
                    
                    # --- Check 3: NoData ---
                    nodata_val = src_img.nodata
                    if nodata_val is not None:
                        img_data = src_img.read()
                        if np.any(img_data == nodata_val):
                            file_stats["errors"].append("HasNoData")
                
                all_stats.append(file_stats)

            except Exception as e:
                file_stats["errors"].append(f"ReadError: {e}")
                all_stats.append(file_stats)
                
    return pd.DataFrame(all_stats)

def report_results(df):
    """
    打印验证结果的摘要报告。
    """
    
    print("\n--- 全量验证报告 (Check 3 & 4) ---")

    # --- 报告 Check 3: 错误 ---
    print("\n[Check 3: 完整性与 NoData 检查]")
    
    if df.empty:
        print("  状态: 未找到任何可分析的图块。")
        return False # 返回失败
    
    df['has_error'] = df['errors'].apply(lambda x: len(x) > 0)
    error_df = df[df['has_error'] == True]

    if error_df.empty:
        print("  状态: 成功！")
        print(f"  所有 {len(df)} 个图块均通过了 Shape 和 NoData 检查。")
    else:
        print(f"  状态: 失败！发现 {len(error_df)} 个有问题的图块。")
        print("\n  错误摘要:")
        error_summary = df.explode('errors')['errors'].value_counts()
        print(error_summary)
        print("\n  [请检查上述错误图块，并考虑从数据集中删除它们]")

    # --- 报告 Check 4: 分布 ---
    print("\n[Check 4: 类别分布检查]")
    
    valid_df = df[df['has_error'] == False]
    if valid_df.empty:
        print("  未能分析分布（没有有效的图块）。")
        return False # 返回失败

    for split in valid_df['split'].unique():
        split_df = valid_df[valid_df['split'] == split]
        total_patches = len(split_df)
        
        print(f"\n  --- {split.capitalize()} 子集 (共 {total_patches} 个有效图块) ---")
        
        all_land_count = (split_df['water_percent'] == 0.0).sum()
        all_water_count = (split_df['water_percent'] == 1.0).sum()
        mixed_count = total_patches - all_land_count - all_water_count
        
        if total_patches > 0:
            print(f"    - 完全为陆地 (水=0%):   {all_land_count} ({all_land_count/total_patches:.1%})")
            print(f"    - 完全为水体 (水=100%): {all_water_count} ({all_water_count/total_patches:.1%})")
            print(f"    - 混合图块 (0%<水<100%): {mixed_count} ({mixed_count/total_patches:.1%})")
        else:
            print("    - 没有图块可供分析。")

    return True # 返回成功

def main_full_validation():
    """全量验证的主函数"""
    if not DATASET_DIR.is_dir():
        print(f"Error: 数据集目录不存在: {DATASET_DIR}")
        return False
        
    results_df = validate_dataset(DATASET_DIR)
    
    if not results_df.empty:
        return report_results(results_df)
    else:
        print("验证脚本未产生任何结果。")
        return False

# --- 3. Check 2: 目视检查函数 ---

def plot_sample(img_path, mask_path, output_png_path):
    """
    绘制并保存一个样本（VV, VH, Mask）。
    """
    try:
        with rasterio.open(img_path) as src_img:
            if src_img.count < 2:
                print(f"  [警告] 图像 {img_path.name} 波段数少于 2，跳过...")
                return
            img_vv = src_img.read(1)
            img_vh = src_img.read(2)
            
        with rasterio.open(mask_path) as src_mask:
            mask_data = src_mask.read(1)
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_vv, cmap='gray')
        axes[0].set_title(f"S1 - VV 波段")
        axes[0].axis('off')
        
        axes[1].imshow(img_vh, cmap='gray')
        axes[1].set_title(f"S1 - VH 波段")
        axes[1].axis('off')

        axes[2].imshow(mask_data, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title(f"S1 - Mask 蒙版")
        axes[2].axis('off')
        
        fig.suptitle(f"样本: {img_path.name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_png_path)
        plt.close(fig)
        
    except Exception as e:
        print(f"  [错误] 绘制 {img_path.name} 失败: {e}")

def main_visual_check():
    """目视检查的主函数"""
    if not DATASET_DIR.is_dir():
        print(f"Error: 数据集目录不存在: {DATASET_DIR}")
        return
        
    QC_OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n--- 目视检查报告 (Check 2) ---")
    print(f"开始进行目视检查... 结果将保存到: {QC_OUTPUT_DIR}")

    for split in ["train", "val", "test"]:
        print(f"\n--- 正在处理 {split} 子集 ---")
        mask_dir = DATASET_DIR / split / "mask"
        img_dir = DATASET_DIR / split / "img"
        
        if not mask_dir.is_dir():
            print(f"  目录不存在: {mask_dir}，已跳过。")
            continue
            
        mask_files = list(mask_dir.glob("*.tif"))
        if not mask_files:
            print(f"  {split} 子集没有找到 .tif 文件。")
            continue
            
        num_to_sample = min(N_SAMPLES_PER_SPLIT, len(mask_files))
        if num_to_sample == 0:
            print(f"  {split} 子集没有文件可供抽样。")
            continue

        samples = random.sample(mask_files, num_to_sample)
        
        print(f"  总共 {len(mask_files)} 个图块，随机抽取 {num_to_sample} 个进行可视化...")
        
        for mask_path in samples:
            img_name = mask_path.name
            img_path = img_dir / img_name
            output_png_path = QC_OUTPUT_DIR / f"{split}_{img_name.replace('.tif', '.png')}"
            
            if not img_path.exists():
                print(f"  [警告] 丢失匹配的图像文件: {img_path.name}，已跳过。")
                continue
                
            plot_sample(img_path, mask_path, output_png_path)
    
    print("\n目视检查完成。请检查 `dataset_qc_visuals` 文件夹中的 .png 文件。")

# --- 4. 主执行 ---

if __name__ == "__main__":
    
    # 依赖库: pip install rasterio matplotlib numpy pandas tqdm
    
    # 忽略 rasterio 在处理某些 TIF 时可能发出的无害警告
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    
    print("========================================")
    print("== 开始执行数据集质量控制 (QC) 脚本 ==")
    print("========================================")
    
    # --- 第一部分: 全量验证 ---
    validation_success = main_full_validation()
    
    print("\n----------------------------------------\n")
    
    # --- 第二部分: 目视检查 ---
    # 即使全量验证发现错误，我们仍然进行目视检查，这有助于 debug
    main_visual_check()
    
    print("\n========================================")
    print("== QC 脚本执行完毕 ==")
    print("========================================")