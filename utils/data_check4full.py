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

# 您的数据集根目录 (更新为全尺寸数据集路径)
DATASET_DIR = Path("/mnt/data1/rove/dataset/S1_Water_Full")
# 保存可视化结果的目录
QC_OUTPUT_DIR = Path("/home/rove/ThesisExp2026/utils/dataset_qc_visuals_full")
# 每个子集 (train/val/test) 随机抽查多少个?
N_SAMPLES_PER_SPLIT = 5  # 全图比较大，抽查数量可以适当减少

# --- 2. Check 3 & 4: 全量验证函数 ---

def validate_dataset(base_dir):
    """
    遍历所有全尺寸图像，执行 Check 3 (一致性, NoData) 和 Check 4 (分布)
    """
    
    all_stats = []
    
    print("开始全量验证 (Full Scene Mode)... 这可能需要一些时间。")

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
            
        print(f"\n--- 正在验证 {split} 子集 ({len(mask_files)} 个场景) ---")
        
        for mask_path in tqdm(mask_files, desc=f"Validating {split}"):
            img_path = img_dir / mask_path.name
            
            file_stats = {
                "split": split,
                "filename": mask_path.name,
                "errors": [],
                "water_percent": -1.0,
                "shape": None
            }

            try:
                if not img_path.exists():
                    file_stats["errors"].append("ImgMissing")
                    all_stats.append(file_stats)
                    continue

                # --- Check 3: Shape Consistency (Mask vs Img) ---
                with rasterio.open(mask_path) as src_mask, rasterio.open(img_path) as src_img:
                    
                    mask_shape = (src_mask.height, src_mask.width)
                    img_shape = (src_img.height, src_img.width)
                    file_stats["shape"] = img_shape

                    # 检查图像和掩膜尺寸是否匹配
                    if mask_shape != img_shape:
                        file_stats["errors"].append(f"ShapeMismatch: Img{img_shape} vs Mask{mask_shape}")
                    
                    # --- Check 3: NoData ---
                    nodata_val = src_img.nodata
                    # 注意：对于全图，读取整个数组可能很慢，我们只读取概览或分块检查
                    # 这里为了简单起见，我们只检查中心区域或降采样区域是否存在 NoData
                    # 或者如果内存允许，读取全图
                    try:
                        # 读取全图进行统计 (如果内存不足，可改为 read(1, out_shape=...))
                        img_data = src_img.read()
                        mask_data = src_mask.read(1)
                        
                        if nodata_val is not None:
                            if np.any(img_data == nodata_val):
                                file_stats["errors"].append("HasNoData")
                        
                        # --- Check 4: Distribution ---
                        water_pixels = np.sum(mask_data == 1)
                        total_pixels = mask_data.size
                        file_stats["water_percent"] = water_pixels / total_pixels
                        
                    except MemoryError:
                        file_stats["errors"].append("MemoryError(TooLarge)")

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
    print("\n[Check 3: 完整性与一致性检查]")
    
    if df.empty:
        print("  状态: 未找到任何可分析的文件。")
        return False 
    
    df['has_error'] = df['errors'].apply(lambda x: len(x) > 0)
    error_df = df[df['has_error'] == True]

    if error_df.empty:
        print("  状态: 成功！")
        print(f"  所有 {len(df)} 个场景均通过了尺寸一致性检查。")
    else:
        print(f"  状态: 失败！发现 {len(error_df)} 个有问题的文件。")
        print("\n  错误摘要:")
        error_summary = df.explode('errors')['errors'].value_counts()
        print(error_summary)

    # --- 新增: 报告尺寸分布 ---
    print("\n[Check 3.5: 图像尺寸分布]")
    if 'shape' in df.columns:
        # 过滤掉 None (如果有错误导致没读到shape)
        valid_shapes = df[df['shape'].notna()]['shape']
        
        if not valid_shapes.empty:
            shape_counts = valid_shapes.value_counts()
            print(f"  发现 {len(shape_counts)} 种不同的尺寸组合 (Height, Width):")
            # 只打印前10个最常见的，避免列表过长
            for shape, count in shape_counts.head(10).items():
                print(f"    - {shape}: {count} 个场景")
            if len(shape_counts) > 10:
                print(f"    - ... 以及其他 {len(shape_counts) - 10} 种组合")
                
            heights = valid_shapes.apply(lambda x: x[0])
            widths = valid_shapes.apply(lambda x: x[1])
            
            print(f"\n  尺寸统计:")
            print(f"    - 高度范围: {heights.min()} - {heights.max()}")
            print(f"    - 宽度范围: {widths.min()} - {widths.max()}")
        else:
             print("  没有有效的尺寸数据。")

    # --- 报告 Check 4: 分布 ---
    print("\n[Check 4: 类别分布检查]")
    
    valid_df = df[df['has_error'] == False]
    if valid_df.empty:
        print("  未能分析分布。")
        return False

    for split in valid_df['split'].unique():
        split_df = valid_df[valid_df['split'] == split]
        total_scenes = len(split_df)
        
        print(f"\n  --- {split.capitalize()} 子集 (共 {total_scenes} 个场景) ---")
        
        # 统计平均水体占比
        avg_water = split_df['water_percent'].mean()
        print(f"    - 平均水体占比: {avg_water:.2%}")
        
        # 找出极端情况
        no_water = (split_df['water_percent'] == 0.0).sum()
        if no_water > 0:
            print(f"    - 完全无水场景数: {no_water}")

    return True 

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
    针对全尺寸大图进行了降采样优化。
    """
    try:
        # 计算降采样步长，防止图片过大
        # 目标是将长边限制在 1000 像素左右用于预览
        with rasterio.open(img_path) as src:
            h, w = src.height, src.width
            step = max(1, max(h, w) // 1000)
        
        with rasterio.open(img_path) as src_img:
            # 使用切片进行降采样读取 [::step, ::step]
            img_vv = src_img.read(1)[::step, ::step]
            img_vh = src_img.read(2)[::step, ::step]
            
        with rasterio.open(mask_path) as src_mask:
            mask_data = src_mask.read(1)[::step, ::step]
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 简单的归一化以便可视化 (截断过亮的值)
        def normalize(arr):
            vmin, vmax = np.percentile(arr, 2), np.percentile(arr, 98)
            return np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)

        axes[0].imshow(normalize(img_vv), cmap='gray')
        axes[0].set_title(f"S1 - VV (1/{step} scale)")
        axes[0].axis('off')
        
        axes[1].imshow(normalize(img_vh), cmap='gray')
        axes[1].set_title(f"S1 - VH (1/{step} scale)")
        axes[1].axis('off')

        axes[2].imshow(mask_data, cmap='jet', vmin=0, vmax=1, interpolation='nearest')
        axes[2].set_title(f"S1 - Mask (1/{step} scale)")
        axes[2].axis('off')
        
        fig.suptitle(f"Scene: {img_path.name}\nOriginal Size: {w}x{h}", fontsize=14)
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
            continue
            
        mask_files = list(mask_dir.glob("*.tif"))
        if not mask_files:
            continue
            
        num_to_sample = min(N_SAMPLES_PER_SPLIT, len(mask_files))
        samples = random.sample(mask_files, num_to_sample)
        
        print(f"  随机抽取 {num_to_sample} 个场景进行可视化...")
        
        for mask_path in samples:
            img_name = mask_path.name
            img_path = img_dir / img_name
            output_png_path = QC_OUTPUT_DIR / f"{split}_{img_name.replace('.tif', '.png')}"
            
            if not img_path.exists():
                print(f"  [警告] 丢失匹配的图像文件: {img_path.name}")
                continue
                
            plot_sample(img_path, mask_path, output_png_path)
    
    print("\n目视检查完成。")

# --- 4. 主执行 ---

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    
    print("========================================")
    print("== 开始执行全尺寸数据集 QC 脚本 ==")
    print("========================================")
    
    main_full_validation()
    
    print("\n----------------------------------------\n")
    
    main_visual_check()
    
    print("\n========================================")
    print("== QC 脚本执行完毕 ==")
    print("========================================")