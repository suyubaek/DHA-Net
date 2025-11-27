import os
import shutil
import random
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# --- 配置路径 ---
DATA_ROOT = Path("/mnt/data1/rove/dataset/S1_Water_512")
TEST_IMG_DIR = DATA_ROOT / "test" / "img"
TEST_MASK_DIR = DATA_ROOT / "test" / "mask"
VIS_DIR = DATA_ROOT / "vis"

# --- 配置采样参数 ---
TOTAL_SAMPLES = 12
# 修改后的分桶策略
BINS = [
    (0.0, 0.01, "纯陆地/极少水"),   # 0% - 1% (测试误报)
    (0.01, 0.20, "细小水体"),       # 1% - 20% (测试细小目标召回)
    (0.20, 0.60, "典型混合区"),     # 20% - 60% (测试复杂边界)
    (0.60, 0.999, "大面积水体(非纯水)") # 60% - 99.9% (排除 100% 纯水)
]
SAMPLES_PER_BIN = TOTAL_SAMPLES // len(BINS)  # 每个桶采 3 张

def calculate_water_ratio(mask_path):
    """读取掩膜并计算水体像素占比"""
    try:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            water_pixels = np.count_nonzero(mask > 0)
            total_pixels = mask.size
            return water_pixels / total_pixels
    except Exception as e:
        print(f"读取错误 {mask_path}: {e}")
        return -1.0

def main():
    # 1. 准备目录
    vis_img_dir = VIS_DIR / "img"
    vis_mask_dir = VIS_DIR / "mask"
    
    if VIS_DIR.exists():
        print(f"警告: {VIS_DIR} 已存在，正在清空...")
        shutil.rmtree(VIS_DIR)
    
    vis_img_dir.mkdir(parents=True, exist_ok=True)
    vis_mask_dir.mkdir(parents=True, exist_ok=True)

    # 2. 扫描所有测试集掩膜
    mask_files = list(TEST_MASK_DIR.glob("*.tif"))
    if not mask_files:
        print("错误: 未找到测试集掩膜文件！")
        return

    print(f"正在扫描 {len(mask_files)} 个测试样本的水体占比...")
    
    binned_files = {i: [] for i in range(len(BINS))}
    ignored_pure_water = 0
    
    for mask_file in tqdm(mask_files):
        ratio = calculate_water_ratio(mask_file)
        if ratio < 0: continue
        
        # 特殊处理：统计被忽略的纯水样本
        if ratio > 0.999:
            ignored_pure_water += 1
            continue

        # 分桶
        for i, (low, high, name) in enumerate(BINS):
            if low <= ratio < high:
                binned_files[i].append((mask_file, ratio))
                break

    print(f"已忽略 {ignored_pure_water} 个纯水(100%)样本，以聚焦边界性能。")

    # 3. 执行采样并复制
    print("\n--- 开始采样 ---")
    selected_files = []

    for i, (low, high, name) in enumerate(BINS):
        candidates = binned_files[i]
        count = len(candidates)
        print(f"桶 [{name}]: 共有 {count} 个样本")
        
        if count == 0:
            print(f"  [警告] 该桶没有样本！")
            continue

        if count < SAMPLES_PER_BIN:
            print(f"  [提示] 样本不足 {SAMPLES_PER_BIN} 个，全部选取。")
            chosen = candidates
        else:
            chosen = random.sample(candidates, SAMPLES_PER_BIN)
            
        for mask_path, ratio in chosen:
            img_path = TEST_IMG_DIR / mask_path.name
            
            if not img_path.exists():
                print(f"  [错误] 对应的图像文件不存在: {img_path}")
                continue
                
            shutil.copy2(img_path, vis_img_dir / mask_path.name)
            shutil.copy2(mask_path, vis_mask_dir / mask_path.name)
            
            print(f"  已复制: {mask_path.name} (占比: {ratio:.2%})")
            selected_files.append(mask_path.name)

    print(f"\n完成！共选取 {len(selected_files)} 个样本存放在 {VIS_DIR}")

if __name__ == "__main__":
    main()