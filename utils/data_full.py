import json
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# --- 1. 配置路径 ---
SOURCE_DIR = "/mnt/data1/datasets/Sentinel_Water"
OUTPUT_DIR = "/mnt/data1/rove/dataset/S1_Water_Full" # 建议修改输出目录，避免与分块数据混淆
# PATCH_SIZE = 256 # 不再需要分块大小

def process_scene(scene_dir, output_base_dir):
    """
    处理单个S1S2-Water场景。
    读取全图，利用 valid 文件过滤掩膜（置为背景），并保存全尺寸图像和掩膜。
    """
    scene_id = scene_dir.name
    
    try:
        # --- 2. 找到所有必需的文件 ---
        meta_file = scene_dir / f"sentinel12_{scene_id}_meta.json"
        img_file = scene_dir / f"sentinel12_s1_{scene_id}_img.tif"
        mask_file = scene_dir / f"sentinel12_s1_{scene_id}_msk.tif"
        valid_file = scene_dir / f"sentinel12_s1_{scene_id}_valid.tif"

        # 检查文件是否存在
        for f in [meta_file, img_file, mask_file, valid_file]:
            if not f.exists():
                return f"Skipped: Missing file {f.name} in {scene_id}"

        # --- 3. 读取元数据并确定输出路径 ---
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        split = metadata['properties']['split']  # 'train', 'val', or 'test'
        
        target_img_dir = output_base_dir / split / "img"
        target_mask_dir = output_base_dir / split / "mask"
        
        # 确保输出目录存在
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)

        # --- 4. 打开栅格文件并读取数据 ---
        with rasterio.open(img_file) as src_img, \
             rasterio.open(mask_file) as src_mask, \
             rasterio.open(valid_file) as src_valid:

            # 获取图像元数据
            img_meta = src_img.meta.copy()
            mask_meta = src_mask.meta.copy()
            
            # 读取整个数组
            img_data = src_img.read()
            mask_data = src_mask.read(1)  # 掩膜只有1个波段
            valid_data = src_valid.read(1) # 有效掩膜只有1个波段
            
            # --- 5. 处理无效像素 ---
            # valid_file中表示不可用的像素 (非1)，直接归为背景 (0)
            # 假设 valid_data == 1 表示有效
            invalid_pixels = (valid_data != 1)
            mask_data[invalid_pixels] = 0
            
            # 如果图像本身有 NoData，也将其在掩膜中设为背景 (可选，增强鲁棒性)
            if src_img.nodata is not None:
                img_nodata_mask = np.any(img_data == src_img.nodata, axis=0)
                mask_data[img_nodata_mask] = 0

            # --- 6. 保存全尺寸文件 ---
            filename = f"{scene_id}.tif"
            
            # 保存图像 (保持原始数据)
            with rasterio.open(target_img_dir / filename, 'w', **img_meta) as dst:
                dst.write(img_data)

            # 保存掩膜 (已处理无效区域)
            with rasterio.open(target_mask_dir / filename, 'w', **mask_meta) as dst:
                dst.write(mask_data, 1)
            
            return f"Processed {scene_id} ({split}): Saved full scene."

    except Exception as e:
        return f"ERROR processing {scene_id}: {e}"

def main():
    """
    主函数，遍历所有场景并调用处理函数。
    """
    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    if not source_path.is_dir():
        print(f"Error: 源目录不存在: {source_path}")
        return

    # 忽略 rasterio 在处理某些 TIF 时可能发出的无害警告
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    print(f"Starting preprocessing (Full Scene Mode)...")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    # print(f"Patch Size: {PATCH_SIZE}x{PATCH_SIZE}\n")

    # 找到所有场景目录
    scene_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    if not scene_dirs:
        print("Error: 在源目录中未找到任何场景。")
        return

    # 使用 tqdm 创建进度条
    pbar = tqdm(scene_dirs, desc="Processing Scenes")
    for scene_dir in pbar:
        result = process_scene(scene_dir, output_path)
        pbar.set_postfix_str(result, refresh=True)

    print("\nPreprocessing complete.")

if __name__ == "__main__":
    # --- 8. 安装依赖 ---
    # 在运行此脚本之前，请确保已安装所需库:
    # pip install rasterio numpy tqdm
    
    main()