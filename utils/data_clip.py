import json
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# --- 1. 配置路径 ---
SOURCE_DIR = "/mnt/data1/datasets/Sentinel_Water"
OUTPUT_DIR = "/mnt/data1/rove/dataset/S1_Water_512"
PATCH_SIZE = 512

def process_scene(scene_dir, output_base_dir, patch_size=256):
    """
    处理单个S1S2-Water场景。
    读取、切片、过滤并保存 S1 图像和掩膜。
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

        # --- 新增: 确保输出目录存在 ---
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)

        # --- 4. 打开栅格文件并读取数据 ---
        with rasterio.open(img_file) as src_img, \
             rasterio.open(mask_file) as src_mask, \
             rasterio.open(valid_file) as src_valid:

            # 获取图像元数据和 NoData 值
            img_meta = src_img.meta.copy()
            nodata_val = src_img.nodata
            
            mask_meta = src_mask.meta.copy()
            
            # 读取整个数组 (我们将手动切片)
            img_data = src_img.read()
            mask_data = src_mask.read(1)  # 掩膜只有1个波段
            valid_data = src_valid.read(1) # 有效掩膜只有1个波段
            
            # 创建一个合并的 NoData 掩码
            # 如果任一波段 (VV or VH) 是 NoData，则该像素无效
            if nodata_val is not None:
                # np.any(..., axis=0) 会检查所有波段
                combined_nodata_mask = np.any(img_data == nodata_val, axis=0)
            else:
                # 如果没有定义 NoData 值, 我们就假设所有数据都有效
                combined_nodata_mask = np.zeros_like(mask_data, dtype=bool)

            height, width = src_img.height, src_img.width
            patch_count = 0

            # --- 5. 循环切片 ---
            for r_idx, r in enumerate(range(0, height, patch_size)):
                for c_idx, c in enumerate(range(0, width, patch_size)):
                    
                    # 定义窗口
                    window = rasterio.windows.Window(c, r, patch_size, patch_size)
                    
                    # 检查窗口是否超出边界
                    if window.width < patch_size or window.height < patch_size:
                        continue
                    
                    # --- 6. 过滤图块 ---
                    
                    # 提取图块
                    valid_patch = valid_data[window.row_off:window.row_off + window.height, 
                                             window.col_off:window.col_off + window.width]
                    
                    nodata_patch = combined_nodata_mask[window.row_off:window.row_off + window.height, 
                                                        window.col_off:window.col_off + window.width]

                    # 条件 A: 检查 's1_valid_file' (假设 1 为有效)
                    is_quality_valid = np.all(valid_patch == 1)
                    
                    # 条件 B: 检查图像 'NoData'
                    # (np.any(nodata_patch) 会告诉我们是否存在 *任何* NoData 像素)
                    has_nodata = np.any(nodata_patch)
                    
                    if not is_quality_valid or has_nodata:
                        continue  # 丢弃这个图块

                    # --- 7. 保存有效图块 ---
                    
                    # 计算新图块的地理转换信息
                    patch_transform = rasterio.windows.transform(window, src_img.transform)
                    
                    # 保存图像图块
                    img_patch = img_data[:, window.row_off:window.row_off + window.height, 
                                         window.col_off:window.col_off + window.width]
                    
                    img_meta.update({
                        'height': patch_size,
                        'width': patch_size,
                        'transform': patch_transform
                    })
                    
                    filename = f"{scene_id}_{r_idx}_{c_idx}.tif"
                    with rasterio.open(target_img_dir / filename, 'w', **img_meta) as dst:
                        dst.write(img_patch)

                    # 保存掩膜图块
                    mask_patch = mask_data[window.row_off:window.row_off + window.height, 
                                           window.col_off:window.col_off + window.width]
                    
                    mask_meta.update({
                        'height': patch_size,
                        'width': patch_size,
                        'transform': patch_transform
                    })
                    
                    with rasterio.open(target_mask_dir / filename, 'w', **mask_meta) as dst:
                        dst.write(mask_patch, 1)

                    patch_count += 1
            
            return f"Processed {scene_id} ({split}): Saved {patch_count} valid patches."

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

    print(f"Starting preprocessing...")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Patch Size: {PATCH_SIZE}x{PATCH_SIZE}\n")

    # 找到所有场景目录
    scene_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    if not scene_dirs:
        print("Error: 在源目录中未找到任何场景。")
        return

    # 使用 tqdm 创建进度条
    pbar = tqdm(scene_dirs, desc="Processing Scenes")
    for scene_dir in pbar:
        result = process_scene(scene_dir, output_path, PATCH_SIZE)
        pbar.set_postfix_str(result, refresh=True)

    print("\nPreprocessing complete.")

if __name__ == "__main__":
    # --- 8. 安装依赖 ---
    # 在运行此脚本之前，请确保已安装所需库:
    # pip install rasterio numpy tqdm
    
    main()