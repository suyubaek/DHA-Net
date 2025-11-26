import rasterio
from rasterio.windows import Window
import numpy as np
import os

def analyze_file(filepath, crop_size=50000):
    print(f"\n{'='*20} 分析文件: {os.path.basename(filepath)} (Top-Left {crop_size}x{crop_size}) {'='*20}")
    
    try:
        with rasterio.open(filepath) as src:
            # 1. 基本形状信息
            print(f"原始形状 (Height, Width): {src.shape}")
            print(f"波段数 (Count): {src.count}")
            print(f"数据类型 (Dtype): {src.dtypes}")
            print(f"坐标系 (CRS): {src.crs}")
            
            # 定义读取窗口 (左上角)
            # 确保窗口不超过图像实际大小
            read_h = min(crop_size, src.height)
            read_w = min(crop_size, src.width)
            window = Window(col_off=0, row_off=0, width=read_w, height=read_h)
            
            print(f"实际读取区域: {read_w} x {read_h}")

            # 只读取窗口内的数据
            data = src.read(window=window)
            
            # 2. 数据分布情况
            print(f"\n--- 数据分布统计 (局部采样) ---")
            for i in range(src.count):
                band_data = data[i]
                # 过滤掉无效值 (如果有 nodata 定义)
                if src.nodata is not None:
                    valid_mask = band_data != src.nodata
                    valid_data = band_data[valid_mask]
                else:
                    valid_data = band_data.flatten()
                
                print(f"Band {i+1}:")
                if len(valid_data) == 0:
                    print("  [警告] 该区域无有效数据")
                    continue

                print(f"  Min: {np.min(valid_data)}")
                print(f"  Max: {np.max(valid_data)}")
                print(f"  Mean: {np.mean(valid_data):.4f}")
                print(f"  Std: {np.std(valid_data):.4f}")
                
                # 简单的直方图概览
                percentiles = np.percentile(valid_data, [1, 5, 50, 95, 99])
                print(f"  分位数 [1%, 5%, 50%, 95%, 99%]: {percentiles}")
                
                # 检查是否为二值掩膜 (针对 watermask)
                # 为了性能，只在唯一值很少时打印
                if len(valid_data) < 100000: # 只有数据量很小时才做全量unique，否则太慢
                     unique_vals = np.unique(valid_data)
                     if len(unique_vals) < 10:
                         print(f"  包含的唯一值: {unique_vals}")
                else:
                    # 采样检查唯一值
                    sample = np.random.choice(valid_data, size=min(10000, len(valid_data)), replace=False)
                    unique_vals = np.unique(sample)
                    if len(unique_vals) < 10:
                         print(f"  (采样) 包含的唯一值: {unique_vals}")

    except Exception as e:
        print(f"读取错误: {e}")

if __name__ == "__main__":
    base_dir = "/mnt/data1/rove/dataset/S1_Water/infer"
    files = ["2412vv_vh.tif"]
    
    for f in files:
        path = os.path.join(base_dir, f)
        if os.path.exists(path):
            analyze_file(path, crop_size=50000)
        else:
            print(f"文件不存在: {path}")