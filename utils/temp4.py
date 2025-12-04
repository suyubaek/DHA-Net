import rasterio
import numpy as np
import os

def check_tif_info(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> {file_path}")
        return

    print(f"正在检查文件: {file_path}")
    print("=" * 60)

    try:
        with rasterio.open(file_path) as src:
            # 1. 基础信息
            print(f"图像大小 (Height x Width): {src.height} x {src.width}")
            print(f"波段数量 (Count): {src.count}")
            print(f"数据类型 (Dtype): {src.dtypes}")
            print(f"坐标系 (CRS): {src.crs}")
            print("-" * 60)

            total_pixels = src.height * src.width

            # 2. 逐波段统计
            for i in range(1, src.count + 1):
                print(f"正在读取波段 {i} ...")
                band_data = src.read(i)
                
                # 检查 NaN (非数值)
                nan_count = np.isnan(band_data).sum()
                nan_ratio = (nan_count / total_pixels) * 100
                has_nan = nan_count > 0
                
                # 检查 Inf (无穷大)
                inf_count = np.isinf(band_data).sum()
                inf_ratio = (inf_count / total_pixels) * 100
                
                # 计算最大最小值 (使用 nanmin/nanmax 忽略 NaN 带来的影响)
                if band_data.size == nan_count:
                    b_min = "N/A (All NaN)"
                    b_max = "N/A (All NaN)"
                    b_mean = "N/A"
                else:
                    b_min = np.nanmin(band_data)
                    b_max = np.nanmax(band_data)
                    b_mean = np.nanmean(band_data)

                print(f"【波段 {i} 统计结果】:")
                print(f"  Min 值: {b_min}")
                print(f"  Max 值: {b_max}")
                print(f"  Mean 值: {b_mean:.4f}")
                
                # 打印 NaN 统计
                nan_status = "⚠️ 存在 NaN" if has_nan else "正常"
                print(f"  NaN 统计: {nan_count} 像素 (占比 {nan_ratio:.4f}%) \t[{nan_status}]")
                
                # 打印 Inf 统计
                if inf_count > 0:
                    print(f"  Inf 统计: {inf_count} 像素 (占比 {inf_ratio:.4f}%) \t[⚠️ 存在 Inf]")
                else:
                    print(f"  Inf 统计: 0 像素 (0.00%) \t[正常]")
                
                # 额外检查：是否全为 0
                if b_min == 0 and b_max == 0:
                    print(f"  ⚠️ 警告: 该波段全为 0 (可能是空数据)")
                
                print("-" * 60)

    except Exception as e:
        print(f"读取文件时发生错误: {e}")

if __name__ == "__main__":
    # 你指定的文件路径
    target_file = "/mnt/data1/datasets/Sentinel_Water/1/sentinel12_s1_1_img.tif"
    # target_file = "/mnt/data1/rove/dataset/S1_Water/infer/2412_rec_vv_vh.tif"
    check_tif_info(target_file)