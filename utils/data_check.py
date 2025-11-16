import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import warnings

# --- 配置 ---
# 您的训练数据 img 文件夹
TRAIN_IMG_DIR = Path("/mnt/data1/rove/dataset/S1_Water/train/img")
# 假设的波段数 (VV, VH)
NUM_CHANNELS = 2

def calculate_stats():
    """
    遍历所有训练图像，计算每个通道的均值和标准差。
    使用 Welford's algorithm 的变体进行稳定计算。
    """
    
    img_files = list(TRAIN_IMG_DIR.glob("*.tif"))
    if not img_files:
        print(f"Error: 在 {TRAIN_IMG_DIR} 中未找到 .tif 文件")
        return

    print(f"开始计算 {len(img_files)} 个训练图块的均值和标准差...")

    # 使用 float64 以保证精度
    channel_sum = np.zeros(NUM_CHANNELS, dtype=np.float64)
    channel_sum_sq = np.zeros(NUM_CHANNELS, dtype=np.float64)
    pixel_count = 0

    # 忽略 rasterio 警告
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    for img_path in tqdm(img_files, desc="Calculating Stats"):
        try:
            with rasterio.open(img_path) as src:
                # 读取图像 (C, H, W)
                img_data = src.read().astype(np.float64)
                
                # 检查通道数
                if img_data.shape[0] != NUM_CHANNELS:
                    print(f"[警告] {img_path.name} 的通道数为 {img_data.shape[0]}，已跳过。")
                    continue
                
                # 累加
                # np.sum 沿着 H 和 W 轴
                channel_sum += np.sum(img_data, axis=(1, 2))
                channel_sum_sq += np.sum(np.square(img_data), axis=(1, 2))
                
                # H * W
                pixel_count += img_data.shape[1] * img_data.shape[2]

        except Exception as e:
            print(f"Error reading {img_path.name}: {e}")

    if pixel_count == 0:
        print("未能处理任何文件。")
        return

    # --- 计算最终的均值和标准差 ---
    
    # 均值
    mean = channel_sum / pixel_count
    
    # 方差: E[X^2] - (E[X])^2
    variance = (channel_sum_sq / pixel_count) - np.square(mean)
    std = np.sqrt(variance)

    print("\n--- 统计计算完成 ---")
    print(f"总像素数: {pixel_count}")
    
    print("\n请将这些值复制到您的 dataprocess.py 脚本中：")
    
    # 打印为 Python 列表格式，方便复制
    print(f"MEAN = {mean.tolist()}")
    print(f"STD = {std.tolist()}")

    print("\n详细值:")
    for i in range(NUM_CHANNELS):
        print(f"  Channel {i} (e.g., {'VV' if i==0 else 'VH'}): Mean={mean[i]:.6f}, Std={std[i]:.6f}")

if __name__ == "__main__":
    # 依赖库: pip install numpy rasterio tqdm
    calculate_stats()