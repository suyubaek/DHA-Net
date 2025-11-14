import os
import shutil
import numpy as np
from PIL import Image
import math
from tqdm import tqdm

# --- 1. 定义你的路径 ---

# 你的(base) songyufei@...:~/...$ 提示符告诉我，
# 你可能在 ~ (home) 目录。
# 使用绝对路径是最安全、最不会出错的。

# 你的根目录 (lancang)
base_dir = "/mnt/sda1/songyufei/asset/lancang"

# 输入目录
input_dir = os.path.join(base_dir, "data_managed")
image_src_dir = os.path.join(input_dir, "image")
mask_src_dir = os.path.join(input_dir, "mask")

# 输出目录 (我们新建一个 "data_bucketed" 文件夹来存放)
output_dir = os.path.join(base_dir, "data_bucketed")

# --- 2. 创建所有的目标文件夹 (0 到 11) ---
print(f"正在创建 12 个分桶目录于: {output_dir}")
for i in range(12):
    bucket_str = str(i)
    os.makedirs(os.path.join(output_dir, bucket_str, "image"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, bucket_str, "mask"), exist_ok=True)

print("目录创建完毕。")

# --- 3. 核心逻辑：遍历、计算、拷贝 ---

# 我们遍历 mask 文件夹，因为它是分桶的依据
mask_filenames = os.listdir(mask_src_dir)

print(f"开始处理 {len(mask_filenames)} 个掩码文件...")

# 使用 tqdm 来显示进度条
for filename in tqdm(mask_filenames, desc="Processing files"):
    
    # --- 构建路径 ---
    mask_path = os.path.join(mask_src_dir, filename)
    image_path = os.path.join(image_src_dir, filename) # 假设 image 和 mask 同名

    # --- 健壮性检查 ---
    if not os.path.exists(image_path):
        print(f"警告: 找不到对应的影像 {filename}，跳过此文件。")
        continue

    try:
        # --- 计算水体含量 ---
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)

        # 鲁棒性检查: 
        # 你说 mask 是 0/1，但以防万一它是 0/255，我们做个归一化
        if np.max(mask_array) > 1:
            mask_array = mask_array / 255.0
        
        # 计算水体像素的百分比 (均值)
        water_content = np.mean(mask_array)

        # --- 确定桶号 (Bucket) ---
        bucket_num = -1 # 初始值

        if water_content == 0.0:
            bucket_num = 0  # 全背景
        elif water_content == 1.0:
            bucket_num = 11 # 全水体
        elif water_content > 0.9: # 对应你的 (0.9, 1.0) 
            bucket_num = 10 
        else: 
            # 对应 (0, 0.9]
            # math.ceil(0.05 * 10) = 1  (桶 1: (0, 0.1])
            # math.ceil(0.1 * 10)  = 1  (桶 1: (0, 0.1])
            # math.ceil(0.85 * 10) = 9  (桶 9: (0.8, 0.9])
            # math.ceil(0.9 * 10)  = 9  (桶 9: (0.8, 0.9])
            # 这个逻辑完美符合你的 1-9 桶
            bucket_num = int(math.ceil(water_content * 10))

        # --- 拷贝文件 ---
        bucket_str = str(bucket_num)
        
        # 目标路径
        dest_image_path = os.path.join(output_dir, bucket_str, "image", filename)
        dest_mask_path = os.path.join(output_dir, bucket_str, "mask", filename)

        # 使用 shutil.copy 来拷贝
        shutil.copy(image_path, dest_image_path)
        shutil.copy(mask_path, dest_mask_path)

    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}，跳过。")

print("--- 所有文件处理完毕！ ---")
print(f"数据已按水体含量分桶，存放在: {output_dir}")