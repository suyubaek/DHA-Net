import os
import shutil
import random
from tqdm import tqdm

# --- 1. 配置参数 ---

# (保持和你之前的脚本一致)
base_dir = "/mnt/sda1/songyufei/asset/lancang"
bucketed_dir = os.path.join(base_dir, "data_bucketed")
final_dir = "/mnt/sda1/songyufei/dataset/lancang"

# 你的策略：从桶0 (2000张) 中随机保留 400 张
BUCKET_0_KEEP_COUNT = 400
# 训练集和验证集的划分比例 (80% 训练, 20% 验证)
TRAIN_RATIO = 0.8

random.seed(42) # 固定随机种子，确保每次运行结果一致

# --- 2. 创建目录结构 ---
print(f"正在创建最终数据集目录: {final_dir}")
os.makedirs(os.path.join(final_dir, "train", "image"), exist_ok=True)
os.makedirs(os.path.join(final_dir, "train", "mask"), exist_ok=True)
os.makedirs(os.path.join(final_dir, "val", "image"), exist_ok=True)
os.makedirs(os.path.join(final_dir, "val", "mask"), exist_ok=True)

# --- 3. 定义一个辅助函数来拷贝文件 ---
#     (src_file_list 是一个元组列表: [(filename, bucket_str), ...])
#     (split_name 是 "train" 或 "val")
def copy_file_pairs(src_file_list, split_name):
    print(f"正在拷贝 {len(src_file_list)} 个文件到 {split_name} 集...")
    
    for filename, bucket_num_str in tqdm(src_file_list, desc=f"Copying to {split_name}"):
        # 1. 定义源路径
        src_img_path = os.path.join(bucketed_dir, bucket_num_str, "image", filename)
        src_mask_path = os.path.join(bucketed_dir, bucket_num_str, "mask", filename)
        
        # 2. 定义目标路径
        dest_img_path = os.path.join(final_dir, split_name, "image", filename)
        dest_mask_path = os.path.join(final_dir, split_name, "mask", filename)
        
        # 3. 拷贝
        shutil.copy(src_img_path, dest_img_path)
        shutil.copy(src_mask_path, dest_mask_path)

# --- 4. 核心逻辑：处理 "有水" 的桶 (1 到 11) ---
print("开始处理 '有水' 的桶 (1-11)...")
water_files = [] # 存储所有 (filename, bucket_str)
for i in range(1, 12): # 遍历 1, 2, 3, ..., 11
    bucket_str = str(i)
    mask_dir = os.path.join(bucketed_dir, bucket_str, "mask")
    
    if os.path.isdir(mask_dir):
        filenames = os.listdir(mask_dir)
        for fname in filenames:
            water_files.append((fname, bucket_str))

print(f"共找到 {len(water_files)} 个 '有水' 样本 (桶 1-11)。") # 应该等于 2338

# 随机打乱并划分
random.shuffle(water_files)
split_idx = int(len(water_files) * TRAIN_RATIO)
train_water_list = water_files[:split_idx]
val_water_list = water_files[split_idx:]

print(f" '有水' 样本: {len(train_water_list)} 训练, {len(val_water_list)} 验证。")

# --- 5. 核心逻辑：处理 "全背景" 的桶 (0) ---
print(f"开始处理 '全背景' 的桶 (0)，将随机抽取 {BUCKET_0_KEEP_COUNT} 个...")
bucket_str = "0"
mask_dir = os.path.join(bucketed_dir, bucket_str, "mask")
all_bg_files = os.listdir(mask_dir) # 2000 个

# 随机降采样
selected_bg_files = random.sample(all_bg_files, BUCKET_0_KEEP_COUNT)

# 将文件名打包成 (filename, bucket_str) 格式
bg_files_with_bucket = [(fname, bucket_str) for fname in selected_bg_files]

# 随机打乱并划分
random.shuffle(bg_files_with_bucket)
split_idx_bg = int(len(bg_files_with_bucket) * TRAIN_RATIO)
train_bg_list = bg_files_with_bucket[:split_idx_bg]
val_bg_list = bg_files_with_bucket[split_idx_bg:]

print(f" '背景' 样本: {len(train_bg_list)} 训练, {len(val_bg_list)} 验证。")

# --- 6. 执行拷贝 ---
copy_file_pairs(train_water_list, "train")
copy_file_pairs(val_water_list, "val")
copy_file_pairs(train_bg_list, "train")
copy_file_pairs(val_bg_list, "val")

# --- 7. 最终总结 ---
total_train = len(train_water_list) + len(train_bg_list)
total_val = len(val_water_list) + len(val_bg_list)
total_all = total_train + total_val

print("\n--- 数据集构建完毕！---")
print(f"最终数据集位置: {final_dir}")
print("="*30)
print(f"训练集 (Train): {total_train} 张图片")
print(f"验证集 (Val)  : {total_val} 张图片")
print(f"总计          : {total_all} 张图片")
print("="*30)
print("你现在可以直接使用 final_dataset 目录进行训练了。")