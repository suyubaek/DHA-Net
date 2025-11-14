import os

# --- 1. 定义你的路径 ---
# 你的分桶数据的主目录
base_output_dir = "/mnt/sda1/songyufei/asset/lancang/data_bucketed"

total_files = 0
bucket_counts = {} # 使用字典来存储每个桶的计数

print(f"--- 正在统计分桶数据: {base_output_dir} ---")

# --- 2. 遍历 12 个桶 (0 到 11) ---
for i in range(12):
    bucket_str = str(i)
    
    # 我们选择统计 mask 文件夹，因为它和 image 是一一对应的
    bucket_mask_dir = os.path.join(base_output_dir, bucket_str, "mask")
    
    count = 0 # 默认为0
    
    if os.path.isdir(bucket_mask_dir):
        # os.listdir() 会列出所有文件名，len() 就是文件数量
        files_in_bucket = os.listdir(bucket_mask_dir)
        count = len(files_in_bucket)
    else:
        print(f"警告: 找不到目录 {bucket_mask_dir}，计为 0")
        
    bucket_counts[i] = count
    total_files += count

print("\n--- 统计完毕 ---")

# --- 3. 打印结果 ---
print(f"总文件数: {total_files} (这应该和 4338 匹配)\n")
print("各分桶统计详情:")
print("=" * 30)
print("桶号 | 文件数 | 占比 (%)")
print("---- | ------ | ---------")

if total_files == 0:
    print("错误：总文件数为 0，无法计算百分比。")
else:
    # 格式化输出
    for i in range(12):
        count = bucket_counts[i]
        percentage = (count / total_files) * 100.0
        
        # {i:>2}  让桶号右对齐，占2个字符位
        # {count:>6} 让文件数右对齐，占6个字符位
        # {percentage:>7.2f} 让百分比右对齐，占7个字符位，保留2位小数
        print(f" {i:>2} | {count:>6} | {percentage:>7.2f} %")

print("=" * 30)

print("\n桶定义 (供参考):")
print(" 0: 全背景 (0%)")
print(" 1: (0%, 10%]")
print(" 2: (10%, 20%]")
print(" ...")
print(" 10: (90%, 100%)")
print(" 11: 全水体 (100%)")