import os
from PIL import Image
import numpy as np
import cv2
import sys

# --- 要检查的文件 ---
# 既然你已经在这个文件夹里了，我们直接用文件名
file_name = "/mnt/sda1/songyufei/asset/lancang/data_managed/image/10-cut-158.tif"
# --------------------

print(f"--- 正在检查文件: {file_name} ---")

# 1. 检查文件是否存在
if not os.path.exists(file_name):
    print(f"错误：在当前目录下找不到文件: {file_name}")
else:
    print("文件存在。正在读取...")

    # --- 方法 1: 使用 Pillow (PIL) 库检查 ---
    print("\n--- 1. 使用 Pillow (PIL) 库检查 ---")
    try:
        # 设置 PIL 以处理大型图像
        Image.MAX_IMAGE_PIXELS = None
        
        img_pil = Image.open(file_name)
        print(f"  - 图像格式: {img_pil.format}")
        print(f"  - 图像尺寸 (宽, 高): {img_pil.size}")
        print(f"  - 图像模式 (Mode): {img_pil.mode}")
        print(f"  - 波段数量 (Bands): {len(img_pil.getbands())}")

        # 转换为 NumPy 数组来获取更详细信息
        np_array_pil = np.array(img_pil)
        print(f"\n  (通过 NumPy 转换)")
        print(f"  - NumPy 数组形状: {np_array_pil.shape}")
        print(f"  - NumPy 数据类型: {np_array_pil.dtype}")
        print(f"  - 像素最小值: {np_array_pil.min()}")
        print(f"  - 像素最大值: {np_array_pil.max()}")

    except Exception as e:
        print(f"  使用 PIL 读取时出错: {e}")

    # --- 方法 2: 使用 OpenCV (cv2) ---
    print("\n--- 2. 使用 OpenCV (cv2) 库检查 ---")
    try:
        # cv2.IMREAD_UNCHANGED 会尝试按原样读取，包括alpha通道和原始位深度
        img_cv = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

        if img_cv is not None:
            print(f"  - NumPy 数组形状: {img_cv.shape}")
            print(f"  - NumPy 数据类型: {img_cv.dtype}")
            print(f"  - 像素最小值: {img_cv.min()}")
            print(f"  - 像素最大值: {img_cv.max()}")

            if len(img_cv.shape) == 2:
                print("  - 解释: 这是一个单通道 (灰度) 图像。")
            elif len(img_cv.shape) == 3:
                print(f"  - 解释: 这是一个多通道图像，有 {img_cv.shape[2]} 个波段。")
        else:
            print("  OpenCV 无法读取此图像 (返回 None)。")

    except Exception as e:
        print(f"  使用 OpenCV 读取时出错: {e}")

print("\n--- 检查完毕 ---")