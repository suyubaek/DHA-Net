import torch
import numpy as np
import rasterio
from tqdm import tqdm
import os
from config import config
from model import Model

# --------------------------------------------------------------------------------
# 用户配置区域 (User Configuration)
# --------------------------------------------------------------------------------
# 1. 模型权重地址
MODEL_WEIGHTS_PATH = "./checkpoints/DHA_Net_best.pth" 

# 2. 推理文件地址 (单文件)
INFERENCE_FILE_PATH = "/path/to/your/research_area_image.tif"

# 3. 预测结果保存地址
RESULT_SAVE_PATH = "./results/research_area_result.tif"

# 4. 数据标准化参数 (需与训练时一致)
# 请替换为您训练集计算得到的均值和标准差
# 这里的默认值仅为示例，请务必修改！
NORM_MEAN = [0.0, 0.0] 
NORM_STD = [1.0, 1.0]

# --------------------------------------------------------------------------------

def get_gaussian_mask(size, sigma_scale=1.0/8):
    """生成高斯权重掩膜，用于平滑拼接"""
    tmp = np.zeros((size, size))
    center = size // 2
    sig = size * sigma_scale
    y, x = np.ogrid[-center:size-center, -center:size-center]
    mask = np.exp(-(x**2 + y**2) / (2 * sig**2))
    return mask

def predict_sliding_window(model, image_path, save_path, patch_size, stride, num_classes, mean, std, device):
    """
    使用高斯加权滑动窗口对大图进行推理
    """
    print(f"Processing: {image_path}")
    
    # 1. 读取图像 (使用 rasterio 保留地理信息)
    with rasterio.open(image_path) as src:
        image = src.read() # (C, H, W)
        profile = src.profile
        
        # 确保数据是 float32 用于计算
        image = image.astype(np.float32)
        c, h, w = image.shape
        
        print(f"Image Shape: {image.shape}")
        
    # 2. 准备高斯掩膜
    gaussian_mask = get_gaussian_mask(patch_size)
    gaussian_mask_tensor = torch.from_numpy(gaussian_mask).float().to(device)
    
    # 3. Padding 处理
    # 使得图像尺寸能被 stride 整除，且最后一块能完整包含 patch_size
    pad_h = stride - (h % stride) if h % stride != 0 else 0
    pad_w = stride - (w % stride) if w % stride != 0 else 0
    
    # 还需要确保 padding 后的尺寸至少能容纳一个 patch
    if h + pad_h < patch_size: pad_h = patch_size - h
    if w + pad_w < patch_size: pad_w = patch_size - w

    # 对 (C, H, W) 进行 padding，只在 H 和 W 方向 pad
    # image 是 (C, H, W)，pad width 格式为 ((before_1, after_1), ... (before_N, after_N))
    image_padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    _, h_pad, w_pad = image_padded.shape
    
    # 4. 初始化结果容器
    # num_classes 为 1 (Binary Segmentation)
    prob_map = torch.zeros((num_classes, h_pad, w_pad), device=device)
    weight_map = torch.zeros((h_pad, w_pad), device=device)
    
    # 准备标准化参数
    mean_tensor = torch.tensor(mean).view(c, 1, 1).to(device)
    std_tensor = torch.tensor(std).view(c, 1, 1).to(device)

    model.eval()
    
    # 5. 滑动窗口推理
    with torch.no_grad():
        # 生成所有窗口坐标
        rows = list(range(0, h_pad - patch_size + 1, stride))
        cols = list(range(0, w_pad - patch_size + 1, stride))
        
        total_windows = len(rows) * len(cols)
        
        with tqdm(total=total_windows, desc="Inference") as pbar:
            for r in rows:
                for c in cols:
                    # 截取 Patch (C, H, W)
                    patch = image_padded[:, r:r+patch_size, c:c+patch_size]
                    
                    # 转 Tensor 并标准化
                    patch_tensor = torch.from_numpy(patch).float().to(device)
                    patch_tensor = (patch_tensor - mean_tensor) / std_tensor
                    
                    # 增加 Batch 维度 (1, C, H, W)
                    patch_tensor = patch_tensor.unsqueeze(0)
                    
                    # 推理
                    output = model(patch_tensor)
                    
                    # Sigmoid 归一化 (Binary Segmentation)
                    output = torch.sigmoid(output) # (1, 1, H, W)
                    pred_patch = output.squeeze(0) # (1, H, W)
                    
                    # 加权累加
                    prob_map[:, r:r+patch_size, c:c+patch_size] += pred_patch * gaussian_mask_tensor
                    weight_map[r:r+patch_size, c:c+patch_size] += gaussian_mask_tensor
                    
                    pbar.update(1)

    # 6. 归一化与裁剪
    # 避免除以 0
    weight_map = torch.where(weight_map == 0, torch.ones_like(weight_map), weight_map)
    prob_map /= weight_map # (1, H_pad, W_pad)
    
    # 裁剪回原始大小
    prob_map = prob_map[:, :h, :w]
    
    # 7. 生成最终结果 (阈值 0.5)
    result_mask = (prob_map > 0.5).float().cpu().numpy().astype(np.uint8)
    # result_mask shape: (1, H, W)
    
    # 8. 保存结果
    # 更新 profile 信息
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(result_mask[0], 1) # 写入第一个波段
        
    print(f"Inference finished. Result saved to: {save_path}")


if __name__ == "__main__":
    # ----------------------------------------------------------------------------
    # 初始化配置
    # ----------------------------------------------------------------------------
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 设置滑动窗口参数
    # "滑动窗口大小（直接用config中的image_size的一半就行）"
    PATCH_SIZE = config['image_size'] // 2
    STRIDE = PATCH_SIZE // 2  # 50% overlap for smooth stitching
    
    print(f"Config Image Size: {config['image_size']}")
    print(f"Inference Patch Size: {PATCH_SIZE}")
    print(f"Inference Stride: {STRIDE}")

    # 2. 加载模型
    print("Loading model...")
    # 注意：config['num_classes'] 为 1，对应 Sigmoid 输出
    model = Model(in_channels=config['in_channels'], num_classes=config['num_classes']).to(device)
    
    # 加载权重
    if os.path.exists(MODEL_WEIGHTS_PATH):
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        # 兼容保存整个 checkpoint 或只保存 state_dict 的情况
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model weights loaded from {MODEL_WEIGHTS_PATH}")
    else:
        print(f"Warning: Model weights not found at {MODEL_WEIGHTS_PATH}. Cannot proceed with inference.")

    # 3. 执行推理
    if os.path.exists(INFERENCE_FILE_PATH) and os.path.exists(MODEL_WEIGHTS_PATH):
        predict_sliding_window(
            model=model,
            image_path=INFERENCE_FILE_PATH,
            save_path=RESULT_SAVE_PATH,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            num_classes=config['num_classes'],
            mean=NORM_MEAN,
            std=NORM_STD,
            device=device
        )
    else:
        if not os.path.exists(INFERENCE_FILE_PATH):
            print(f"Error: Inference file not found at {INFERENCE_FILE_PATH}")
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            print(f"Error: Model weights not found at {MODEL_WEIGHTS_PATH}")
        print("Please configure paths in the script.")