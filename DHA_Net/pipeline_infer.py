import torch
import numpy as np
import rasterio
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from config import config
from model import Model

# 推理参数设置
MODEL_WEIGHTS_PATH = "checkpoints/DHA_Net_1125/best_model.pth" 
INFERENCE_FILE_PATH = "/mnt/data1/rove/dataset/S1_Water/infer/2412_rec_vv_vh.tif"
RESULT_SAVE_PATH = "./results/research_area_result.tif"
NORM_MEAN = [-1148.2476, -1944.6511]
NORM_STD  = [594.0617, 973.9897]
INFER_BATCH_SIZE = 32
INFER_NUM_WORKERS = 8

class SlidingWindowDataset(Dataset):
    def __init__(self, image_padded, rows, cols, patch_size, mean, std):
        self.image = image_padded
        self.rows = rows
        self.cols = cols
        self.patch_size = patch_size
        
        self.mean = torch.tensor(mean).view(-1, 1, 1).float()
        self.std = torch.tensor(std).view(-1, 1, 1).float()

    def __len__(self):
        return len(self.rows) * len(self.cols)

    def __getitem__(self, idx):
        # 计算当前 idx 对应的行列索引
        r_idx = idx // len(self.cols)
        c_idx = idx % len(self.cols)
        
        r = self.rows[r_idx]
        c = self.cols[c_idx]
        
        # 切片 (C, H, W)
        patch = self.image[:, r:r+self.patch_size, c:c+self.patch_size]
        
        # 转 Tensor 并标准化
        patch_tensor = torch.from_numpy(patch).float()
        patch_tensor = (patch_tensor - self.mean) / self.std
        
        # 返回 patch 和 它的坐标，以便拼回去
        return patch_tensor, r, c

def get_gaussian_mask(size, sigma_scale=1.0/8):
    """生成高斯权重掩膜"""
    tmp = np.zeros((size, size))
    center = size // 2
    sig = size * sigma_scale
    y, x = np.ogrid[-center:size-center, -center:size-center]
    mask = np.exp(-(x**2 + y**2) / (2 * sig**2))
    return mask

def predict_sliding_window(model, image_path, save_path, patch_size, stride, num_classes, mean, std, device):
    print(f"\n{'='*20} 开始处理 {'='*20}")
    print(f"目标文件: {image_path}")
    
    # 1. 读取图像
    print(f"[1/6] 正在读取图像数据...")
    with rasterio.open(image_path) as src:
        image = src.read() # (C, H, W)
        profile = src.profile
        image = image.astype(np.float32)
        c, h, w = image.shape
        print(f"      -> 原始尺寸: {c}x{h}x{w}")
        
    # 2. Padding
    print(f"[2/6] 正在进行边缘填充 (Padding)...")
    pad_h = stride - (h % stride) if h % stride != 0 else 0
    pad_w = stride - (w % stride) if w % stride != 0 else 0
    if h + pad_h < patch_size: pad_h = patch_size - h
    if w + pad_w < patch_size: pad_w = patch_size - w

    image_padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    _, h_pad, w_pad = image_padded.shape
    print(f"      -> 填充后尺寸: {image_padded.shape}")
    
    # 3. 准备坐标列表
    print(f"[3/6] 计算滑动窗口坐标...")
    rows = list(range(0, h_pad - patch_size + 1, stride))
    cols = list(range(0, w_pad - patch_size + 1, stride))
    total_patches = len(rows) * len(cols)
    print(f"      -> 总计切片数量: {total_patches}")
    
    # 4. 构建 Dataset 和 DataLoader
    print(f"[4/6] 初始化 DataLoader (Workers: {INFER_NUM_WORKERS})...")
    dataset = SlidingWindowDataset(image_padded, rows, cols, patch_size, mean, std)
    loader = DataLoader(
        dataset, 
        batch_size=INFER_BATCH_SIZE, 
        shuffle=False, 
        num_workers=INFER_NUM_WORKERS, 
        pin_memory=True
    )
    
    # 5. 准备结果容器 (修改点：放在 CPU 上)
    print(f"[5/6] 在 CPU RAM 上分配结果容器 (节省显存)...")
    # 注意：这里 device='cpu'
    prob_map = torch.zeros((num_classes, h_pad, w_pad), dtype=torch.float32, device='cpu')
    weight_map = torch.zeros((h_pad, w_pad), dtype=torch.float32, device='cpu')
    
    # 高斯掩膜还是放在 GPU 上，用于加速计算
    gaussian_mask = get_gaussian_mask(patch_size)
    gaussian_mask_tensor = torch.from_numpy(gaussian_mask).float().to(device)
    
    model.eval()
    
    # 6. 批量推理
    print(f"[6/6] 开始批量推理 (Batch Size: {INFER_BATCH_SIZE})...")
    with torch.no_grad():
        for batch_patches, batch_r, batch_c in tqdm(loader, desc="推理进度", unit="batch", leave=True):
            # batch_patches: (B, C, H, W) -> GPU
            batch_patches = batch_patches.to(device)
            
            # GPU 推理
            outputs = model(batch_patches)
            outputs = torch.sigmoid(outputs) # (B, 1, H, W)
            
            # 在 GPU 上先乘好高斯权重 (利用 GPU 并行计算优势)
            # 此时 outputs 还在 GPU 上
            outputs = outputs * gaussian_mask_tensor # 广播乘法
            
            # 将计算好的加权结果移回 CPU 进行拼图
            outputs_cpu = outputs.cpu() 
            
            for i in range(outputs_cpu.shape[0]):
                r = batch_r[i].item()
                c = batch_c[i].item()
                
                # 取出单个 patch (已经在 CPU 上了)
                weighted_patch = outputs_cpu[i].squeeze(0) 
                
                # 累加到 CPU 大图上
                prob_map[:, r:r+patch_size, c:c+patch_size] += weighted_patch
                
                # 权重图累加 (gaussian_mask_tensor 也要转回 CPU)
                # 为了效率，可以在循环外生成一个 cpu 版的 mask，或者这里转一次
                weight_map[r:r+patch_size, c:c+patch_size] += gaussian_mask_tensor.cpu()

    # 7. 归一化与保存 (全部在 CPU 上进行)
    print(f"\n正在保存结果到磁盘...")
    
    # 避免除以 0
    weight_map = torch.where(weight_map == 0, torch.ones_like(weight_map), weight_map)
    prob_map /= weight_map
    
    # 裁剪掉 Padding
    prob_map = prob_map[:, :h, :w]
    
    # 生成掩膜 (修改为 0/255)
    result_mask = ((prob_map > 0.5).float().numpy() * 255).astype(np.uint8)
    
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(result_mask[0], 1)
        
    print(f"完成! 结果已保存至: {save_path}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    PATCH_SIZE = config['image_size'] // 2
    STRIDE = PATCH_SIZE // 2
    
    print(f"Config Image Size: {config['image_size']}")
    print(f"Inference Patch Size: {PATCH_SIZE}")
    print(f"Inference Stride: {STRIDE}")

    model = Model(in_channels=config['in_channels'], num_classes=config['num_classes']).to(device)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model weights loaded from {MODEL_WEIGHTS_PATH}")
    else:
        print(f"Warning: Model weights not found.")

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