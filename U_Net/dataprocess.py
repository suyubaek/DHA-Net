import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from pathlib import Path
import warnings
import torchvision.transforms.functional as TF
import random
from tqdm import tqdm

# --- 1. S1Water Dataset 类 (核心逻辑) ---

class S1WaterDataset(Dataset):
    """
    一个“智能”的 Dataset 类，它会自动处理负采样、
    统计计算和结果缓存。
    """
    
    def __init__(self, data_dir, split, 
                 neg_sample_ratio=1.0, 
                 seed=42, 
                 override_stats=None):
        """
        Args:
            data_dir (Path or str): 数据集根目录 (S1_Water)
            split (str): "train", "val", 或 "test"
            neg_sample_ratio (float): (仅用于 'train') 保留的纯陆地样本比例。
            seed (int): (仅用于 'train') 随机采样的种子。
            override_stats (tuple): (仅用于 'val'/'test') 
                                    一个 (mean, std) 元组，用于标准化。
        """
        self.split = split
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / split / "img"
        self.mask_dir = self.data_dir / split / "mask"
        
        # 预先定义波段数
        self.num_channels = 2 # (VV, VH)
        
        # 忽略 rasterio 警告
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

        # --- 核心逻辑分支 ---
        
        if split == 'train':
            # 1. 为训练集：执行采样和统计计算
            
            # 创建一个基于 ratio 和 seed 的唯一缓存文件名
            cache_filename = f"train_cache_ratio_{neg_sample_ratio}_seed_{seed}.pt"
            cache_file = self.data_dir / cache_filename
            
            if cache_file.exists():
                # 2. 如果缓存存在，则加载
                print(f"--- 正在从缓存加载训练数据: {cache_filename} ---")
                cache_data = torch.load(cache_file)
                self.file_list = cache_data['file_list']
                self.mean = cache_data['mean']
                self.std = cache_data['std']
                
            else:
                # 3. 如果缓存不存在，则计算
                print(f"--- 未找到缓存，正在为 {split} split 创建新数据... ---")
                print(f"--- (这在第一次运行时会比较慢) ---")
                
                # 3a. 执行负采样
                self.file_list = self._get_sampled_file_list(
                    self.mask_dir, neg_sample_ratio, seed
                )
                
                # 3b. 计算统计数据
                self.mean, self.std = self._calculate_stats(
                    self.file_list, self.img_dir
                )
                
                # 3c. 保存到缓存
                print(f"--- 计算完成，保存到缓存: {cache_file} ---")
                torch.save({
                    'file_list': self.file_list,
                    'mean': self.mean,
                    'std': self.std
                }, cache_file)

        else:
            # 4. 为 Val/Test 集：使用完整数据和传入的统计数据
            print(f"--- 正在为 {split} split 加载完整数据... ---")
            if override_stats is None:
                raise ValueError(f"{split} split 必须提供 'override_stats' (来自训练集)")
            
            self.file_list = sorted(list(self.mask_dir.glob("*.tif")))
            self.mean, self.std = override_stats # 使用传入的 train stats

        # --- 收尾工作 ---
        if not self.file_list:
            raise FileNotFoundError(f"在 {self.mask_dir} 中未找到 .tif 文件。")
            
        # 将均值和标准差调整为 (C, 1, 1) 以便进行广播
        self.mean = self.mean.view(self.num_channels, 1, 1)
        self.std = self.std.view(self.num_channels, 1, 1)

    def _get_sampled_file_list(self, mask_dir, ratio, seed):
        """
        [内部方法] 扫描所有掩膜，将文件分为正/负，然后对负样本进行降采样。
        """
        print(f"正在扫描 {mask_dir} 以构建文件列表...")
        positive_files = [] # 包含任何水体 (water_percent > 0)
        negative_files = [] # 纯陆地 (water_percent == 0)
        
        all_mask_files = list(mask_dir.glob("*.tif"))
        if not all_mask_files:
            raise FileNotFoundError(f"在 {mask_dir} 中未找到 .tif 文件。")

        for mask_path in tqdm(all_mask_files, desc="扫描掩膜分布"):
            try:
                with rasterio.open(mask_path) as src:
                    if src.read(1, window=rasterio.windows.Window(0, 0, 1, 1)).size == 0:
                        continue
                    if np.any(src.read(1)): 
                        positive_files.append(mask_path)
                    else:
                        negative_files.append(mask_path)
            except Exception as e:
                print(f"  [警告] 读取 {mask_path.name} 失败: {e}")

        print(f"扫描完成：找到 {len(positive_files)} 个正/混合样本，{len(negative_files)} 个负样本（纯陆地）。")

        if ratio >= 1.0:
            print("NEG_SAMPLE_RATIO >= 1.0，保留所有样本。")
            return all_mask_files

        random.seed(seed)
        n_neg_to_keep = int(len(negative_files) * ratio)
        downsampled_neg_files = random.sample(negative_files, n_neg_to_keep)
        
        print(f"降采样：保留 100% 的正样本 ({len(positive_files)} 个)")
        print(f"降采样：保留 {ratio*100:.1f}% 的负样本 ({n_neg_to_keep} / {len(negative_files)} 个)")

        final_file_list = positive_files + downsampled_neg_files
        random.shuffle(final_file_list) 
        
        print(f"最终训练集大小: {len(final_file_list)} 个图块。")
        return final_file_list

    def _calculate_stats(self, file_list, img_dir):
        """
        [内部方法] 在 *采样后* 的文件列表上计算均值和标准差。
        """
        print(f"开始计算 {len(file_list)} 个训练图块的均值和标准差...")
        channel_sum = np.zeros(self.num_channels, dtype=np.float64)
        channel_sum_sq = np.zeros(self.num_channels, dtype=np.float64)
        pixel_count = 0

        for mask_path in tqdm(file_list, desc="Calculating Stats"):
            img_path = img_dir / mask_path.name
            if not img_path.exists(): continue

            try:
                with rasterio.open(img_path) as src:
                    img_data = src.read().astype(np.float64)
                    if img_data.shape[0] != self.num_channels: continue
                    
                    channel_sum += np.sum(img_data, axis=(1, 2))
                    channel_sum_sq += np.sum(np.square(img_data), axis=(1, 2))
                    pixel_count += img_data.shape[1] * img_data.shape[2]
            except Exception as e:
                print(f"Error reading {img_path.name}: {e}")

        if pixel_count == 0:
            raise RuntimeError("未能处理任何文件来计算统计数据。")

        mean = torch.tensor(channel_sum / pixel_count, dtype=torch.float32)
        variance = (channel_sum_sq / pixel_count) - np.square(mean.numpy())
        std = torch.tensor(np.sqrt(variance), dtype=torch.float32)

        return mean, std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mask_path = self.file_list[idx]
        img_path = self.img_dir / mask_path.name
        
        try:
            with rasterio.open(img_path) as src_img:
                image = src_img.read().astype(np.float32)
            with rasterio.open(mask_path) as src_mask:
                mask = src_mask.read(1).astype(np.int64)

            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = (image - self.mean) / self.std
            
            if self.split == 'train':
                # (数据增强)
                if torch.rand(1) > 0.5: image, mask = TF.hflip(image), TF.hflip(mask)
                if torch.rand(1) > 0.5: image, mask = TF.vflip(image), TF.vflip(mask)
                if torch.rand(1) > 0.5:
                    angle = (torch.rand(1).item() - 0.5) * 60
                    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                    mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
                if torch.rand(1) > 0.5:
                    noise_std = torch.rand(1).item() * 0.1 
                    image = image + torch.randn_like(image) * noise_std
            
            return image, mask

        except Exception as e:
            print(f"Error loading sample {idx} ({mask_path.name}): {e}")
            return None, None

# --- 2. Dataloader 辅助函数 (用户调用的函数) ---

def get_loaders(data_dir, 
                batch_size, 
                num_workers,
                neg_sample_ratio=1.0, 
                seed=42):
    """
    创建 train 和 val Dataloaders。
    
    Args:
        data_dir (str or Path): 数据集根目录 (e.g., "/mnt/data1/rove/dataset/S1_Water")
        batch_size (int): 批量大小
        num_workers (int): Dataloader 的工作进程数
        neg_sample_ratio (float, optional): 
            (仅 'train') 保留的纯陆地样本比例。默认为 1.0 (不采样)。
        seed (int, optional): 
            (仅 'train') 随机采样的种子。默认为 42。
    """
    
    def collate_fn_skip_none(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch:
            return torch.tensor([]), torch.tensor([])
        return torch.utils.data.dataloader.default_collate(batch)

    data_dir = Path(data_dir)

    # 1. 创建训练集。这将触发采样和统计计算（或加载缓存）
    train_dataset = S1WaterDataset(
        data_dir=data_dir,
        split='train',
        neg_sample_ratio=neg_sample_ratio,
        seed=seed
    )
    
    # 2. 创建验证集，并传入训练集的统计数据
    val_dataset = S1WaterDataset(
        data_dir=data_dir,
        split='val',
        override_stats=(train_dataset.mean.squeeze(), train_dataset.std.squeeze())
    )
    
    # 3. 创建 Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_skip_none
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_skip_none
    )
    
    return train_loader, val_loader

# --- 3. 示例用法 ---
if __name__ == '__main__':
    
    print("--- 开始创建 Dataloaders (带自动采样和缓存) ---")
    
    DATA_ROOT = "/mnt/data1/rove/dataset/S1_Water" # 您的主 S1_Water 目录
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # !! 在这里测试您的新参数 !!
    # 第一次运行会很慢，第二次运行会立即加载
    NEG_RATIO = 0.5 

    train_loader, val_loader = get_loaders(
        data_dir=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        neg_sample_ratio=NEG_RATIO, # <-- 传入参数
        seed=42
    )
    
    print(f"\nTrain Dataloader (采样率 {NEG_RATIO}): {len(train_loader)} 批次")
    print(f"Val Dataloader (完整):       {len(val_loader)} 批次")
    
    print("\n--- 测试加载一个批次 (Train) ---")
    try:
        images, masks = next(iter(train_loader))
        print(f"图像 (Images) 批次形状: {images.shape}")
        print(f"图像 (Images) 数据类型: {images.dtype}")
        print(f"掩膜 (Masks) 批次形状: {masks.shape}")
        print(f"掩膜 (Masks) 数据类型: {masks.dtype}")
    except Exception as e:
        print(f"加载批次时出错: {e}")