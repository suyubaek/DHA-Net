import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from pathlib import Path
import warnings
import torchvision.transforms.functional as TF
import random
from tqdm import tqdm


class S1WaterDataset(Dataset):
    def __init__(self, data_dir, split, 
                 neg_sample_ratio=1.0, 
                 seed=42, 
                 override_stats=None,
                 preload=True):
        """
        Args:
            data_dir (Path or str): 数据集根目录 (S1_Water)
            split (str): "train", "val", 或 "test"
            neg_sample_ratio (float): (仅用于 'train') 保留的纯陆地样本比例。
            seed (int): (仅用于 'train') 随机采样的种子。
            override_stats (tuple): (仅用于 'val'/'test') 
                                    一个 (mean, std) 元组，用于标准化。
            preload (bool): 是否将数据预加载到内存。建议 Train/Val 为 True，Test 为 False。
        """
        self.split = split
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / split / "img"
        self.mask_dir = self.data_dir / split / "mask"
        self.preload = preload
        
        self.num_channels = 2 # (VV, VH)
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

        if split == 'train':
            cache_filename = f"train_cache_ratio_{neg_sample_ratio}_seed_{seed}.pt"
            cache_file = self.data_dir / cache_filename
            
            if cache_file.exists():
                print(f"--- 正在从缓存加载训练数据: {cache_filename} ---")
                cache_data = torch.load(cache_file, weights_only=False)
                self.file_list = cache_data['file_list']
                self.mean = cache_data['mean']
                self.std = cache_data['std']
            else:
                print(f"--- 未找到缓存，正在为 {split} split 创建新数据... ---")
                print(f"--- (这在第一次运行时会比较慢) ---")
                self.file_list = self._get_sampled_file_list(
                    self.mask_dir, neg_sample_ratio, seed
                )
                self.mean, self.std = self._calculate_stats(
                    self.file_list, self.img_dir
                )
                print(f"--- 计算完成，保存到缓存: {cache_file} ---")
                torch.save({
                    'file_list': self.file_list,
                    'mean': self.mean,
                    'std': self.std
                }, cache_file)

        else:
            print(f"--- 正在为 {split} split 加载完整数据... ---")
            if override_stats is None:
                raise ValueError(f"{split} split 必须提供 'override_stats' (来自训练集)")
            
            self.file_list = sorted(list(self.mask_dir.glob("*.tif")))
            self.mean, self.std = override_stats # 使用传入的 train stats
        if not self.file_list:
            raise FileNotFoundError(f"在 {self.mask_dir} 中未找到 .tif 文件。")

        self.mean = self.mean.view(self.num_channels, 1, 1)
        self.std = self.std.view(self.num_channels, 1, 1)

        # --- Data Preloading ---
        self.images = []
        self.masks = []
        
        if self.preload:
            print(f"--- Preloading {len(self.file_list)} images into RAM for {split} split... ---")
            for mask_path in tqdm(self.file_list, desc=f"Loading {split} data"):
                img_path = self.img_dir / mask_path.name
                try:
                    with rasterio.open(img_path) as src_img:
                        image = src_img.read().astype(np.float32)
                    with rasterio.open(mask_path) as src_mask:
                        mask = src_mask.read(1).astype(np.int64)
                    
                    # Convert to tensor and normalize immediately to save processing time later
                    image_t = torch.from_numpy(image)
                    mask_t = torch.from_numpy(mask)
                    
                    # Normalize here to save CPU time during training
                    image_t = (image_t - self.mean) / self.std
                    
                    self.images.append(image_t)
                    self.masks.append(mask_t)
                    
                except Exception as e:
                    print(f"Error loading {mask_path.name}: {e}")
                    pass
            
            if len(self.images) != len(self.file_list):
                 print(f"Warning: Only loaded {len(self.images)}/{len(self.file_list)} images.")
        else:
            print(f"--- {split} split: Lazy loading mode (Reading from disk on-the-fly) ---")

    def _get_sampled_file_list(self, mask_dir, ratio, seed):
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

    def _infer_cls_label(self, mask: torch.Tensor) -> torch.Tensor:
        has_water = torch.any(mask > 0)
        has_land = torch.any(mask == 0)
        if has_water and has_land:
            cls_id = 1  # 混合
        elif has_water:
            cls_id = 2  # 纯水
        else:
            cls_id = 0  # 纯陆地
        return torch.tensor(cls_id, dtype=torch.long)

    def __len__(self):
        if self.preload:
            return len(self.images)
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            if self.preload:
                # Retrieve from memory
                image = self.images[idx]
                mask = self.masks[idx]
            else:
                # Lazy load from disk
                mask_path = self.file_list[idx]
                img_path = self.img_dir / mask_path.name
                
                with rasterio.open(img_path) as src_img:
                    image = src_img.read().astype(np.float32)
                with rasterio.open(mask_path) as src_mask:
                    mask = src_mask.read(1).astype(np.int64)

                image = torch.from_numpy(image)
                mask = torch.from_numpy(mask)
                image = (image - self.mean) / self.std

            cls_label = self._infer_cls_label(mask)
            
            # CPU Augmentation: Only Flips (Only for train)
            if self.split == 'train':
                if torch.rand(1) > 0.5: image, mask = TF.hflip(image), TF.hflip(mask)
                if torch.rand(1) > 0.5: image, mask = TF.vflip(image), TF.vflip(mask)
                
                # Rotation and Noise moved to GPU
            
            return image, mask, cls_label

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None, None




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
            empty = torch.empty(0)
            empty_long = torch.empty(0, dtype=torch.long)
            return empty, empty_long, empty_long
        return torch.utils.data.dataloader.default_collate(batch)

    data_dir = Path(data_dir)

    train_dataset = S1WaterDataset(
        data_dir=data_dir,
        split='train',
        neg_sample_ratio=neg_sample_ratio,
        seed=seed,
        preload=True
    )
    val_dataset = S1WaterDataset(
        data_dir=data_dir,
        split='val',
        override_stats=(train_dataset.mean.squeeze(), train_dataset.std.squeeze()),
        preload=True
    )
    test_dataset = S1WaterDataset(
        data_dir=data_dir,
        split='test',
        override_stats=(train_dataset.mean.squeeze(), train_dataset.std.squeeze()),
        preload=False
    )

    
    # 3. 创建 Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_skip_none,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_skip_none,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_skip_none,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    
    print("--- 开始创建 Dataloaders (带自动采样和缓存) ---")
    
    DATA_ROOT = "/mnt/data1/rove/dataset/S1_Water" # 您的主 S1_Water 目录
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # !! 在这里测试您的新参数 !!
    # 第一次运行会很慢，第二次运行会立即加载
    NEG_RATIO = 0.3

    train_loader, val_loader = get_loaders(
        data_dir=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        neg_sample_ratio=NEG_RATIO, # <-- 传入参数
        seed=3047
    )
    
    print(f"\nTrain Dataloader (采样率 {NEG_RATIO}): {len(train_loader)} 批次")
    print(f"Val Dataloader (完整):       {len(val_loader)} 批次")
    
    print("\n--- 测试加载一个批次 (Train) ---")
    try:
        # 注意：__getitem__ 返回 3 个值 (image, mask, cls_label)
        batch_data = next(iter(train_loader))
        images, masks, cls_labels = batch_data
        
        print(f"图像 (Images) 批次形状: {images.shape}")
        print(f"图像 (Images) 数据类型: {images.dtype}")
        print(f"掩膜 (Masks) 批次形状: {masks.shape}")
        print(f"掩膜 (Masks) 数据类型: {masks.dtype}")
        print(f"类别 (Labels) 批次形状: {cls_labels.shape}")

        print("\n--- 数据分布详细检查 ---")
        
        # 1. 图像数值分布检查 (检查标准化效果)
        print(f"[图像分布]")
        print(f"  Min: {images.min():.4f}")
        print(f"  Max: {images.max():.4f}")
        print(f"  Mean: {images.mean():.4f} (理想值接近 0)")
        print(f"  Std:  {images.std():.4f}  (理想值接近 1)")
        
        # 检查是否有 NaN 或 Inf
        if torch.isnan(images).any() or torch.isinf(images).any():
            print("  [警告] 图像数据中包含 NaN 或 Inf！")
        else:
            print("  数值有效性检查通过 (无 NaN/Inf)")

        # 2. 掩膜数值分布检查
        print(f"\n[掩膜分布]")
        unique_vals = torch.unique(masks)
        print(f"  包含的唯一值: {unique_vals.tolist()}")
        
        water_pixels = (masks == 1).sum().item()
        total_pixels = masks.numel()
        water_ratio = water_pixels / total_pixels
        print(f"  水体像素占比 (Batch): {water_ratio:.2%}")
        
        if not torch.all((masks == 0) | (masks == 1)):
             print("  [警告] 掩膜中包含非 0/1 的异常值！")

        # 3. 类别标签分布
        print(f"\n[图像级标签分布]")
        # 0: 纯陆地, 1: 混合, 2: 纯水
        cls_counts = torch.bincount(cls_labels, minlength=3)
        print(f"  纯陆地 (0): {cls_counts[0]} 张")
        print(f"  混合 (1):   {cls_counts[1]} 张")
        print(f"  纯水 (2):   {cls_counts[2]} 张")

    except Exception as e:
        print(f"加载批次时出错: {e}")
        import traceback
        traceback.print_exc()