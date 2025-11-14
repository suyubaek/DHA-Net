import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- 1. 定义数据增强管线 ---

# 训练集的数据增强
# - 几何变换: 翻转、旋转
# - 归一化: 我们知道数据是 uint8 (0-255)，所以我们直接除以 255.0
# - 转换为 Tensor
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # 归一化: max_pixel_value=255.0 会自动帮我们做 (img / 255.0)
        A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2(), # 转换为 (C, H, W) 的 Pytorch Tensor
    ]
)

# 验证集的数据增强 (不应包含翻转等几何变换)
val_transform = A.Compose(
    [
        A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


# --- 2. 自定义 Dataset 类 ---

class WaterDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): 包含 'train' 和 'val' 文件夹的根目录
            split (string): "train" 或 "val"
            transform (callable, optional): 应用于样本的 Albumentations 变换
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.image_dir = os.path.join(root_dir, split, "image")
        self.mask_dir = os.path.join(root_dir, split, "mask")
        
        # 获取所有影像的文件名 (假设 image 和 mask 文件名一一对应)
        self.image_files = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. 获取文件名和路径
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # 2. 读取数据 (用 NumPy 数组格式)
        #    我们知道影像是 L (单通道)
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # 3. 预处理 Mask (非常重要!)
        #    我们的 mask 可能是 0/255，我们需要 0/1 的浮点数
        if np.max(mask) > 1.0:
            mask = mask / 255.0
        
        # 确保 mask 是 float32，以便 PyTorch 正确处理
        mask = mask.astype(np.float32)
        
        # 4. 应用数据增强 (Albumentations)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 5. 确保 mask 也有一个通道维度 (H, W) -> (1, H, W)
        #    这对于 (B, C, H, W) 格式的分割损失函数至关重要
        mask = mask.unsqueeze(0)
        
        return image, mask


def get_loaders(root_dir, batch_size, num_workers=4, pin_memory=True):
    train_dataset = WaterDataset(
        root_dir=root_dir,
        split="train",
        transform=train_transform
    )
    
    val_dataset = WaterDataset(
        root_dir=root_dir,
        split="val",
        transform=val_transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True, # 训练集需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    DATA_ROOT = "/mnt/sda1/songyufei/dataset/lancang"
    
    BATCH_SIZE = 4
    
    print(f"正在使用数据根目录: {DATA_ROOT}")
    print("开始创建 DataLoaders...")
    
    try:
        train_loader, val_loader = get_loaders(
            root_dir=DATA_ROOT,
            batch_size=BATCH_SIZE
        )
        print(f"成功创建 Train Loader (共 {len(train_loader.dataset)} 样本)")
        print(f"成功创建 Val Loader   (共 {len(val_loader.dataset)} 样本)")
        
        print("\n--- 正在测试从 Train Loader 取一个批次... ---")
        
        # 从 train_loader 取一个 batch
        images, masks = next(iter(train_loader))
        
        # 打印这个 batch 的信息
        print(f"  影像批次形状 (B, C, H, W): {images.shape}")
        print(f"  掩码批次形状 (B, C, H, W): {masks.shape}")
        print(f"  影像数据类型: {images.dtype}")
        print(f"  掩码数据类型: {masks.dtype}")
        print(f"  影像 Min/Max (归一化后): {images.min():.2f} / {images.max():.2f}")
        print(f"  掩码 Min/Max (归一化后): {masks.min():.2f} / {masks.max():.2f}")
        print(f"  掩码中的唯一值: {torch.unique(masks)}")
        
        print("\n--- 测试通过！---")
        print("你现在可以在你的主训练脚本中导入 get_loaders 函数了。")
        
    except FileNotFoundError:
        print("\n--- !!! 测试失败: 文件未找到 !!! ---")
        print(f"错误: 在 {DATA_ROOT} 中找不到 'train' 或 'val' 文件夹。")
        print("请打开 dataprocess.py 文件，并修改 'DATA_ROOT' 变量为您正确的路径。")
    except Exception as e:
        print(f"\n--- !!! 测试失败: 发生意外错误 !!! ---")
        print(f"错误: {e}")