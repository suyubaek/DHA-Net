from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def _resolve_root(path: Path | str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def _collect_files(directory: Path) -> Dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Expected directory missing: {directory}")

    files: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stem = path.stem
        if stem in files:
            raise ValueError(f"Duplicate file stem detected in {directory}: {stem}")
        files[stem] = path
    if not files:
        raise FileNotFoundError(f"No supported files found in {directory}")
    return files


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as img:
            array = np.array(img)
    return array


def _prepare_input(array: np.ndarray) -> torch.Tensor:
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    elif array.dtype == np.uint16:
        array = array.astype(np.float32) / 65535.0
    else:
        array = array.astype(np.float32)

    if array.ndim == 2:
        tensor = torch.from_numpy(array).unsqueeze(0)
    elif array.ndim == 3:
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported array shape for input tensor: {array.shape}")
    return tensor


def _prepare_mask(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Segmentation mask must be HxW, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        array = np.rint(array).astype(np.int64)
    else:
        array = array.astype(np.int64)
    # 新增：将255全部转为1
    array[array == 255] = 1
    tensor = torch.from_numpy(array)
    return tensor


@dataclass
class SamplePaths:
    key: str
    sar: Path
    optical: Path
    mask: Path


class WHUOptSarDataset(Dataset):
    """Dataset wrapper for the WHU OPT-SAR multimodal segmentation benchmark."""

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        transform: Optional[
            Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")

        self.root = _resolve_root(root)
        self.split = split
        self.transform = transform

        # 1. 定位目录
        base = self.root / split
        sar_dir = base / "sar"
        opt_dir = base / "opt"
        mask_dir = base / "mask"

        # 2. 收集所有文件路径 (返回 Dict[stem, Path])
        sar_files = _collect_files(sar_dir)
        opt_files = _collect_files(opt_dir)
        mask_files = _collect_files(mask_dir)

        # 3. 获取所有文件名的 stems (集合)
        sar_stems = set(sar_files.keys())
        opt_stems = set(opt_files.keys())
        mask_stems = set(mask_files.keys())

        # 4. 找到三者共有的 stems (交集)
        common_keys = sorted(sar_stems & opt_stems & mask_stems)
        if not common_keys:
            raise RuntimeError(
                "No matching triplets found across sar/opt/mask directories"
            )

        # 5. 诊断性检查 (逻辑与原来一致，结构更清晰)
        # 检查是否存在 "有 mask，但缺少对应 data" 的情况
        missing_sar_for_masks = sorted(mask_stems - sar_stems)
        missing_opt_for_masks = sorted(mask_stems - opt_stems)

        error_hints = []
        if missing_sar_for_masks:
            error_hints.append(
                f"missing SAR files for mask keys: {missing_sar_for_masks[:5]}"
            )
        if missing_opt_for_masks:
            error_hints.append(
                f"missing OPT files for mask keys: {missing_opt_for_masks[:5]}"
            )

        if error_hints:
            # 如果有任何缺失，统一抛出异常
            raise RuntimeError("Inconsistent dataset structure: " + ", ".join(error_hints))

        # 6. 创建最终的样本路径列表
        self.samples: List[SamplePaths] = [
            SamplePaths(
                key=k, sar=sar_files[k], optical=opt_files[k], mask=mask_files[k]
            )
            for k in common_keys
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:  # <-- 返回类型改为 Any
        sample_paths = self.samples[index]

        # 1. 加载Numpy数组
        sar_array = _load_array(sample_paths.sar)
        opt_array = _load_array(sample_paths.optical)
        mask_array = _load_array(sample_paths.mask)

        # 2. 准备Tensors，放入一个字典
        data = {
            "sar": _prepare_input(sar_array),
            "opt": _prepare_input(opt_array),
            "mask": _prepare_mask(mask_array),
        }

        # 3. 仅对Tensors字典应用 transform
        if self.transform is not None:
            data = self.transform(data)

        # 4. 在 transform 之后，再加入 key 等元数据
        data["key"] = sample_paths.key

        return data


def create_whuoptsar_dataloader(
    root: Path | str,
    split: str,
    batch_size: int,
    transform: Optional[
        Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    ] = None,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    dataset = WHUOptSarDataset(root=root, split=split, transform=transform)
    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )