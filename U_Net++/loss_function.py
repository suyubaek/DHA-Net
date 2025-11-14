from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[Sequence[float]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        buffer = None
        if weight is not None:
            buffer = torch.as_tensor(weight, dtype=torch.float32)
        self.register_buffer("weight", buffer, persistent=False)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Expect logits shaped as (N, C, ...) and target shaped as (N, ...).
        weight = None if self.weight is None else self.weight.to(logits.device)
        return F.cross_entropy(
            logits,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(-1)
        target_flat = target.view(-1)

        intersection = (probs_flat * target_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1.0 - dice_score


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()

        if logits.shape != target.shape:
            if logits.dim() == 4 and target.dim() == 3:
                target = target.unsqueeze(1)
            else:
                # 如果维度不匹配且不是预期的情况，则抛出错误
                raise ValueError(
                    f"Shape mismatch: logits {logits.shape}, target {target.shape}. "
                    "Cannot automatically unsqueeze."
                )
        
        dice_loss = self.dice_loss(logits, target)
        bce_loss = self.bce_loss(logits, target)

        combined_loss = (self.dice_weight * dice_loss) + (self.bce_weight * bce_loss)
        return combined_loss
