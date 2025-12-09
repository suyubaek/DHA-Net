from typing import Dict, Optional, Sequence, Tuple

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


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        # (N) -> (N, 1)
        labels = labels.view(-1, 1)

        # mask[i, j] = 1 if labels[i] == labels[j]
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 对特征进行 L2 归一化
        features = F.normalize(features, dim=1)

        # 计算余弦相似度并除以温度
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # 为了数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 构造一个 (N, N) 的对角线为 0 的掩码
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        # mask = 相同标签的样本 (除去自身)
        mask = mask * logits_mask

        # exp_logits (N, N)
        exp_logits = torch.exp(logits) * logits_mask
        
        # log_prob (N, 1)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # 计算正样本对的平均 log-probability
        # mask.sum(1) 是每个样本的正样本对数量
        mask_sum = mask.sum(1)
        # 处理一个类只有一个样本的情况 (mask_sum = 0)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # 最终损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        align_lambda: float = 0.1,
        cls_lambda: float = 0.1,
        con_temperature: float = 0.07,
    ) -> None:
        """
        组合分割、对齐、分类三种损失
        Args:
            align_lambda: 对齐损失权重
            cls_lambda: 分类损失权重
            con_temperature: SupCon 温度
        """
        super().__init__()
        self.align_lambda = align_lambda
        self.cls_lambda = cls_lambda

        self.loss_seg = DiceBCELoss()
        self.loss_align = SupervisedContrastiveLoss(temperature=con_temperature)
        self.loss_cls = CrossEntropyLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,
        align_emb: torch.Tensor,
        cls_logits: torch.Tensor,
        seg_labels: torch.Tensor,
        align_labels: torch.Tensor,
        cls_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            seg_logits: (B, 1, H, W)
            align_emb: (B, C) —— CLS embedding
            cls_logits: (B, num_classes)
            seg_labels: (B, 1, H, W)
            align_labels: (B,)
            cls_labels: (B,)
        """
        loss_seg = self.loss_seg(seg_logits, seg_labels)
        loss_align = self.loss_align(align_emb, align_labels)
        loss_cls = self.loss_cls(cls_logits, cls_labels)

        total_loss = loss_seg + self.align_lambda * loss_align + self.cls_lambda * loss_cls

        loss_components = {
            "total": total_loss,
            "seg": loss_seg,
            "align": loss_align,
            "cls": loss_cls,
        }
        return total_loss, loss_components


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, logits, target):
        target = target.float()
        if logits.shape != target.shape:
            if logits.dim() == 4 and target.dim() == 3:
                target = target.unsqueeze(1)
            else:
                raise ValueError(f"Shape mismatch: logits {logits.shape}, target {target.shape}")

        dice = self.dice_loss(logits, target)
        focal = self.focal_loss(logits, target)
        
        # Loss = α * L_dice + (1 - α) * L_focal
        return self.alpha * dice + (1 - self.alpha) * focal


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # 惩罚 FP (误报)
        self.beta = beta    # 惩罚 FN (漏报) -> 调大这个值提升 Recall
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: Logits (未经过 Sigmoid)
        # targets: 0 or 1
        
        inputs = torch.sigmoid(inputs)
        
        # Flatten: (N, C, H, W) -> (N*C*H*W)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - Tversky

class MixedLoss(nn.Module):
    """
    结合 BCE 和 Tversky Loss
    """
    def __init__(self, alpha=0.3, beta=0.7, weight_bce=0.5, weight_tversky=0.5):
        super(MixedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.weight_bce = weight_bce
        self.weight_tversky = weight_tversky

    def forward(self, inputs, targets):
        # 1. 确保 target 是 float 类型 (BCE 需要 float target)
        targets = targets.float()
        
        # 2. 维度对齐: 如果 inputs 是 (B, 1, H, W) 而 targets 是 (B, H, W)，则扩充 targets
        if inputs.shape != targets.shape:
            if inputs.dim() == 4 and targets.dim() == 3:
                targets = targets.unsqueeze(1)
            else:
                # 如果维度完全对不上，抛出异常
                raise ValueError(f"Shape mismatch: inputs {inputs.shape}, targets {targets.shape}")

        loss_bce = self.bce(inputs, targets)
        loss_tversky = self.tversky(inputs, targets)
        
        return self.weight_bce * loss_bce + self.weight_tversky * loss_tversky