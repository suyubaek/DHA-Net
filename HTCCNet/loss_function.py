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
    def __init__(self, contrastive_lambda=0.1, con_temperature=0.07):
        """
        聚合分割损失和对比损失
        Args:
            contrastive_lambda (float): 对比损失的权重
            con_temperature (float): SupCon 损失的温度参数
        """
        super().__init__()
        self.contrastive_lambda = contrastive_lambda
        
        # 1. 实例化各个子损失
        self.loss_seg = DiceBCELoss()
        self.loss_con = SupervisedContrastiveLoss(temperature=con_temperature)

    def forward(self, seg_logits, con_emb, seg_labels, con_labels):
        """
        计算所有损失并返回
        Args:
            seg_logits: 来自模型的分割输出 (B, 1, H, W)
            con_emb: 来自模型的对比嵌入 (B, C_con)
            seg_labels: 真实的分割掩码 (B, 1, H, W)
            con_labels: 真实的类别标签 (B,)
        
        Returns:
            total_loss (torch.Tensor): 用于反向传播的总损失
            loss_components (dict): 包含各个子损失的字典，用于 logging
        """
        
        # 1. 计算各个损失
        loss_seg = self.loss_seg(seg_logits, seg_labels)
        loss_con = self.loss_con(con_emb, con_labels)
        
        # 2. 加权求和
        total_loss = loss_seg + self.contrastive_lambda * loss_con
        
        # 3. 准备一个字典用于日志记录
        loss_components = {
            "total": total_loss,
            "seg": loss_seg,
            "con": loss_con
        }
        
        return total_loss, loss_components