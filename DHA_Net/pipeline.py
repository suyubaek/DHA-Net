import os
import sys
from datetime import datetime
from datetime import timedelta
import random
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

from model import Model
from loss_function import MixedLoss
from dataprocess import get_loaders, S1WaterDataset
from config import config
from metrics import calculate_metrics
from message2lark import send_message
from visualization import create_sample_images
import torchvision.transforms.functional as TF


class GPUAugmentor:
    """
    GPU-accelerated augmentations: Rotation and Noise.
    """
    def __init__(self, rotate_prob=0.5, noise_prob=0.5, max_angle=60, noise_std=0.1):
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.max_angle = max_angle
        self.noise_std = noise_std

    def __call__(self, images, masks):
        # images: (B, C, H, W)
        # masks: (B, H, W) or (B, 1, H, W)
        
        B = images.shape[0]
        
        # 1. Random Rotation
        if torch.rand(1) < self.rotate_prob:
            angle = (torch.rand(1).item() - 0.5) * 2 * self.max_angle
            images = TF.rotate(images, angle, interpolation=TF.InterpolationMode.BILINEAR)
            
            # Ensure mask has channel dim for rotation
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
                masks = TF.rotate(masks.float(), angle, interpolation=TF.InterpolationMode.NEAREST)
                masks = masks.squeeze(1).long()
            else:
                masks = TF.rotate(masks.float(), angle, interpolation=TF.InterpolationMode.NEAREST).long()

        # 2. Random Noise
        if torch.rand(1) < self.noise_prob:
            noise_scale = torch.rand(1).item() * self.noise_std
            noise = torch.randn_like(images) * noise_scale
            images = images + noise
            
        return images, masks

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def save_checkpoint(model, optimizer, epoch, iou, path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': iou,
    }
    
    torch.save(state, path)

def train_one_epoch(model, train_loader, optimizer, loss_fc, device, epoch,
                    train_dataset=None, grad_cam=None, epoch_state=None):
    model.train()
    total_loss = seg_loss_total = bce_loss_total = tversky_loss_total = 0.0
    iou, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    metric_count = 0
    
    # Initialize GPU Augmentor
    # augmentor = GPUAugmentor(rotate_prob=0.5, noise_prob=0.5)#数据增强
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images, labels, class_labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)

        # Apply GPU Augmentation
        # with torch.no_grad():
        #     images, labels = augmentor(images, labels)

        optimizer.zero_grad() #清空历史梯度

        #seg_logits = model(images) #前向传播  分割
        model_output=model(images)

        # 模型如果返回 tuple（例如 (seg, aux_feat, ...)），先取第一个作为分割输出
        if isinstance(model_output, (list,tuple)):
            seg_logits = model_output[0]
            s1=model_output[1]
            s2=model_output[2]
        else:
            seg_logits = model_output
            s1=s2=None

        if isinstance(seg_logits, list):
            # Deep supervision case
            loss = 0
            bce_loss_sum = 0
            tversky_loss_sum = 0
            for logits in seg_logits:
                l, b, t = loss_fc(logits, labels)
                loss += l
                bce_loss_sum += b
                tversky_loss_sum += t
            loss /= len(seg_logits)
            #分别记录和观察每个loss的变化情况
            loss_bce = bce_loss_sum / len(seg_logits)
            loss_tversky = tversky_loss_sum / len(seg_logits)
            seg_logits = seg_logits[0] # Use the first output for metrics
        else:
            loss, loss_bce, loss_tversky = loss_fc(seg_logits, labels)
        
        loss.backward() #计算梯度

        # Grad-CAM & 余弦相似度：只在每 10 个 epoch 的第一个 batch 上做一次
        if (
            grad_cam is not None
            and train_dataset is not None
            and batch_idx == 0
            and (epoch % 10 == 0)
        ):
            try:
                stats = (train_dataset.mean, train_dataset.std)

                # 先获取所有 CAM 并进行可视化
                cams = grad_cam.get_all(upsample_size=images.shape[-2:])
                if cams:
                    log_grad_cam(images, cams, epoch, stats)

                # 1) Stem/First Layer：样本间特征余弦相似度
                stem_act = grad_cam.activations.get("Stem_Conv1")
                if stem_act is not None and stem_act.size(0) > 1:
                    stem_feat = stem_act.mean(dim=(2, 3))  # (B, C)
                    sim_mat = F.cosine_similarity(
                        stem_feat.unsqueeze(1), stem_feat.unsqueeze(0), dim=-1
                    )  # (B, B)
                    mask = ~torch.eye(
                        sim_mat.size(0), dtype=torch.bool, device=sim_mat.device
                    )
                    sim_vals = sim_mat[mask]
                    if sim_vals.numel() > 0:
                        wandb.log(
                            {
                                "CosSim/Stem_Sample_Mean": sim_vals.mean().item(),
                                "CosSim/Stem_Sample_Std": sim_vals.std().item(),
                            }
                        )

                # 2) Bottleneck（ASPP 输出）：相邻 epoch 特征余弦相似度
                if epoch_state is not None:
                    bott_act = grad_cam.activations.get("Bottleneck_ASPP")
                    if bott_act is not None:
                        cur_vec = bott_act.mean(dim=(0, 2, 3))  # (C,)
                        prev_vec = epoch_state.get("prev_bottleneck")
                        if prev_vec is not None:
                            cs = F.cosine_similarity(cur_vec, prev_vec, dim=0).item()
                            wandb.log(
                                {"CosSim/Bottleneck_Epoch_to_prev": cs}
                            )
                        epoch_state["prev_bottleneck"] = cur_vec.detach()

                # 2.5) CNN vs ViT 分支：全局特征余弦相似度（判断两条支路是否互补）
                cnn_act = grad_cam.activations.get("Branch_CNN")
                vit_act = grad_cam.activations.get("Branch_ViT")
                if cnn_act is not None and vit_act is not None:
                    # 通过 GAP 得到每个样本的全局特征向量
                    cnn_feat = cnn_act.mean(dim=(2, 3))  # (B, Cc)
                    vit_feat = vit_act.mean(dim=(2, 3))  # (B, Cv)
                    Bc, Cc = cnn_feat.shape
                    Bv, Cv = vit_feat.shape
                    Bmin = min(Bc, Bv)
                    if Bmin > 0:
                        # 若通道数不同，先降维到相同维度
                        Cmin = min(Cc, Cv)
                        cf = cnn_feat[:Bmin, :Cmin]
                        vf = vit_feat[:Bmin, :Cmin]
                        cs = F.cosine_similarity(cf, vf, dim=1)  # (Bmin,)
                        wandb.log(
                            {
                                "CosSim/Branch_CNN_vs_ViT_Mean": cs.mean().item(),
                                "CosSim/Branch_CNN_vs_ViT_Std": cs.std().item()
                                if cs.numel() > 1
                                else 0.0,
                            }
                        )

                # 3) 预测前一层：水体 vs 背景特征余弦相似度
                pen_act = grad_cam.activations.get("Penultimate")
                if pen_act is not None:
                    with torch.no_grad():
                        B, C, H, W = pen_act.shape
                        feats = pen_act.permute(0, 2, 3, 1).reshape(-1, C)
                        lbl_flat = labels.view(-1)
                        water_mask = lbl_flat > 0
                        back_mask = lbl_flat == 0
                        if water_mask.any() and back_mask.any():
                            wf = feats[water_mask]
                            bf = feats[back_mask]
                            max_samples = 20000
                            if wf.size(0) > max_samples:
                                idx = torch.randperm(wf.size(0), device=wf.device)[
                                    :max_samples
                                ]
                                wf = wf[idx]
                            if bf.size(0) > max_samples:
                                idx = torch.randperm(bf.size(0), device=bf.device)[
                                    :max_samples
                                ]
                                bf = bf[idx]
                            w_mean = wf.mean(dim=0)
                            b_mean = bf.mean(dim=0)
                            cs = F.cosine_similarity(
                                w_mean.unsqueeze(0), b_mean.unsqueeze(0), dim=-1
                            ).item()
                            wandb.log(
                                {
                                    "CosSim/Penultimate_Water_vs_Background": cs
                                }
                            )

            except Exception as e:
                logging.warning(f"Grad-CAM logging failed: {e}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #梯度裁剪，防止梯度爆炸 #通过什么指标确认这个步骤是有用的
        optimizer.step() #更新参数

        if grad_cam is not None:
            grad_cam.clear()

        total_loss += loss.item()
        seg_loss_total += loss.item()
        bce_loss_total += loss_bce.item()
        tversky_loss_total += loss_tversky.item()
        
        # Calculate metrics every 50 batches
        if (batch_idx + 1) % 50 == 0:
            outputs = torch.sigmoid(seg_logits).detach()
            cur_iou, cur_precision, cur_recall, cur_f1 = calculate_metrics(outputs, labels)
            iou += cur_iou
            precision += cur_precision
            recall += cur_recall
            f1 += cur_f1
            metric_count += 1

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
            }
        )
    
    avg_loss = total_loss / len(train_loader)
    avg_seg_loss = seg_loss_total / len(train_loader)
    avg_bce_loss = bce_loss_total / len(train_loader)
    avg_tversky_loss = tversky_loss_total / len(train_loader)
    
    if metric_count > 0:
        iou /= metric_count
        precision /= metric_count
        recall /= metric_count
        f1 /= metric_count
    
    return (
        avg_loss,
        avg_seg_loss,
        avg_bce_loss,
        avg_tversky_loss,
        iou,
        precision,
        recall,
        f1,
    )

def validate(model, val_loader, loss_fc, device, epoch):
    model.eval()
    total_loss = seg_loss_total = bce_loss_total = tversky_loss_total = 0.0
    iou, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images, labels, class_labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            seg_logits = model(images)
            if isinstance(seg_logits, list):
                loss = 0
                bce_loss_sum = 0
                tversky_loss_sum = 0
                for logits in seg_logits:
                    l, b, t = loss_fc(logits, labels)
                    loss += l
                    bce_loss_sum += b
                    tversky_loss_sum += t
                loss /= len(seg_logits)
                loss_bce = bce_loss_sum / len(seg_logits)
                loss_tversky = tversky_loss_sum / len(seg_logits)
                seg_logits = seg_logits[0]
            else:
                loss, loss_bce, loss_tversky = loss_fc(seg_logits, labels)

            total_loss += loss.item()
            seg_loss_total += loss.item()
            bce_loss_total += loss_bce.item()
            tversky_loss_total += loss_tversky.item()

            outputs = torch.sigmoid(seg_logits).detach()
            cur_iou, cur_precision, cur_recall, cur_f1 = calculate_metrics(outputs, labels)
            iou += cur_iou
            precision += cur_precision
            recall += cur_recall
            f1 += cur_f1

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}"
                }
            )
    avg_loss = total_loss / len(val_loader)
    avg_seg_loss = seg_loss_total / len(val_loader)
    avg_bce_loss = bce_loss_total / len(val_loader)
    avg_tversky_loss = tversky_loss_total / len(val_loader)
    iou /= len(val_loader)
    precision /= len(val_loader)
    recall /= len(val_loader)
    f1 /= len(val_loader)

    return (
        avg_loss,
        avg_seg_loss,
        avg_bce_loss,
        avg_tversky_loss,
        iou,
        precision,
        recall,
        f1,
    )


def test(model, test_loader, loss_fc, device):
    model.eval()
    total_loss = seg_loss_total = 0.0
    iou, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, batch in enumerate(pbar):
            images, labels, class_labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            seg_logits = model(images)
            if isinstance(seg_logits, list):
                loss = 0
                for logits in seg_logits:
                    loss += loss_fc(logits, labels)
                loss /= len(seg_logits)
                seg_logits = seg_logits[0]
            else:
                loss = loss_fc(seg_logits, labels)

            total_loss += loss.item()
            seg_loss_total += loss.item()

            outputs = torch.sigmoid(seg_logits).detach()
            cur_iou, cur_precision, cur_recall, cur_f1 = calculate_metrics(outputs, labels)
            iou += cur_iou
            precision += cur_precision
            recall += cur_recall
            f1 += cur_f1

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}"
                }
            )
    
    avg_loss = total_loss / len(test_loader)
    avg_seg_loss = seg_loss_total / len(test_loader)
    iou /= len(test_loader)
    precision /= len(test_loader)
    recall /= len(test_loader)
    f1 /= len(test_loader)

    return (
        avg_loss,
        avg_seg_loss,
        iou,
        precision,
        recall,
        f1,
    )


class GradCAM:
    """通用 Grad-CAM 实现，可注册多个目标层。"""

    def __init__(self, model, target_layers: dict):
        """
        Args:
            model: 要可视化的模型
            target_layers (dict[str, nn.Module]): 名称 -> 层对象
        """
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name, layer in self.target_layers.items():
            # 只使用 forward hook，并在其中给输出注册 Tensor 级别的 grad hook，
            # 避免使用 register_full_backward_hook 引入的自定义 BackwardFunction
            self.handles.append(
                layer.register_forward_hook(self._make_forward_hook(name))
            )

    def _make_forward_hook(self, name):
        def hook(module, inp, out):
            # 保存前向激活
            self.activations[name] = out.detach()

            # 在输出 Tensor 上注册梯度 hook，保存反向梯度
            def _grad_hook(grad):
                self.gradients[name] = grad.detach()

            if isinstance(out, torch.Tensor):
                out.register_hook(_grad_hook)

        return hook

    def get_cam(self, layer_name, upsample_size=None):
        """返回指定层的 Grad-CAM，shape: (B, H, W)。"""
        acts = self.activations.get(layer_name)
        grads = self.gradients.get(layer_name)
        if acts is None or grads is None:
            return None

        # 1. GAP 得到通道权重
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # 2. 加权求和 -> (B, H, W)
        cam = (weights * acts).sum(dim=1)  # (B, H, W)
        cam = F.relu(cam)

        # 3. 归一化到 [0, 1]
        B, H, W = cam.shape
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0]
        cam_max = cam_flat.max(dim=1, keepdim=True)[0]
        cam_norm = (cam_flat - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam_norm.view(B, H, W)

        # 4. 上采样到输入分辨率
        if upsample_size is not None and (H, W) != upsample_size:
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=upsample_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return cam

    def get_all(self, upsample_size=None):
        cams = {}
        for name in self.target_layers.keys():
            cam = self.get_cam(name, upsample_size)
            if cam is not None:
                cams[name] = cam
        return cams

    def clear(self):
        self.activations.clear()
        self.gradients.clear()

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def log_grad_cam(images, cams_dict, epoch, dataset_stats, num_samples=4):
    """将多层 Grad-CAM 与输入图像一起可视化并记录到 wandb。

    Args:
        images (torch.Tensor): 输入图像 (B, C, H, W)，已标准化。
        cams_dict (dict[str, torch.Tensor]): 名称 -> CAM (B, H, W)。
        epoch (int): 当前 epoch。
        dataset_stats (tuple): (mean, std) 用于反标准化。
        num_samples (int): 可视化的样本数量。
    """
    try:
        if not cams_dict:
            return

        images = images.detach().cpu()
        num_samples = min(num_samples, images.shape[0])
        if num_samples == 0:
            return

        # 反标准化输入，便于可视化
        mean, std = dataset_stats
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
        vis_images = images * std + mean
        vis_images = vis_images[:, 0, :, :]  # 只看 VV 通道

        layer_names = list(cams_dict.keys())
        cams_cpu = {k: v.detach().cpu() for k, v in cams_dict.items()}
        num_layers = len(layer_names)

        fig, axes = plt.subplots(
            num_samples,
            1 + num_layers,
            figsize=(4 * (1 + num_layers), 4 * num_samples),
            constrained_layout=True,
        )
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # 原图
            ax0 = axes[i, 0]
            ax0.imshow(vis_images[i], cmap="gray")
            ax0.set_title(f"Sample {i+1} - Input")
            ax0.axis("off")

            # 不同层的 Grad-CAM 叠加
            for j, name in enumerate(layer_names):
                cam = cams_cpu[name][i]
                ax = axes[i, j + 1]
                ax.imshow(vis_images[i], cmap="gray")
                ax.imshow(cam, cmap="jet", alpha=0.5)
                ax.set_title(name)
                ax.axis("off")

        wandb.log({f"GradCAM/Epoch_{epoch}": wandb.Image(fig)})
        plt.close(fig)

    except Exception as e:
        logging.warning(f"Could not generate Grad-CAM visualizations: {e}")
        plt.close("all")


def main():
    # 设置随机种子
    set_seed(config["seed"])
    # 设置计算设备
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # W&B实验看板初始化
    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    wandb.init(
        project="lancang river",
        name=experiment_name,
        config=config,
        tags=["DHA"],
    )

    try:
        train_loader, val_loader, test_loader = get_loaders(
            data_dir=config["data_root"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            neg_sample_ratio=0.3,
            seed=config["seed"],
            preload=False
        )
        #考虑从验证集中抽取部分数据作为可视化集
        vis_dataset = S1WaterDataset(
            data_dir=config["data_root"],
            split='vis',
            override_stats=(train_loader.dataset.mean.squeeze(), train_loader.dataset.std.squeeze()),
            preload=False
        )
        vis_loader = torch.utils.data.DataLoader(
            vis_dataset,
            batch_size=12,
            shuffle=False,
            num_workers=4,
            collate_fn=val_loader.collate_fn
        )

        # 模型初始化
        model = Model(
            in_channels=config['in_channels'],
            num_classes=config['num_classes']
        )
        model = model.to(device)
        wandb.watch(model, log="all", log_freq=100)

        # Grad-CAM：监视多个关键层（包括 CNN / ViT 分支）
        target_layers = {
            # 编码器底层附近的卷积：使用 layer1[0].conv1（在当前 CNNEncoder 版本中始终存在）
            "Stem_Conv1": model.cnn.layer1[0].conv1,
            # CNN 分支：layer4 输出 (C4)
            "Branch_CNN": model.cnn.layer4,
            # ViT 分支：vit_down 输出 (与 C4 对齐)
            "Branch_ViT": model.vit_down,
            # 瓶颈层：ASPP 输出（全局上下文）
            "Bottleneck_ASPP": model.aspp,
            # 解码器最后两层上采样块
            "Decoder_Dec3": model.dec3,
            "Decoder_Dec2": model.dec2,
            # 预测输出前一层：final_conv 中的 ReLU 输出 (32 通道)
            "Penultimate": model.final_conv[2],
        }
        grad_cam = GradCAM(model, target_layers)

        # 计算并记录模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(
            {
                "model_total_params": total_params,
                "model_trainable_params": trainable_params
            }
        )

        # 优化器和学习率调度器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config["warmup_epochs"],
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["num_epochs"] - config["warmup_epochs"],
            eta_min=config["min_lr"],
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],#两个阶段
            milestones=[config["warmup_epochs"]],
        )
        
        # 创建检查点目录
        checkpoint_dir = os.path.join("/home/suyubaek/lancing/checkpoints", experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 本地日志配置
        local_log_path = os.path.join(checkpoint_dir, "train_log.txt")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(local_log_path, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

        logging.info(f"实验开始: {experiment_name}")
        logging.info(
            f"模型: {config['model_name']}, 总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}"
        )
        start_time = datetime.now()
        send_message(
            title=f"实验开始: {experiment_name}",
            content=(
                f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M')}\n"
                f"模型: {config['model_name']}\n"
                f"总参数: {total_params:,}\n"
                f"可训练参数: {trainable_params:,}\n"
                f"训练轮数: {config['num_epochs']}\n"
                f"学习率: {config['learning_rate']}\n"
            )
        )

        # 训练循环
        best_iou, report_iou = 0.0, 0.0
        best_epoch = -1
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        loss_fc = MixedLoss(alpha=0.4, beta=0.6, weight_bce=0.5, weight_tversky=0.5).to(device)#loss的实例化
        epoch_state = {"prev_bottleneck": None}

        for epoch in range(config["num_epochs"]):
            
            train_loss, train_seg_loss, train_bce_loss, train_tversky_loss, \
            train_iou, train_precision, train_recall, train_f1 = train_one_epoch(
                model, train_loader, optimizer, loss_fc, device, epoch + 1,
                train_loader.dataset, grad_cam, epoch_state
            )

            val_loss, val_seg_loss, val_bce_loss, val_tversky_loss, \
            val_iou, val_precision, val_recall, val_f1 = validate(
                model, val_loader, loss_fc, device, epoch + 1
            )

            scheduler.step()

            wandb.log(
                {   "Epoch": epoch + 1,
                    "Comparison Board/IoU": val_iou,
                    "Comparison Board/F1": val_f1,
                    "Comparison Board/Precision": val_precision,
                    "Comparison Board/Recall": val_recall,

                    "Train_info/Loss/Train": train_loss,
                    "Train_info/Loss/Val": val_loss,
                    #"Train_info/Seg_Loss/Train": train_seg_loss,
                    #"Train_info/Seg_Loss/Val": val_seg_loss,
                    "Train_info/BCE_Loss/Train": train_bce_loss,
                    "Train_info/BCE_Loss/Val": val_bce_loss,
                    "Train_info/Tversky_Loss/Train": train_tversky_loss,
                    "Train_info/Tversky_Loss/Val": val_tversky_loss,
                    "Train_info/IoU/Train": train_iou,
                    "Train_info/IoU/Val": val_iou,
                    "Train_info/F1/Train": train_f1,
                    "Train_info/F1/Val": val_f1,

                    "Train_info/Learning_Rate": optimizer.param_groups[0]["lr"]
                }
            )

            logging.info(
                f"epoch: {epoch+1}, "
                f"train_iou: {train_iou:.4f}, val_iou: {val_iou:.4f}, "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.6g}"
            )

            # 每10个epoch保存一次可视化结果
            if (epoch + 1) % 10 == 0:
                figure = create_sample_images(model, vis_loader, device, epoch + 1, num_samples=12)
                if figure:
                    wandb.log({"Prediction_Summary": wandb.Image(figure)})
                    plt.close(figure)

            if val_iou > best_iou:
                if val_iou > 0.7 and val_iou - report_iou > 0.01:
                    report_iou = val_iou
                    elapsed = (datetime.now() - start_time).total_seconds()
                    eta_seconds = (elapsed / (epoch + 1)) * (config["num_epochs"] - (epoch + 1))
                    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                    send_message(
                        title=f"{experiment_name}：模型最佳指标更新",
                        content=(
                            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                            f"Epoch: {best_epoch}\n"
                            f"Val IoU: {report_iou:.4f}\n"
                            f"Cur lr: {optimizer.param_groups[0]['lr']:.6g}\n"
                            f"预计结束时间: {eta_time.strftime('%Y-%m-%d %H:%M')}"
                        ),
                    )
                best_iou = val_iou
                best_epoch = epoch + 1
                save_checkpoint(model, optimizer, epoch, best_iou, best_model_path)
                logging.info(f"New best model saved at epoch {best_epoch} with IoU: {best_iou:.4f}")

            if (epoch + 1) % 10 == 0 and epoch > 100:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
                )
                save_checkpoint(model, optimizer, epoch, best_iou, checkpoint_path)

        wandb.summary["best_iou"] = best_iou
        wandb.summary["best_epoch"] = best_epoch

        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logging.info("训练成功，开始清理临时检查点...")
            for filename in os.listdir(checkpoint_dir):
                if filename.startswith("checkpoint_epoch_") and filename.endswith(".pth"):
                    file_to_delete = os.path.join(checkpoint_dir, filename)
                    try:
                        os.remove(file_to_delete)
                        logging.info(f"已删除临时检查点: {file_to_delete}")
                    except OSError as e:
                        logging.error(f"删除文件 {file_to_delete} 时出错: {e}")
        
        # Test Process
        logging.info("开始测试...")
        test_loss, test_seg_loss, test_iou, test_precision, test_recall, test_f1 = test(
            model, test_loader, loss_fc, device
        )
        logging.info(f"Test Results - IoU: {test_iou:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        wandb.log({
            "Test/IoU": test_iou,
            "Test/F1": test_f1,
            "Test/Precision": test_precision,
            "Test/Recall": test_recall,
            "Test/Loss": test_loss
        })

        send_message(
            title=f"实验结束: {experiment_name}",
            content=f"训练完成!\n最佳 Val IoU: {best_iou:.4f} (at epoch {best_epoch})\nTest IoU: {test_iou:.4f}",
        )

    except Exception as exc:
        logging.error(f"An error occurred: {exc}", exc_info=True)
        send_message(
            title=f"实验失败: {experiment_name}",
            content=f"实验运行时发生错误: \n{exc}",
        )
        raise

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()