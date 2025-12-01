import os
import sys
from datetime import datetime
from datetime import timedelta
import random
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

from model import Model
from loss_function import DiceBCELoss
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

def train_one_epoch(model, train_loader, optimizer, loss_fc, device, epoch):
    model.train()
    total_loss = seg_loss_total = 0.0
    iou, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    metric_count = 0
    
    # Initialize GPU Augmentor
    augmentor = GPUAugmentor(rotate_prob=0.5, noise_prob=0.5)
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images, labels, class_labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)

        # Apply GPU Augmentation
        with torch.no_grad():
            images, labels = augmentor(images, labels)

        optimizer.zero_grad()
        
        seg_logits = model(images)
        if isinstance(seg_logits, list):
            # Deep supervision case
            loss = 0
            for logits in seg_logits:
                loss += loss_fc(logits, labels)
            loss /= len(seg_logits)
            seg_logits = seg_logits[0] # Use the first output for metrics
        else:
            loss = loss_fc(seg_logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        seg_loss_total += loss.item()
        
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
    
    if metric_count > 0:
        iou /= metric_count
        precision /= metric_count
        recall /= metric_count
        f1 /= metric_count
    
    return (
        avg_loss,
        avg_seg_loss,
        iou,
        precision,
        recall,
        f1,
    )

def validate(model, val_loader, loss_fc, device, epoch):
    model.eval()
    total_loss = seg_loss_total = 0.0
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
    avg_loss = total_loss / len(val_loader)
    avg_seg_loss = seg_loss_total / len(val_loader)
    iou /= len(val_loader)
    precision /= len(val_loader)
    recall /= len(val_loader)
    f1 /= len(val_loader)

    return (
        avg_loss,
        avg_seg_loss,
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


def train_cls_one_epoch(model, train_loader, optimizer, loss_fc, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Stage 1 (CLS) Training Epoch {epoch}")
    for batch in pbar:
        images, _, cls_labels = batch # Ignore masks
        images = images.to(device, non_blocking=True)
        
        # Convert labels: 0 -> 0 (No Water), 1/2 -> 1 (Water)
        targets = (cls_labels > 0).float().unsqueeze(1).to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        logits = model(images, mode='cls')
        loss = loss_fc(logits, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix({"CLS Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.4f}"})
        
    return total_loss / len(train_loader), correct / total

def validate_cls(model, val_loader, loss_fc, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Stage 1 (CLS) Validation Epoch {epoch}")
        for batch in pbar:
            images, _, cls_labels = batch
            images = images.to(device, non_blocking=True)
            targets = (cls_labels > 0).float().unsqueeze(1).to(device, non_blocking=True)
            
            logits = model(images, mode='cls')
            loss = loss_fc(logits, targets)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({"Val CLS Loss": f"{loss.item():.4f}", "Val Acc": f"{correct/total:.4f}"})
            
    return total_loss / len(val_loader), correct / total

def main():
    # 设置随机种子
    set_seed(config["seed"])
    # 设置计算设备
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # W&B实验看板初始化
    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    wandb.init(
        project="LanCang River",
        name=experiment_name,
        config=config,
        tags=["DHA", "CoTrain"],
    )

    try:
        train_loader, val_loader, test_loader = get_loaders(
            data_dir=config["data_root"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            neg_sample_ratio=0.3,
            seed=config["seed"],
        )
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

        # 计算并记录模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(
            {
                "model_total_params": total_params,
                "model_trainable_params": trainable_params
            }
        )
        
        # 创建检查点目录
        checkpoint_dir = os.path.join("/home/rove/lancing/checkpoints", experiment_name)
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
        start_time = datetime.now()
        
        # --- Stage 1: Classification Training ---
        logging.info(">>> Starting Stage 1: Classification Training")
        cls_optimizer = optim.AdamW(model.parameters(), lr=config['cls_lr'], weight_decay=config['weight_decay'])
        cls_loss_fc = torch.nn.BCEWithLogitsLoss().to(device)
        
        best_cls_acc = 0.0
        
        for epoch in range(config['cls_epochs']):
            train_loss, train_acc = train_cls_one_epoch(model, train_loader, cls_optimizer, cls_loss_fc, device, epoch + 1)
            val_loss, val_acc = validate_cls(model, val_loader, cls_loss_fc, device, epoch + 1)
            
            wandb.log({
                "Stage1/Train_Loss": train_loss, "Stage1/Train_Acc": train_acc,
                "Stage1/Val_Loss": val_loss, "Stage1/Val_Acc": val_acc,
                "Stage1/Epoch": epoch + 1
            })
            
            logging.info(f"Stage 1 Epoch {epoch+1}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")
            
            if val_acc > best_cls_acc:
                best_cls_acc = val_acc
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_cls_model.pth"))
        
        logging.info(f"Stage 1 Finished. Best Acc: {best_cls_acc:.4f}")
        
        # Load best classification weights
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_cls_model.pth")))
        
        # --- Stage 2: Segmentation Training ---
        logging.info(">>> Starting Stage 2: Segmentation Training")
        
        # Optimizer for Stage 2
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        
        # Schedulers
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=config["warmup_epochs"])
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config["seg_epochs"] - config["warmup_epochs"], eta_min=config["min_lr"])
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config["warmup_epochs"]])
        
        loss_fc = DiceBCELoss(dice_weight=0.5, bce_weight=0.5).to(device)
        
        best_iou, report_iou = 0.0, 0.0
        best_epoch = -1
        best_model_path = os.path.join(checkpoint_dir, "best_seg_model.pth")
        
        for epoch in range(config["seg_epochs"]):
            current_epoch = epoch + 1
            
            train_loss, train_seg_loss, train_iou, train_precision, train_recall, train_f1 = train_one_epoch(
                model, train_loader, optimizer, loss_fc, device, current_epoch
            )

            val_loss, val_seg_loss, val_iou, val_precision, val_recall, val_f1 = validate(
                model, val_loader, loss_fc, device, current_epoch
            )

            scheduler.step()

            wandb.log(
                {   "Stage2/Epoch": current_epoch,
                    "Comparison Board/IoU": val_iou,
                    "Comparison Board/F1": val_f1,
                    "Comparison Board/Precision": val_precision,
                    "Comparison Board/Recall": val_recall,

                    "Train_info/Loss/Train": train_loss,
                    "Train_info/Loss/Val": val_loss,
                    "Train_info/Seg_Loss/Train": train_seg_loss,
                    "Train_info/Seg_Loss/Val": val_seg_loss,
                    "Train_info/IoU/Train": train_iou,
                    "Train_info/IoU/Val": val_iou,
                    "Train_info/F1/Train": train_f1,
                    "Train_info/F1/Val": val_f1,

                    "Train_info/Learning_Rate": optimizer.param_groups[0]["lr"]
                }
            )

            logging.info(
                f"Stage 2 Epoch: {current_epoch}, "
                f"train_iou: {train_iou:.4f}, val_iou: {val_iou:.4f}, "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.6g}"
            )

            # Visualization
            if current_epoch % 10 == 0:
                figure = create_sample_images(model, vis_loader, device, current_epoch, num_samples=12)
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
                best_epoch = current_epoch
                save_checkpoint(model, optimizer, current_epoch, best_iou, best_model_path)
                logging.info(f"New best seg model saved at epoch {best_epoch} with IoU: {best_iou:.4f}")

            if current_epoch % 10 == 0 and current_epoch > 100:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{current_epoch}.pth"
                )
                save_checkpoint(model, optimizer, current_epoch, best_iou, checkpoint_path)
                

        wandb.summary["best_iou"] = best_iou
        wandb.summary["best_epoch"] = best_epoch
        
        # Test Process
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
            content=f"训练完成!\n最佳 Val IoU: {best_iou:.4f}\nTest IoU: {test_iou:.4f}",
        )

    except Exception as exc:
        logging.error(f"An error occurred: {exc}", exc_info=True)
        send_message(title=f"实验失败: {experiment_name}", content=f"错误: \n{exc}")
        raise

    finally:
        wandb.finish()

if __name__ == "__main__":
    main()