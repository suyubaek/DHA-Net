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
from loss_function import CombinedLoss, DiceBCELoss
from dataprocess import get_loaders
from config import config
from metrics import calculate_metrics
from message2lark import send_message
from visualization import create_sample_images

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    total_loss = seg_loss_total = align_loss_total = cls_loss_total = 0.0
    iou, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images, labels, class_labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        seg_logits, align_emb, cls_logits = model(images)
        loss, loss_components = loss_fc(
            seg_logits=seg_logits,
            align_emb=align_emb,
            cls_logits=cls_logits,
            seg_labels=labels,
            align_labels=class_labels,
            cls_labels=class_labels,
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        seg_loss_total += loss_components["seg"].item()
        align_loss_total += loss_components["align"].item()
        cls_loss_total += loss_components["cls"].item()

        outputs = torch.sigmoid(seg_logits).detach().cpu()
        labels = labels.cpu()
        cur_iou, cur_precision, cur_recall, cur_f1 = calculate_metrics(outputs, labels)
        iou += cur_iou
        precision += cur_precision
        recall += cur_recall
        f1 += cur_f1

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Seg_L": f"{loss_components['seg']:.4f}",
                "Align_L": f"{loss_components['align']:.4f}",
                "Cls_L": f"{loss_components['cls']:.4f}",
            }
        )
    
    avg_loss = total_loss / len(train_loader)
    avg_seg_loss = seg_loss_total / len(train_loader)
    avg_align_loss = align_loss_total / len(train_loader)
    avg_cls_loss = cls_loss_total / len(train_loader)
    iou /= len(train_loader)
    precision /= len(train_loader)
    recall /= len(train_loader)
    f1 /= len(train_loader)

    return (
        avg_loss,
        avg_seg_loss,
        avg_align_loss,
        avg_cls_loss,
        iou,
        precision,
        recall,
        f1,
    )

def validate(model, val_loader, loss_fc, device, epoch):
    model.eval()
    total_loss = seg_loss_total = align_loss_total = cls_loss_total = 0.0
    iou, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images, labels, class_labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            seg_logits, align_emb, cls_logits = model(images) 
            loss, loss_components = loss_fc(
                seg_logits=seg_logits,
                align_emb=align_emb,
                cls_logits=cls_logits,
                seg_labels=labels,
                align_labels=class_labels,
                cls_labels=class_labels,
            )

            total_loss += loss.item()
            seg_loss_total += loss_components["seg"].item()
            align_loss_total += loss_components["align"].item()
            cls_loss_total += loss_components["cls"].item()

            outputs = torch.sigmoid(seg_logits).detach().cpu()
            labels = labels.cpu()
            cur_iou, cur_precision, cur_recall, cur_f1 = calculate_metrics(outputs, labels)
            iou += cur_iou
            precision += cur_precision
            recall += cur_recall
            f1 += cur_f1

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Seg_L": f"{loss_components['seg']:.4f}",
                    "Align_L": f"{loss_components['align']:.4f}",
                    "Cls_L": f"{loss_components['cls']:.4f}",
                }
            )
    avg_loss = total_loss / len(val_loader)
    avg_seg_loss = seg_loss_total / len(val_loader)
    avg_align_loss = align_loss_total / len(val_loader)
    avg_cls_loss = cls_loss_total / len(val_loader)
    iou /= len(val_loader)
    precision /= len(val_loader)
    recall /= len(val_loader)
    f1 /= len(val_loader)

    return (
        avg_loss,
        avg_seg_loss,
        avg_align_loss,
        avg_cls_loss,
        iou,
        precision,
        recall,
        f1,
    )


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
        tags=["HTCC"],
    )

    try:
        train_loader, val_loader = get_loaders(
            data_dir=config["data_root"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            neg_sample_ratio=0.3,
            seed=config["seed"],
        )

        # 模型初始化
        model = Model(
            img_size=config.get('image_size', 256),
            patch_size=config.get('patch_size', 16),
            in_chans=config['in_channels'],
            num_classes=config['num_classes'],
            embed_dim=config.get('embed_dim', 768),
            depth=config.get('depth', 6),
            num_heads=config.get('num_heads', 12),
            mlp_ratio=config.get('mlp_ratio', 4.0)
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
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config["warmup_epochs"]],
        )
        
        # 创建检查点目录
        checkpoint_dir = os.path.join("/home/rove/ThesisExp2026/checkpoints", experiment_name)
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
        loss_fc = CombinedLoss(
            align_lambda=config.get("align_lambda", 0.1),
            cls_lambda=config.get("cls_lambda", 0.1),
            con_temperature=config.get("con_temperature", 0.07),
        ).to(device)

        for epoch in range(config["num_epochs"]):
            
            train_loss, train_seg_loss, train_align_loss, train_cls_loss, \
            train_iou, train_precision, train_recall, train_f1 = train_one_epoch(
                model, train_loader, optimizer, loss_fc, device, epoch + 1
            )

            val_loss, val_seg_loss, val_align_loss, val_cls_loss, \
            val_iou, val_precision, val_recall, val_f1 = validate(
                model, val_loader, loss_fc, device, epoch + 1
            )

            scheduler.step()

            wandb.log(
                {   
                    "Comparison Board/IoU": val_iou,
                    "Comparison Board/F1": val_f1,
                    "Comparison Board/Precision": val_precision,
                    "Comparison Board/Recall": val_recall,

                    "Train_info/Loss/Train": train_loss,
                    "Train_info/Loss/Val": val_loss,
                    "Train_info/Seg_Loss/Train": train_seg_loss,
                    "Train_info/Seg_Loss/Val": val_seg_loss,
                    "Train_info/Align_Loss/Train": train_align_loss,
                    "Train_info/Align_Loss/Val": val_align_loss,
                    "Train_info/Cls_Loss/Train": train_cls_loss,
                    "Train_info/Cls_Loss/Val": val_cls_loss,
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
            # if (epoch + 1) % 10 == 0:
            #     figure = create_sample_images(model, vis_loader, device, epoch + 1, num_samples=len(vis_loader))
            #     if figure:
            #         wandb.log({"Prediction_Summary": wandb.Image(figure)})
            #         plt.close(figure)

            if val_iou > best_iou:
                if val_iou > 0.84 and val_iou - report_iou > 0.01:
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

        send_message(
            title=f"实验结束: {experiment_name}",
            content=f"训练完成!\n最佳 Val IoU: {best_iou:.4f} (at epoch {best_epoch})",
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