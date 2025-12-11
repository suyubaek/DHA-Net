import os
import numpy as np
import rasterio
import pandas as pd
from sklearn.metrics import confusion_matrix

def calculate_metrics(pred_path, gt_path):
    """
    计算单张预测图与 GT 的 IoU, Precision, Recall
    """
    # 1. 读取数据
    with rasterio.open(pred_path) as src_pred:
        pred = src_pred.read(1)
        nodata_pred = src_pred.nodata

    with rasterio.open(gt_path) as src_gt:
        gt = src_gt.read(1)
        nodata_gt = src_gt.nodata

    # 2. 形状检查
    if pred.shape != gt.shape:
        print(f"Warning: Shape mismatch {pred.shape} vs {gt.shape}. Skipping.")
        return None

    # 3. 数据预处理 (展平)
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    # 4. 掩膜处理 (排除 NoData 区域)
    # 只有当 GT 和 Pred 都不是 NoData 时，才参与计算
    # 注意：根据之前的逻辑，NoData 可能是 254 或 NaN
    
    # 创建有效掩膜
    if np.isnan(nodata_gt):
        valid_mask = ~np.isnan(gt_flat)
    else:
        valid_mask = (gt_flat != nodata_gt)
        
    if nodata_pred is not None:
        if np.isnan(nodata_pred):
            valid_mask &= ~np.isnan(pred_flat)
        else:
            valid_mask &= (pred_flat != nodata_pred)

    # 提取有效像素
    y_true = gt_flat[valid_mask]
    y_pred = pred_flat[valid_mask]

    # 5. 二值化处理
    # 假设水体是 255，背景是 0。我们需要转成 1 和 0
    # 如果已经是 0/1 则不变，如果是 0/255 则转为 0/1
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    # 6. 计算混淆矩阵
    # tn, fp, fn, tp
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        # 如果只有一类数据，confusion_matrix 可能会报错或返回形状不对
        return None

    # 7. 计算指标
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # IoU = TP / (TP + FP + FN)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }

def main():
    # 配置路径
    base_dir = "/home/rove/lancing/lancang_infer/clip"
    gt_filename = "watermask2412_clip.tif"
    gt_path = os.path.join(base_dir, gt_filename)

    if not os.path.exists(gt_path):
        print(f"Error: Ground Truth file not found: {gt_path}")
        return

    print(f"Ground Truth: {gt_filename}")
    print("-" * 60)
    print(f"{'Filename':<40} | {'IoU':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<8}")
    print("-" * 60)

    results = []

    # 遍历文件夹
    files = sorted(os.listdir(base_dir))
    for f in files:
        # 过滤条件
        if not f.endswith(".tif"):
            continue
        if f == gt_filename: # 跳过 GT 本身
            continue
        
        pred_path = os.path.join(base_dir, f)
        
        # 计算指标
        metrics = calculate_metrics(pred_path, gt_path)
        
        if metrics:
            print(f"{f:<40} | {metrics['IoU']:.4f}   | {metrics['Precision']:.4f}     | {metrics['Recall']:.4f}   | {metrics['F1']:.4f}")
            
            # 记录结果
            res_dict = {"Filename": f}
            res_dict.update(metrics)
            results.append(res_dict)

    # 保存结果到 CSV
    if results:
        df = pd.DataFrame(results)
        # 调整列顺序
        cols = ["Filename", "IoU", "Precision", "Recall", "F1", "TP", "FP", "FN"]
        df = df[cols]
        
        save_csv_path = os.path.join(base_dir, "evaluation_results.csv")
        df.to_csv(save_csv_path, index=False)
        print("-" * 60)
        print(f"Results saved to: {save_csv_path}")
    else:
        print("No valid prediction files found.")

if __name__ == "__main__":
    main()

