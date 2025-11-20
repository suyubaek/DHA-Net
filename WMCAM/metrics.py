import torch

def calculate_metrics(pred_img, mask, threshold=0.5):
    pred_binary = (pred_img > threshold).float()
    mask = mask.float()

    pred_flat = pred_binary.view(-1)
    mask_flat = mask.view(-1)

    tp = (pred_flat * mask_flat).sum()  # 真阳性：预测为1，实际也为1
    fp = (pred_flat * (1 - mask_flat)).sum()  # 假阳性：预测为1，实际为0
    fn = ((1 - pred_flat) * mask_flat).sum()  # 假阴性：预测为0，实际为1

    epsilon = 1e-6

    # 4. 计算各项指标
    # IoU (交并比): TP / (TP + FP + FN)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)

    # Precision (精确率): TP / (TP + FP)
    precision = (tp + epsilon) / (tp + fp + epsilon)

    # Recall (召回率): TP / (TP + FN)
    recall = (tp + epsilon) / (tp + fn + epsilon)

    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return iou.item(), precision.item(), recall.item(), f1.item()


if __name__ == '__main__':
    pred_logits = torch.randn(4, 1, 256, 256)
    true_mask = torch.randint(0, 2, (4, 1, 256, 256))

    iou, precision, recall, f1 = calculate_metrics(pred_logits, true_mask, threshold=0.0)
    print("Metrics from logits (threshold=0.0):")
    print(metrics_from_logits)

    pred_probs = torch.sigmoid(pred_logits)
    # 对于 probabilities，我们通常使用 0.5 作为阈值
    metrics_from_probs = calculate_metrics(pred_probs, true_mask, threshold=0.5)
    print("\nMetrics from probabilities (threshold=0.5):")
    print(metrics_from_probs)