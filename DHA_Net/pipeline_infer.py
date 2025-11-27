import torch
import numpy as np
import cv2
from tqdm import tqdm

def predict_sliding_window(model, image_path, patch_size=256, stride=128, num_classes=2):
    """
    使用高斯加权滑动窗口对大图进行推理
    
    Args:
        model: 已加载权重的 PyTorch 模型
        image_path: 大图路径
        patch_size: 训练时的 patch 大小 (256)
        stride: 滑动步长 (通常为 patch_size 的 1/2，即 128)
        num_classes: 分割类别数
    """
    # 1. 读取图像
    # 假设输入是 (H, W, C)，例如 (2432, 2252, 3)
    image = cv2.imread(image_path)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转 RGB
    h, w, c = image.shape
    
    # 2. 生成高斯权重掩膜 (256, 256)
    def get_gaussian_mask(size, sigma_scale=1.0/8):
        tmp = np.zeros((size, size))
        center = size // 2
        sig = size * sigma_scale
        y, x = np.ogrid[-center:size-center, -center:size-center]
        mask = np.exp(-(x**2 + y**2) / (2 * sig**2))
        return mask

    gaussian_mask = get_gaussian_mask(patch_size)
    # 将 mask 转为 tensor 方便后续计算，形状 (1, 1, H, W)
    gaussian_mask_tensor = torch.from_numpy(gaussian_mask).float().cuda() 

    # 3. 初始化结果累加器
    # prob_map: 存储加权后的预测概率 (num_classes, H, W)
    # weight_map: 存储每个像素点的权重累加值 (H, W)
    # 为了防止边缘溢出，我们先对图像进行 Padding
    pad_h = stride - (h % stride) if h % stride != 0 else 0
    pad_w = stride - (w % stride) if w % stride != 0 else 0
    
    # 这里的 padding 策略是 Reflect 或 Constant 均可
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    h_pad, w_pad, _ = image_padded.shape
    
    prob_map = torch.zeros((num_classes, h_pad, w_pad)).cuda()
    weight_map = torch.zeros((h_pad, w_pad)).cuda()

    model.eval()
    with torch.no_grad():
        # 4. 滑动窗口循环
        for r in tqdm(range(0, h_pad - patch_size + 1, stride)):
            for c in range(0, w_pad - patch_size + 1, stride):
                # 截取 Patch
                patch = image_padded[r:r+patch_size, c:c+patch_size, :]
                
                # 预处理 (归一化、转 Tensor、维度变换)
                # 假设模型输入需要 (1, C, H, W) 且归一化到 [0, 1] 或标准化
                patch_tensor = torch.from_numpy(patch).float() / 255.0 
                patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0).cuda() # (1, 3, 256, 256)
                
                # 模型推理
                output = model(patch_tensor) 
                # output 形状通常是 (1, num_classes, 256, 256)
                # 如果输出是 logits，需要 softmax
                output = torch.softmax(output, dim=1)
                
                # 取出预测结果 (num_classes, 256, 256)
                pred_patch = output.squeeze(0)
                
                # 加权累加
                prob_map[:, r:r+patch_size, c:c+patch_size] += pred_patch * gaussian_mask_tensor
                weight_map[r:r+patch_size, c:c+patch_size] += gaussian_mask_tensor

    # 5. 归一化与裁剪
    # 避免除以 0 (虽然理论上覆盖完全不会有0，但为了安全)
    weight_map = torch.where(weight_map == 0, torch.ones_like(weight_map), weight_map)
    
    prob_map /= weight_map # (num_classes, H_pad, W_pad)
    
    # 裁剪回原始大小
    prob_map = prob_map[:, :h, :w]
    
    # 6. 生成最终 Mask
    # 取最大概率的索引作为类别
    result_mask = torch.argmax(prob_map, dim=0).cpu().numpy().astype(np.uint8)
    
    return result_mask

# 使用示例 (伪代码)
# model = MySegmentationModel().cuda()
# model.load_state_dict(...)
# result = predict_sliding_window(model, "path/to/large_image.tif")
# cv2.imwrite("result.png", result * 255) # 假设二分类，乘以255可视化