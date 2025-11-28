import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# 1. DSCFE Module (深度可分离卷积特征提取)
# 对应 Figure 2 
# ==========================================
class DSCFE(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(DSCFE, self).__init__()
        
        # 3x3 Depthwise Conv
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous Convolution (with dilation)
        # 注意：论文 Figure 2 中 Atrous Conv 在 Depthwise 之后
        # 为了保持维度一致，padding 设为 dilation_rate
        self.atrous = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Pointwise Conv (1x1)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.atrous(x)
        x = self.pointwise(x)
        return x

# ==========================================
# 2. DAPP Module (深度空洞金字塔池化)
# 对应 Section 3.3 [cite: 1805-1807]
# ==========================================
class DAPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(DAPP, self).__init__()
        
        # Branch 1: 1x1 Conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2, 3, 4: DSCFE with dilation rates 1, 3, 5 
        self.branch2 = DSCFE(in_channels, out_channels, dilation_rate=1)
        self.branch3 = DSCFE(in_channels, out_channels, dilation_rate=3)
        self.branch4 = DSCFE(in_channels, out_channels, dilation_rate=5)
        
        # Branch 5: Global Average Pooling (Image Pooling)
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final Fusion: 1x1 Conv to reduce channels
        # Concatenation of 5 branches (out_channels * 5) -> out_channels
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) # 论文提到有 Dropout [cite: 1807]
        )

    def forward(self, x):
        size = x.shape[2:]
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        b5 = self.branch5(x)
        b5 = F.interpolate(b5, size=size, mode='bilinear', align_corners=True)
        
        # Stack outputs [cite: 1807]
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        out = self.conv_out(out)
        return out

# ==========================================
# 3. CSAM Module (通道空间注意力模块)
# 对应 Figure 3 和 Section 3.4 [cite: 1839]
# ==========================================
class CSAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CSAM, self).__init__()
        
        # --- Channel Attention ---
        # F_tr: Convolution operator [cite: 1840]
        self.f_tr = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        
        # MLP for Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio), # FC 
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels), # FC restore [cite: 1851]
            nn.Sigmoid()
        )
        
        # --- Spatial Attention ---
        # 7x7 Convolution [cite: 1855]
        # Input is 2 channels (Max Pool + Avg Pool results)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Transform
        u = self.f_tr(x) # [B, C, H, W]
        b, c, h, w = u.size()
        
        # 2. Channel Attention
        # Global Average Pooling [cite: 1845]
        u_flat = F.adaptive_avg_pool2d(u, 1).view(b, c)
        channel_weights = self.mlp(u_flat).view(b, c, 1, 1) # s [cite: 1852]
        
        # Scale operation [cite: 1853]
        u_scaled = u * channel_weights 
        
        # 3. Spatial Attention
        # MaxPool & AvgPool across channels [cite: 1854]
        max_pool, _ = torch.max(u_scaled, dim=1, keepdim=True) # [B, 1, H, W]
        avg_pool = torch.mean(u_scaled, dim=1, keepdim=True)   # [B, 1, H, W]
        
        # Concat [cite: 1854]
        spatial_input = torch.cat([avg_pool, max_pool], dim=1) # [B, 2, H, W]
        
        # 7x7 Conv + Sigmoid [cite: 1859]
        spatial_weights = self.sigmoid(self.spatial_conv(spatial_input))
        
        # Final Output (Applying spatial weights)
        out = u_scaled * spatial_weights
        return out

# ==========================================
# 4. MADF-Net 主网络
# 对应 Figure 1 [cite: 1739]
# ==========================================
class Model(nn.Module):
    def __init__(self, in_channels=2, num_classes=1):
        super(Model, self).__init__()
        
        # 修改这里：使用 resnet50 或 resnet18
        # resnet = models.resnet101(...) 
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) 
        
        # 修改第一层卷积以适应 2 通道输入 (你的数据是 2*256*256)
        # 原始 ResNet 输入是 3 通道
        if in_channels != 3:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        else:
            self.stem = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        
        # Extract layers
        self.layer1 = resnet.layer1 # Low-level features (256 channels)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4 # High-level features (2048 channels)
        
        # --- Encoder: DAPP Module ---
        # 输入为 layer4 输出 (2048 channels)
        self.dapp = DAPP(in_channels=2048, out_channels=256)
        
        # 1x1 Conv after DAPP (如图1所示，DAPP后有一个1x1 Conv减少通道或调整特征)
        self.encoder_out_conv = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # --- Decoder: CSAM Module ---
        # 处理 Low-level features (layer1 输出, 256 channels)
        self.csam = CSAM(in_channels=256)
        
        # 1x1 Conv for Low-level features (Standard DeepLab practice, implied in Figure 1 "1x1 Conv" block)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False), # 降维通常设为 48 或 256，这里设为 48 以平衡高低层特征
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # --- Decoder: Fusion & Final ---
        # Concat: High-level (256) + Low-level (48) = 304
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False), # 3x3 Conv [cite: 1755]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1) # Final output
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        
        # --- Encoder ---
        # Backbone
        x = self.stem(x)      # Stem
        c1 = self.layer1(x)   # Low-level features (for CSAM)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)  # High-level features
        
        # DAPP Processing 
        high_level = self.dapp(c4)
        high_level = self.encoder_out_conv(high_level)
        
        # Upsample High-level features to match Low-level features size
        high_level = F.interpolate(high_level, size=c1.shape[2:], mode='bilinear', align_corners=True)
        
        # --- Decoder ---
        # CSAM Processing on Low-level features [cite: 1737]
        low_level = self.csam(c1)
        low_level = self.low_level_conv(low_level)
        
        # Concatenation 
        # High-level semantic info + Low-level edge features
        fused = torch.cat([high_level, low_level], dim=1)
        
        # Final Convolution and Upsampling to original size
        out = self.decoder_conv(fused)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=True)
        
        return out

# ==========================================
# 5. 测试代码 (针对 2*256*256 输入)
# ==========================================
if __name__ == "__main__":
    # 模拟输入数据: Batch Size=4, Channels=2, Height=256, Width=256
    dummy_input = torch.randn(4, 2, 256, 256)
    
    # 实例化模型
    model = Model(in_channels=2, num_classes=1)
    
    # 检查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # 应为 [4, 1, 256, 256]
    
    # 简单的参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")