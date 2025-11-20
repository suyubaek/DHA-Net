import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class PAM_Module(nn.Module):
    """
    Position Attention Module (PAM) - 对应论文 Eq (1)(2) 和 Figure 6
    """
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    """
    Channel Attention Module (CAM) - 对应论文 Figure 4 中的 CAM 模块
    """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class ASPP(nn.Module):
    """
    ASPP 模块：对应论文设定，膨胀率为 1, 2, 4, 8
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # Rate 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Rate 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Rate 4
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Rate 8
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Global Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x5 = self.avg_pool(x)
        x5 = self.conv_pool(x5)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv_out(x)
        return x

class Model(nn.Module):
    """
    WaterDetectionNet (WDNet) 使用 SMP Encoder
    """
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, num_classes=1):
        super(Model, self).__init__()
        
        # --- 1. SMP Encoder (Backbone) ---
        # 论文使用的是 'xception'。
        # 你可以将 encoder_name 改为 'xception' (需要安装 timm) 或 'resnet50'。
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )
        
        # 获取 Encoder 输出的通道数
        # ResNet50 stages: (3, 64, 256, 512, 1024, 2048)
        # Low-level 通常取 stride=4 (stage 2, index 2) -> 256 channels
        # High-level 通常取 stride=32 (stage 5, index 5) -> 2048 channels
        encoder_channels = self.encoder.out_channels
        low_level_in_channels = encoder_channels[2] # Stride 4
        high_level_in_channels = encoder_channels[-1] # Stride 32
        
        # --- 2. ASPP ---
        # 论文: High-level features 输入 ASPP
        self.aspp = ASPP(in_channels=high_level_in_channels, out_channels=256)
        
        # --- 3. Decoder Branches (Based on Figure 4) ---
        
        # 左侧分支 (Low-level): 论文图示包含 CAM -> Conv 1x1
        self.low_level_cam = CAM_Module(low_level_in_channels)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_in_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 融合后的解码路径: 
        # Concat -> Conv 3x3 -> CAM -> Conv 3x3 -> PAM -> Output
        
        # Concat Channels: ASPP(256) + Low-level(48) = 304
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_cam = CAM_Module(256)
        
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_pam = PAM_Module(256)
        
        # 最终输出
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # --- Encoding (SMP) ---
        # features 是一个列表，包含不同 stride 的特征图
        features = self.encoder(x)
        
        # Extract Low-level (Stride 4) and High-level (Stride 32)
        low_level_feat = features[2] 
        high_level_feat = features[-1]
        
        # --- High-level Processing ---
        # 论文: ASPP -> Conv 1x1 (in ASPP) -> Upsample
        x_high = self.aspp(high_level_feat)
        x_high = F.interpolate(x_high, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        
        # --- Low-level Processing ---
        # 论文: CAM -> Conv 1x1
        x_low = self.low_level_cam(low_level_feat)
        x_low = self.low_level_conv(x_low)
        
        # --- Decoding & Attention ---
        # 1. Concat
        x_fused = torch.cat([x_high, x_low], dim=1)
        
        # 2. Conv 3x3
        x_fused = self.decoder_conv1(x_fused)
        
        # 3. CAM (Channel Attention)
        x_fused = self.decoder_cam(x_fused)
        
        # 4. Conv 3x3
        x_fused = self.decoder_conv2(x_fused)
        
        # 5. PAM (Spatial Attention)
        x_fused = self.decoder_pam(x_fused)
        
        # 6. Final Conv & Upsample to input size
        x_out = self.final_conv(x_fused)
        x_out = F.interpolate(x_out, size=input_shape, mode='bilinear', align_corners=True)
        
        return x_out

# --- 测试代码 ---
if __name__ == "__main__":
    # 1. 定义输入: Batch=2, Channels=2, H=256, W=256
    dummy_input = torch.randn(2, 2, 256, 256)
    
    # 2. 初始化模型 (使用 ResNet50 作为 Backbone，若需复现论文原版可改为 'xception')
    # 论文使用 Xception 
    model = Model(encoder_name="resnet50", in_channels=2, num_classes=1)
    
    # 3. 运行前向传播
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    output = model(dummy_input)
    
    print(f"Model: WDNet (Backbone via SMP)")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # 预期: (2, 1, 256, 256)
    
    # 验证
    if output.shape == (2, 1, 256, 256):
        print("✅ 维度验证通过")
    else:
        print("❌ 维度验证失败")
    
    # Count parameters
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters: ", param)
    
    # Backward pass check
    criterion = nn.BCEWithLogitsLoss()
    target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
    loss = criterion(output, target)
    loss.backward()
    print("Backward pass successful. Loss:", loss.item())