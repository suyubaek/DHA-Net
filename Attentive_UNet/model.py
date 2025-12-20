import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel 'Squeeze & Excitation' (scSE) Block.
    论文使用此模块增强 Encoder 的特征图 [cite: 271, 295]。
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()

        # Channel Squeeze and Excitation (cSE)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

        # Spatial Squeeze and Excitation (sSE)
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()

        # cSE 路径
        chn_se = self.avg_pool(x).view(batch, channels)
        chn_se = self.channel_excitation(chn_se).view(batch, channels, 1, 1)
        chn_se = x * chn_se

        # sSE 路径
        spa_se = self.spatial_se(x)
        spa_se = x * spa_se

        # 并行组合 (Concurrent Combination: Element-wise Add)
        return chn_se + spa_se


class DecoderBlock(nn.Module):
    """
    标准解码器块：转置卷积上采样 + 拼接 Skip Connection + 卷积处理 [cite: 247, 249]。
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 1. 转置卷积上采样 (Upsampling)
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )

        # 2. 计算拼接后的通道数
        # 输入通道减半后 + Skip Connection 的通道
        concat_channels = (in_channels // 2) + skip_channels

        # 3. 卷积层处理融合后的特征
        self.conv = nn.Sequential(
            nn.Conv2d(
                concat_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        # 上采样
        x = self.up(x)

        # 处理尺寸不匹配 (如果输入尺寸不是 2 的倍数可能会发生)
        if x.size() != skip.size():
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        # 拼接 (Concatenation)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Model(nn.Module):
    """
    双流融合网络 (Dual-stream Fusion Network) [cite: 292]。
    - Stream 1: VV, VH, DEM [cite: 293]
    - Stream 2: VV, VH, PW (Permanent Water) [cite: 293]
    - Backbone: ResNet-50 (ImageNet Pretrained) [cite: 305]
    - Fusion: Element-wise Addition after Attention [cite: 296]
    """

    def __init__(self, in_channels=3, num_classes=1):
        super(Model, self).__init__()

        # --- 1. 定义双流编码器 (Dual Encoders) ---
        # 使用 ResNet-50 权重初始化 [cite: 305]
        self.encoder1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 修改第一层卷积以适应输入通道数 (如果不是3通道)
        if in_channels != 3:
            self.encoder1.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.encoder2.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # ResNet-50 特征层通道数:
        # Layer1: 256, Layer2: 512, Layer3: 1024, Layer4: 2048
        filters = [256, 512, 1024, 2048]

        # --- 2. 定义注意力模块 (scSE Blocks) ---
        # 每个编码器的每个特征层都需要经过 scSE [cite: 295]
        # 为了代码简洁，这里定义一组共享参数的 Block，或者分别为两个流定义（论文未明确，通常分开定义更灵活）
        # 这里为两个流分别定义，确保能学习不同模态的特征
        self.scse1_layers = nn.ModuleList([SCSEBlock(c) for c in filters])
        self.scse2_layers = nn.ModuleList([SCSEBlock(c) for c in filters])

        # --- 3. 定义解码器 (Decoder) ---
        # Decoder 4: 处理 F4 (2048) -> 融合 F3 (1024)
        self.decoder4 = DecoderBlock(filters[3], filters[2], 512)

        # Decoder 3: 处理上一层 (512) -> 融合 F2 (512)
        self.decoder3 = DecoderBlock(512, filters[1], 256)

        # Decoder 2: 处理上一层 (256) -> 融合 F1 (256)
        self.decoder2 = DecoderBlock(256, filters[0], 64)

        # Decoder 1: 处理上一层 (64) -> 最后的上采样
        # ResNet Layer1 输出尺寸是原图的 1/4，Decoder2 输出后也是 1/4
        # 我们需要再接卷积和上采样回到原图尺寸
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 上采样到 1/2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 上采样到 原图
            nn.Conv2d(16, num_classes, kernel_size=1),  # 输出二值分割图 [cite: 242]
        )

    def _extract_features(self, encoder, x):
        """提取 ResNet 的 4 个阶段特征"""
        x = encoder.conv1(x)
        x = encoder.bn1(x)
        x = encoder.relu(x)
        x = encoder.maxpool(x)

        f1 = encoder.layer1(x)  # (B, 256, H/4, W/4)
        f2 = encoder.layer2(f1)  # (B, 512, H/8, W/8)
        f3 = encoder.layer3(f2)  # (B, 1024, H/16, W/16)
        f4 = encoder.layer4(f3)  # (B, 2048, H/32, W/32)
        return [f1, f2, f3, f4]

    def forward(self, input1, input2=None):
        """
        Args:
            input1: Tensor (B, C, H, W)
            input2: Tensor (B, C, H, W) - Optional, if None, duplicate input1
        Returns:
            output: Tensor (B, 1, H, W) - Segmentation Map
        """
        if input2 is None:
            input2 = input1

        # 1. 提取双流特征
        features1 = self._extract_features(self.encoder1, input1)  # [f1, f2, f3, f4]
        features2 = self._extract_features(self.encoder2, input2)  # [f1, f2, f3, f4]

        # 2. 注意力增强与特征融合 (Attention & Fusion)
        fused_features = []
        for i in range(4):
            # 分别应用 scSE 注意力
            att_f1 = self.scse1_layers[i](features1[i])
            att_f2 = self.scse2_layers[i](features2[i])

            # 逐元素相加融合 (Element-wise Addition) [cite: 296]
            fused = att_f1 + att_f2
            fused_features.append(fused)

        # 获取融合后的各级特征
        f1, f2, f3, f4 = fused_features

        # 3. 解码过程 (Decoder)
        # f4 是最深层特征，作为解码器起点
        d4 = self.decoder4(f4, f3)  # F4 + Skip(F3)
        d3 = self.decoder3(d4, f2)  # + Skip(F2)
        d2 = self.decoder2(d3, f1)  # + Skip(F1)

        # 4. 最终输出
        # 经过 decoder2 后尺寸为原图 1/4，需要恢复到原图
        out = self.final_conv(d2)

        return out


if __name__ == "__main__":
    print("#### Test Case ###")
    # 模拟论文中的输入尺寸 512x512 [cite: 164, 304]
    batch_size = 2
    dummy_input1 = torch.randn(batch_size, 3, 256, 256)  # VV, VH, DEM
    dummy_input2 = torch.randn(batch_size, 3, 256, 256)  # VV, VH, PW

    model = Model(in_channels=3)

    # 如果有 GPU 则使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input1 = dummy_input1.to(device)
    dummy_input2 = dummy_input2.to(device)

    output = model(dummy_input1, dummy_input2)

    print(f"Input 1 shape: {dummy_input1.shape}")
    print(f"Input 2 shape: {dummy_input2.shape}")
    print(f"Output shape: {output.shape}")  # 预期输出: (2, 1, 256, 256)

    # Count parameters
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters: ", param)

    # Backward pass check
    criterion = nn.BCEWithLogitsLoss()
    target = torch.randint(0, 2, (batch_size, 1, 256, 256)).float().to(device)
    loss = criterion(output, target)
    loss.backward()
    print("Backward pass successful. Loss:", loss.item())

    # [新增] 计算 FLOPs
    from thop import profile

    # 注意：输入尺寸会影响 FLOPs，通常用 256x256 或 512x512 测试
    input_tensor = torch.randn(1, 2, 256, 256).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    print(f"GFLOPs: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")
