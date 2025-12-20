import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GCAM(nn.Module):
    """
    Global Context Attention Module (GCAM)
    对应论文 Figure 4 及公式 (1)-(3) [cite: 216-241]
    结合了 Global Context 建模和 Transform。
    """

    def __init__(self, in_channels, ratio=4):
        super(GCAM, self).__init__()
        # Context Modeling Branch
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        # Transform Branch (Squeeze Transform)
        mid_channels = in_channels // ratio
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # Context Modeling: [B, 1, H*W]
        input_x = x.view(b, c, h * w)
        mask = self.conv_mask(x).view(b, 1, h * w)
        mask = self.softmax(mask)

        # Context Aggregation: [B, C, 1, 1]
        context = torch.matmul(input_x, mask.permute(0, 2, 1))
        context = context.view(b, c, 1, 1)

        # Transform
        transform_out = self.transform(context)

        # Addition (Fusion)
        out = x + transform_out
        return out


class GCASPP(nn.Module):
    """
    Global Context Atrous Spatial Pyramid Pooling (GCASPP)
    对应论文 Figure 2 Encoder 部分 [cite: 198]
    包含不同扩张率的卷积分支，最后通过 GCAM 进行增强。
    """

    def __init__(self, in_channels, out_channels):
        super(GCASPP, self).__init__()

        # 论文图示提到 Rate-1, Rate-6, Rate-6, Rate-6 (可能是笔误，通常为 1, 6, 12, 18)
        # 这里参考 DeepLab 标准设置: 1, 6, 12, 18
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, padding=12, dilation=12, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, padding=18, dilation=18, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 融合后通道数 = 5 * out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 引入 GCAM 增强全局上下文 [cite: 202]
        self.gcam = GCAM(out_channels)

    def forward(self, x):
        size = x.shape[2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = F.interpolate(
            self.branch5(x), size=size, mode="bilinear", align_corners=True
        )

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        out = self.conv_cat(out)
        out = self.gcam(out)
        return out


class AFFM(nn.Module):
    """
    Attention Feature Fusion Module (AFFM)
    对应论文 Figure 3
    融合 Res-2 (Low-mid) 和 Res-3 (High-mid) 特征。
    """

    def __init__(self, low_channels, high_channels, out_channels=256):
        super(AFFM, self).__init__()

        # Concat 后通道数
        concat_channels = low_channels + high_channels  # 512 + 1024 = 1536

        # 1x1 Conv 降维
        self.conv1 = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Attention Branch: AvgPool -> 1x1 -> Sigmoid
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

        # Final 1x1 Conv
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, res2, res3):
        # 1. Upsample Res-3 (2x) to match Res-2 size [cite: 178]
        res3_up = F.interpolate(
            res3, size=res2.shape[2:], mode="bilinear", align_corners=True
        )

        # 2. Concat [cite: 179]
        x = torch.cat([res2, res3_up], dim=1)

        # 3. 1x1 Conv -> BN -> Relu [cite: 180]
        x = self.conv1(x)

        # 4. Attention Weight Generation [cite: 182]
        att = self.attention(x)

        # 5. Multiplication [cite: 184]
        x = x * att

        # 6. Final 1x1 Conv [cite: 185]
        return self.conv_out(x)


class AMM(nn.Module):
    """
    Attention Modulation Module (AMM)
    对应论文 Section 2.4
    用于处理 Decoder 中的低层特征 (Res-1)。
    包含 Channel Attention + Spatial Attention。
    """

    def __init__(self, in_channels):
        super(AMM, self).__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid(),
        )

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel Attention
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x_channel = x * y

        # Spatial Attention
        # MaxPool & AvgPool along channel axis
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        spatial = torch.cat([max_pool, avg_pool], dim=1)
        spatial = self.conv_spatial(spatial)
        spatial = self.sigmoid(spatial)

        # Sequential Modulation [cite: 255]
        out = x_channel * spatial
        return out


class Model(nn.Module):
    """
    Global Context Attention Feature Fusion Network
    对应论文 Figure 2 整体架构 [cite: 124-159]
    """

    def __init__(self, in_channels=2, num_classes=1):
        super(Model, self).__init__()

        # --- Encoder: Backbone (ResNet-101) ---
        # 使用 atrous convolution (Output Stride = 16) [cite: 161]
        # replace_stride_with_dilation=[False, False, True] 使得 Layer4 stride=1, dilation=2
        resnet = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1,
            replace_stride_with_dilation=[False, False, True],
        )

        # 修改第一层以适应 2 通道输入 (原始为3)
        if in_channels != 3:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                ),
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
            )
        else:
            self.stem = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
            )
        self.layer1 = resnet.layer1  # Res-1 (256 ch)
        self.layer2 = resnet.layer2  # Res-2 (512 ch)
        self.layer3 = resnet.layer3  # Res-3 (1024 ch)
        self.layer4 = resnet.layer4  # Res-4 (2048 ch, Dilated)

        # --- Encoder Modules ---
        self.gcaspp = GCASPP(in_channels=2048, out_channels=256)  # [cite: 202]
        self.affm = AFFM(
            low_channels=512, high_channels=1024, out_channels=256
        )  # [cite: 185]

        # --- Decoder Modules ---
        self.amm = AMM(in_channels=256)  # [cite: 250]

        # Decoder Fusion Steps

        # 1. Fuse GCASPP (High) + AFFM (Mid)
        # GCASPP Out (256) + AFFM Out (256) -> Concat (512) -> Conv (256)
        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 2. Reduce AMM (Low) channels
        # 256 -> 48 (参考 DeepLabV3+ 低层特征降维惯例，虽然论文只说 1x1 reduce redundancy)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )

        # 3. Final Fusion
        # Fusion1 (256) + LowLevel (48) = 304
        self.final_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        input_size = x.shape[2:]

        # --- Encoder Pass ---
        x = self.stem(x)  # Stem (1/4 size)
        res1 = self.layer1(x)  # Res-1 (1/4 size, 256 ch)
        res2 = self.layer2(res1)  # Res-2 (1/8 size, 512 ch)
        res3 = self.layer3(res2)  # Res-3 (1/16 size, 1024 ch)
        res4 = self.layer4(res3)  # Res-4 (1/16 size, 2048 ch, Dilated)

        # --- Encoder Feature Extraction ---
        # 1. High-level: GCASPP
        feat_high = self.gcaspp(res4)  # [B, 256, H/16, W/16]

        # 2. Mid-level: AFFM
        feat_mid = self.affm(
            res2, res3
        )  # [B, 256, H/8, W/8] (Output matches Res-2 size)

        # --- Decoder Pass ---
        # 1. Low-level: AMM
        feat_low = self.amm(res1)  # [B, 256, H/4, W/4]
        feat_low = self.low_level_conv(feat_low)  # [B, 48, H/4, W/4]

        # 2. Fuse High + Mid
        # Upsample High (GCASPP) to match Mid (AFFM) size (1/16 -> 1/8)
        feat_high_up = F.interpolate(
            feat_high, size=feat_mid.shape[2:], mode="bilinear", align_corners=True
        )
        fusion_1 = torch.cat([feat_high_up, feat_mid], dim=1)  # [B, 512, H/8, W/8]
        fusion_1 = self.fusion_conv1(fusion_1)  # [B, 256, H/8, W/8]

        # 3. Fuse (High+Mid) + Low
        # Upsample Fusion_1 to match Low (AMM) size (1/8 -> 1/4)
        fusion_1_up = F.interpolate(
            fusion_1, size=feat_low.shape[2:], mode="bilinear", align_corners=True
        )
        fusion_final = torch.cat([fusion_1_up, feat_low], dim=1)  # [B, 304, H/4, W/4]

        # 4. Final Prediction
        out = self.final_conv(fusion_final)

        # Upsample to original input size (1/4 -> 1)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)

        return out


# --- Test Code ---
if __name__ == "__main__":
    # 模拟输入: Batch=2, Channel=2, Size=256x256
    dummy_input = torch.randn(2, 2, 256, 256)
    model = Model(in_channels=2, num_classes=1)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = dummy_input.to(device)

    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")  # Expected: [2, 1, 256, 256]

    # Parameter Count Check
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")

    # [新增] 计算 FLOPs
    from thop import profile

    # 注意：输入尺寸会影响 FLOPs，通常用 256x256 或 512x512 测试
    input_tensor = torch.randn(1, 2, 256, 256).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    print(f"GFLOPs: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")
