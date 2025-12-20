import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 conv
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Atrous convolutions
        rates = [12, 24, 36]
        for rate in rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Image pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            # Handle image pooling case
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                feat = conv(x)
                feat = F.interpolate(
                    feat, size=x.shape[2:], mode="bilinear", align_corners=True
                )
            else:
                feat = conv(x)
            res.append(feat)
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=2, num_classes=1, backbone="resnet50"):
        super(DeepLabV3Plus, self).__init__()

        if backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Modify first layer
            if in_channels != 3:
                old_conv = resnet.conv1
                new_conv = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight[:, :in_channels, :, :])
                resnet.conv1 = new_conv

            self.backbone = resnet
            # ResNet50: layer1=256, layer2=512, layer3=1024, layer4=2048
            low_level_channels = 256
            high_level_channels = 2048
        elif backbone == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            if in_channels != 3:
                old_conv = resnet.conv1
                new_conv = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight[:, :in_channels, :, :])
                resnet.conv1 = new_conv
            self.backbone = resnet
            # ResNet34: layer1=64, layer2=128, layer3=256, layer4=512
            low_level_channels = 64
            high_level_channels = 512

        self.aspp = ASPP(high_level_channels, 256)

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        # Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        low_level_feat = self.backbone.layer1(x)
        x = self.backbone.layer2(low_level_feat)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # ASPP
        x = self.aspp(x)
        x = F.interpolate(
            x, size=low_level_feat.shape[2:], mode="bilinear", align_corners=True
        )

        # Low-level features
        low_level_feat = self.low_level_conv(low_level_feat)

        # Decoder
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)

        return x


class Model(nn.Module):
    def __init__(self, in_channels=2, num_classes=1):
        super(Model, self).__init__()
        self.model = DeepLabV3Plus(in_channels, num_classes, backbone="resnet50")

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(in_channels=2, num_classes=1).to(device)
    dummy_input = torch.randn(2, 2, 256, 256).to(device)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    # [新增] 计算 FLOPs
    from thop import profile

    # 注意：输入尺寸会影响 FLOPs，通常用 256x256 或 512x512 测试
    input_tensor = torch.randn(1, 2, 256, 256).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    print(f"GFLOPs: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")
