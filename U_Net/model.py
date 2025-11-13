import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FeatureFusion(nn.Module):
    """Fuse per-modality feature maps before decoder consumption."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_primary, x_secondary):
        x = torch.cat([x_primary, x_secondary], dim=1)
        return self.fuse(x)


class Model(nn.Module):
    def __init__(self, sar_channels, optical_channels, n_classes, bilinear=False):
        super().__init__()
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc_sar = DoubleConv(sar_channels, 64)
        self.inc_opt = DoubleConv(optical_channels, 64)
        self.fuse_inc = FeatureFusion(128, 64)

        self.down1_sar = Down(64, 128)
        self.down1_opt = Down(64, 128)
        self.fuse_down1 = FeatureFusion(256, 128)

        self.down2_sar = Down(128, 256)
        self.down2_opt = Down(128, 256)
        self.fuse_down2 = FeatureFusion(512, 256)

        self.down3_sar = Down(256, 512)
        self.down3_opt = Down(256, 512)
        self.fuse_down3 = FeatureFusion(1024, 512)

        factor = 2 if bilinear else 1
        bottom_channels = 1024 // factor
        self.down4_sar = Down(512, bottom_channels)
        self.down4_opt = Down(512, bottom_channels)
        self.fuse_bottom = FeatureFusion(bottom_channels * 2, bottom_channels)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x_sar, x_optical):
        x1_sar = self.inc_sar(x_sar)
        x1_opt = self.inc_opt(x_optical)
        x1 = self.fuse_inc(x1_sar, x1_opt)

        x2_sar = self.down1_sar(x1_sar)
        x2_opt = self.down1_opt(x1_opt)
        x2 = self.fuse_down1(x2_sar, x2_opt)

        x3_sar = self.down2_sar(x2_sar)
        x3_opt = self.down2_opt(x2_opt)
        x3 = self.fuse_down2(x3_sar, x3_opt)

        x4_sar = self.down3_sar(x3_sar)
        x4_opt = self.down3_opt(x3_opt)
        x4 = self.fuse_down3(x4_sar, x4_opt)

        x5_sar = self.down4_sar(x4_sar)
        x5_opt = self.down4_opt(x4_opt)
        x5 = self.fuse_bottom(x5_sar, x5_opt)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc_sar = torch.utils.checkpoint(self.inc_sar)
        self.inc_opt = torch.utils.checkpoint(self.inc_opt)
        self.down1_sar = torch.utils.checkpoint(self.down1_sar)
        self.down1_opt = torch.utils.checkpoint(self.down1_opt)
        self.down2_sar = torch.utils.checkpoint(self.down2_sar)
        self.down2_opt = torch.utils.checkpoint(self.down2_opt)
        self.down3_sar = torch.utils.checkpoint(self.down3_sar)
        self.down3_opt = torch.utils.checkpoint(self.down3_opt)
        self.down4_sar = torch.utils.checkpoint(self.down4_sar)
        self.down4_opt = torch.utils.checkpoint(self.down4_opt)
        self.fuse_inc = torch.utils.checkpoint(self.fuse_inc)
        self.fuse_down1 = torch.utils.checkpoint(self.fuse_down1)
        self.fuse_down2 = torch.utils.checkpoint(self.fuse_down2)
        self.fuse_down3 = torch.utils.checkpoint(self.fuse_down3)
        self.fuse_bottom = torch.utils.checkpoint(self.fuse_bottom)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
