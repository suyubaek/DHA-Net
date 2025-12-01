import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DualAttentionBlock(nn.Module):
    """CBAM-like Dual Attention (Channel + Spatial)"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(DualAttentionBlock, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CNNEncoder(nn.Module):
    """ResNet-34 Encoder"""
    def __init__(self, in_channels=2):
        super(CNNEncoder, self).__init__()
        # Load pretrained weights
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Modify first layer for 2-channel input
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Initialize with first 'in_channels' of pretrained weights (e.g. R, G)
            # User suggestion: R and G weights are better than random or average for SAR
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight[:, :in_channels, :, :])
            
            self.stem = nn.Sequential(
                new_conv,
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
            
        self.layer1 = resnet.layer1 # 64
        self.layer2 = resnet.layer2 # 128
        self.layer3 = resnet.layer3 # 256
        self.layer4 = resnet.layer4 # 512
        
    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4

class ViTEncoder(nn.Module):
    """Swin Transformer Tiny Encoder"""
    def __init__(self, in_channels=2):
        super(ViTEncoder, self).__init__()
        # Use torchvision's swin_t
        self.adapter = nn.Conv2d(in_channels, 3, 1)
        swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.features = swin.features
        
    def forward(self, x):
        x = self.adapter(x)
        
        # Swin Transformer features structure:
        # 0: PatchPartition + LinearEmbedding
        # 1: Stage 1 Blocks
        # 2: PatchMerging
        # 3: Stage 2 Blocks
        # 4: PatchMerging
        # 5: Stage 3 Blocks
        # 6: PatchMerging
        # 7: Stage 4 Blocks
        
        # Stage 1 (Output 1/4)
        x = self.features[0](x)
        x = self.features[1](x)
        s1 = x.permute(0, 3, 1, 2) # BHWC -> BCHW
        
        # Stage 2 (Output 1/8)
        x = self.features[2](x)
        x = self.features[3](x)
        s2 = x.permute(0, 3, 1, 2)
        
        # Stage 3 (Output 1/16)
        x = self.features[4](x)
        x = self.features[5](x)
        s3 = x.permute(0, 3, 1, 2)
        
        # Stage 4 (Output 1/32)
        x = self.features[6](x)
        x = self.features[7](x)
        s4 = x.permute(0, 3, 1, 2)
        
        return s1, s2, s3, s4


class FusionBlock(nn.Module):
    def __init__(self, cnn_channels, vit_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cnn_channels + vit_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.att = DualAttentionBlock(out_channels)
        
    def forward(self, cnn_feat, vit_feat):
        # Ensure sizes match (ViT might have slight rounding diffs if not exact)
        if cnn_feat.shape[2:] != vit_feat.shape[2:]:
            vit_feat = F.interpolate(vit_feat, size=cnn_feat.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([cnn_feat, vit_feat], dim=1)
        x = self.conv(x)
        x = self.att(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Use Transposed Convolution for upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.att = DualAttentionBlock(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
             x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.att(x)
        return x


class Model(nn.Module):
    def __init__(self, in_channels=2, num_classes=1):
        super(Model, self).__init__()
        
        # Encoders
        self.cnn_encoder = CNNEncoder(in_channels)
        self.vit_encoder = ViTEncoder(in_channels)
        
        # Channel configs
        # ResNet34: 64, 128, 256, 512
        # Swin-T:   96, 192, 384, 768
        
        # Fusion Blocks
        self.fuse1 = FusionBlock(64, 96, 64)    # 1/4
        self.fuse2 = FusionBlock(128, 192, 128) # 1/8
        self.fuse3 = FusionBlock(256, 384, 256) # 1/16
        self.fuse4 = FusionBlock(512, 768, 512) # 1/32
        
        # Decoder
        # Input to decoder is fused4 (512)
        self.dec4 = DecoderBlock(512, 256, 256) # 512 + 256(skip) -> 256
        self.dec3 = DecoderBlock(256, 128, 128) # 256 + 128(skip) -> 128
        self.dec2 = DecoderBlock(128, 64, 64)   # 128 + 64(skip)  -> 64
        
        # Final layers
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # 1/4 -> 1
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

        # Classification Head (Stage 1)
        # Input: f4 (512, H/32, W/32)
        self.cls_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1) # Binary classification (Water / No Water)
        )

    def forward(self, x, mode='seg'):
        # Encoders
        c1, c2, c3, c4 = self.cnn_encoder(x)
        v1, v2, v3, v4 = self.vit_encoder(x)
        
        # Fusion
        f1 = self.fuse1(c1, v1) # 1/4, 64
        f2 = self.fuse2(c2, v2) # 1/8, 128
        f3 = self.fuse3(c3, v3) # 1/16, 256
        f4 = self.fuse4(c4, v4) # 1/32, 512
        
        if mode == 'cls':
            # Stage 1: Classification
            x = self.cls_avg_pool(f4)
            out = self.cls_head(x)
            return out
            
        # Stage 2: Segmentation
        # Decoder
        d4 = self.dec4(f4, f3) # -> 1/16, 256
        d3 = self.dec3(d4, f2) # -> 1/8, 128
        d2 = self.dec2(d3, f1) # -> 1/4, 64
        
        # Final Output
        out = self.final_up(d2)
        out = self.final_conv(out)
        
        return out

if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(in_channels=2, num_classes=1).to(device)
    dummy_input = torch.randn(2, 2, 256, 256).to(device)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
