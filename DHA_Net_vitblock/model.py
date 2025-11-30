import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
import math

# --- Attention Blocks (Same as original) ---
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

# --- CNN Encoder with Internal Fusion ---
class CNNEncoderWithFusion(nn.Module):
    """ResNet-34 Encoder with Internal Fusion"""
    def __init__(self, in_channels=2):
        super(CNNEncoderWithFusion, self).__init__()
        # Load pretrained weights
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Modify first layer for 2-channel input
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight[:, :in_channels, :, :])
            self.stem = nn.Sequential(new_conv, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            
        self.layer1 = resnet.layer1 # 64, 1/4
        self.layer2 = resnet.layer2 # 128, 1/8
        self.layer3 = resnet.layer3 # 256, 1/16
        self.layer4 = resnet.layer4 # 512, 1/32
        
        # Internal Fusion Layers
        # Global (1/32) -> Medium (1/16) -> Detail (1/4)
        
        # Fusion for Medium: c3 (256) + upsampled Global (512) -> 256
        self.fuse_medium = nn.Sequential(
            nn.Conv2d(256 + 512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Fusion for Detail: c1 (64) + upsampled Medium (256) -> 64
        # Note: Using c1 for 1/4 scale detail
        self.fuse_detail = nn.Sequential(
            nn.Conv2d(64 + 256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x) # 1/4, 64
        c2 = self.layer2(c1) # 1/8, 128
        c3 = self.layer3(c2) # 1/16, 256
        c4 = self.layer4(c3) # 1/32, 512
        
        # Internal Fusion
        # 1. Global: c4
        f_global = c4
        
        # 2. Medium: c3 + upsampled Global
        f_global_up = F.interpolate(f_global, size=c3.shape[2:], mode='bilinear', align_corners=True)
        f_medium = self.fuse_medium(torch.cat([c3, f_global_up], dim=1))
        
        # 3. Detail: c1 + upsampled Medium
        f_medium_up = F.interpolate(f_medium, size=c1.shape[2:], mode='bilinear', align_corners=True)
        f_detail = self.fuse_detail(torch.cat([c1, f_medium_up], dim=1))
        
        return f_global, f_medium, f_detail

# --- ViT Encoder using timm with Manual Weight Loading ---
from torch.hub import load_state_dict_from_url

class ViTEncoder(nn.Module):
    """ ViT-Tiny Encoder using timm with robust weight loading """
    def __init__(self, img_size=256, in_chans=2, pretrained=True, pretrained_path=None):
        super(ViTEncoder, self).__init__()
        
        # Create model without pretrained weights initially
        # We will load them manually to avoid timm's HF download issues
        self.model = timm.create_model(
            'deit_tiny_patch16_224',
            pretrained=False,
            img_size=img_size,
            in_chans=in_chans
        )
        
        # Remove classifier
        self.model.reset_classifier(0)
        
        self.embed_dim = self.model.embed_dim
        self.patch_size = 16
        
        if pretrained:
            self.load_weights(pretrained_path)

    def load_weights(self, pretrained_path):
        # Official DeiT-Tiny URL (Facebook) - usually more accessible than HF
        url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
        
        try:
            if pretrained_path and os.path.exists(pretrained_path):
                print(f"Loading pretrained weights from local path: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location='cpu')
            else:
                print(f"Downloading pretrained weights from: {url}")
                checkpoint = load_state_dict_from_url(url, map_location='cpu', check_hash=True)
            
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # Adapt weights for timm model
            # 1. Patch Embedding: Adapt 3 channels -> in_chans (2)
            if 'patch_embed.proj.weight' in state_dict:
                weight = state_dict['patch_embed.proj.weight']
                if weight.shape[1] != self.model.patch_embed.proj.weight.shape[1]:
                    print(f"Adapting patch_embed from {weight.shape[1]} to {self.model.patch_embed.proj.weight.shape[1]} channels.")
                    new_weight = weight[:, :self.model.patch_embed.proj.weight.shape[1], :, :]
                    state_dict['patch_embed.proj.weight'] = new_weight

            # 2. Positional Embedding: Interpolate if needed
            if 'pos_embed' in state_dict:
                pos_embed = state_dict['pos_embed']
                # timm's pos_embed might include cls token, check shapes
                if pos_embed.shape != self.model.pos_embed.shape:
                    print(f"Resizing pos_embed from {pos_embed.shape} to {self.model.pos_embed.shape}.")
                    # Remove cls token for resizing
                    num_extra_tokens = 2 # DeiT has cls + dist
                    # Check if checkpoint has 1 or 2 extra tokens
                    # This logic assumes we are loading DeiT-Tiny
                    
                    # Extract tokens
                    # DeiT: [1, 198, 192] -> 196 patches + 1 cls + 1 dist
                    # timm deit_tiny: [1, 198, 192] (if img_size=224)
                    
                    # If target size is different, we need to interpolate
                    # Separate extra tokens
                    # Note: timm's deit_tiny might handle dist token differently depending on config
                    # But standard deit_tiny has 2 extra tokens.
                    
                    # Simple resize strategy:
                    # 1. Identify patch tokens
                    # 2. Resize patch tokens
                    # 3. Concatenate back
                    
                    # However, timm provides a utility for this: checkpoint_seq
                    # But we are doing manual load.
                    
                    # Let's try to load with strict=False first, and if pos_embed mismatches, we fix it.
                    # Actually, we should fix it before loading.
                    
                    # Assume standard DeiT-Tiny checkpoint structure
                    n_tokens = pos_embed.shape[1]
                    patch_embed_len = n_tokens - 2 # cls + dist
                    size = int(math.sqrt(patch_embed_len))
                    
                    cls_dist_tokens = pos_embed[:, :2, :]
                    patch_tokens = pos_embed[:, 2:, :]
                    
                    patch_tokens = patch_tokens.transpose(1, 2).reshape(1, self.embed_dim, size, size)
                    
                    # Target size
                    new_size = self.model.patch_embed.num_patches ** 0.5
                    new_size = int(new_size)
                    
                    patch_tokens = F.interpolate(patch_tokens, size=(new_size, new_size), mode='bilinear', align_corners=False)
                    patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
                    
                    new_pos_embed = torch.cat((cls_dist_tokens, patch_tokens), dim=1)
                    state_dict['pos_embed'] = new_pos_embed

            # Load weights
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights loaded with msg: {msg}")
            
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
            print("Training will proceed with random initialization for ViT.")

    def forward(self, x):
        # x: (B, C, H, W)
        
        # forward_features returns (B, N, C)
        x = self.model.forward_features(x)
        
        # Extract patch tokens (remove cls and dist tokens)
        # timm's forward_features for DeiT keeps them
        
        # Check number of extra tokens
        n_patches = self.model.patch_embed.num_patches
        if x.shape[1] > n_patches:
            x = x[:, -n_patches:, :]
            
        B, N, C = x.shape
        
        # Reshape to (B, C, H, W)
        H_feat = int(math.sqrt(N))
        W_feat = int(math.sqrt(N))
        
        x = x.transpose(1, 2).reshape(B, C, H_feat, W_feat)
        
        return x

# --- Decoder & Model ---
class DecoderFusionBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderFusionBlock, self).__init__()
        
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
        if x.shape[2:] != skip.shape[2:]:
             x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.att(x)
        return x

class Model(nn.Module):
    def __init__(self, in_channels=2, num_classes=1, img_size=256, pretrained_path=None):
        super(Model, self).__init__()
        
        # Encoders
        self.cnn_encoder = CNNEncoderWithFusion(in_channels)
        
        # ViT-Tiny Encoder (timm)
        self.vit_encoder = ViTEncoder(
            img_size=img_size, 
            in_chans=in_channels, 
            pretrained=True,
            pretrained_path=pretrained_path
        )
        
        # Channel configs
        # CNN Outputs:
        # Global: 512 (1/32)
        # Medium: 256 (1/16)
        # Detail: 64  (1/4)
        
        # ViT Output: 192 (1/16) for deit_tiny
        vit_dim = 192
        
        # Decoder Stages
        
        # Stage 1: Fuse Global (512, 1/32) and ViT (192, 1/16)
        # We upsample Global to 1/16 and fuse with ViT
        self.fuse_stage1 = DecoderFusionBlock(512, vit_dim, 512) # Output 512, 1/16
        
        # Stage 2: Fuse Stage1 (512, 1/16) and Medium (256, 1/16)
        self.fuse_stage2 = DecoderFusionBlock(512, 256, 256) # Output 256, 1/16
        
        # Stage 3: Fuse Stage2 (256, 1/16) and Detail (64, 1/4)
        # Upsample Stage2 to 1/4
        self.fuse_stage3 = DecoderFusionBlock(256, 64, 128) # Output 128, 1/4
        
        # Final layers
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # 1/4 -> 1
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        # Encoders
        f_global, f_medium, f_detail = self.cnn_encoder(x)
        f_vit = self.vit_encoder(x) # 1/16
        
        # Decoder
        # Stage 1: Global (upsampled) + ViT
        d1 = self.fuse_stage1(f_global, f_vit) # 1/16
        
        # Stage 2: d1 + Medium
        d2 = self.fuse_stage2(d1, f_medium) # 1/16
        
        # Stage 3: d2 (upsampled) + Detail
        d3 = self.fuse_stage3(d2, f_detail) # 1/4
        
        # Final Output
        out = self.final_up(d3)
        out = self.final_conv(out)
        
        return out

if __name__ == "__main__":
    # Test
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(in_channels=2, num_classes=1).to(device)
        dummy_input = torch.randn(2, 2, 256, 256).to(device)
        output = model(dummy_input)
        print(f"Input: {dummy_input.shape}")
        print(f"Output: {output.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
    except Exception as e:
        print(f"Error during test: {e}")
