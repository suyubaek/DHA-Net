import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

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


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Global Average Pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        
        # Project after concatenation
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class CNNEncoder(nn.Module):
    """ResNet-50 Encoder"""
    def __init__(self, in_channels=2):
        super(CNNEncoder, self).__init__()
        # Load pretrained weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify first layer for 2-channel input
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Initialize with first 'in_channels' of pretrained weights
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
            
        self.layer1 = resnet.layer1 # 256
        self.layer2 = resnet.layer2 # 512
        self.layer3 = resnet.layer3 # 1024
        self.layer4 = resnet.layer4 # 2048
        
    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4

class DeiTTiny(nn.Module):
    """Manual DeiT-Tiny Implementation with Pretrained Weights"""
    def __init__(self, in_channels=2, img_size=256, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., pretrained=True):
        super(DeiTTiny, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token and distill token 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=int(embed_dim * mlp_ratio), 
                                                   activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Init weights
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=.02)
        torch.nn.init.trunc_normal_(self.dist_token, std=.02)
        self.apply(self._init_weights)
        
        if pretrained:
            self._load_pretrained_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _load_pretrained_weights(self):
        url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth"
        try:
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            
            new_dict = {}
            for k, v in state_dict.items():
                # Map keys from official DeiT to nn.TransformerEncoder structure
                if "blocks" in k:
                    parts = k.split(".")
                    if len(parts) < 3:
                        continue
                    layer_idx = parts[1]
                    
                    # Filter out layers that exceed our current depth
                    if int(layer_idx) >= self.depth:
                        continue
                        
                    module = parts[2] 
                    
                    prefix = f"blocks.layers.{layer_idx}"
                    
                    if module == "norm1":
                        new_k = f"{prefix}.norm1.{parts[-1]}"
                        new_dict[new_k] = v
                    elif module == "norm2":
                        new_k = f"{prefix}.norm2.{parts[-1]}"
                        new_dict[new_k] = v
                    elif module == "attn":
                        if len(parts) < 4: continue
                        sub = parts[3] # qkv or proj
                        if sub == "qkv":
                            new_k = f"{prefix}.self_attn.in_proj_{parts[-1]}" 
                            new_dict[new_k] = v
                        elif sub == "proj":
                            new_k = f"{prefix}.self_attn.out_proj.{parts[-1]}"
                            new_dict[new_k] = v
                    elif module == "mlp":
                        if len(parts) < 4: continue
                        sub = parts[3] # fc1 or fc2
                        if sub == "fc1":
                            new_k = f"{prefix}.linear1.{parts[-1]}"
                            new_dict[new_k] = v
                        elif sub == "fc2":
                            new_k = f"{prefix}.linear2.{parts[-1]}"
                            new_dict[new_k] = v
                elif "patch_embed.proj" in k:
                    # Resize patch_embed if channels mismatch (3 vs 2)
                    if "weight" in k and v.ndim >= 2 and v.shape[1] == 3 and self.patch_embed.weight.shape[1] != 3:
                        print(f"Adapting patch_embed from 3 to {self.patch_embed.weight.shape[1]} channels")
                        v_new = v[:, :self.patch_embed.weight.shape[1], :, :]
                        new_dict["patch_embed.weight"] = v_new
                    else:
                        new_k = k.replace("patch_embed.proj", "patch_embed")
                        new_dict[new_k] = v
                        
                elif k in ["cls_token", "dist_token", "pos_embed"]:
                     if k == "pos_embed":
                         # Resize pos_embed
                         # Pretrained: (1, 198, 192) -> (1, 14*14+2, 192) = (1, 198, 192). 
                         # If img_size=256, patches=16 -> 16*16=256 patches. Total 258.
                         # Need to interpolate pos_embed.
                         n_tokens = 2 # cls + dist
                         old_pos = v
                         new_pos = self.pos_embed
                         if old_pos.shape != new_pos.shape:
                             print(f"Interpolating pos_embed: {old_pos.shape} -> {new_pos.shape}")
                             # Extract tokens
                             old_pos_tokens = old_pos[:, :n_tokens]
                             old_pos_grid = old_pos[:, n_tokens:]
                             
                             # Reshape grid to square
                             gs_old = int(math.sqrt(old_pos_grid.shape[1]))
                             old_pos_grid = old_pos_grid.transpose(1, 2).reshape(1, self.embed_dim, gs_old, gs_old)
                             
                             # Interpolate
                             gs_new = int(math.sqrt(new_pos.shape[1] - n_tokens))
                             new_pos_grid = F.interpolate(old_pos_grid, size=(gs_new, gs_new), mode='bilinear', align_corners=False)
                             new_pos_grid = new_pos_grid.flatten(2).transpose(1, 2)
                             
                             new_v = torch.cat([old_pos_tokens, new_pos_grid], dim=1)
                             new_dict[k] = new_v
                         else:
                             new_dict[k] = v
                     else:
                        new_dict[k] = v
                elif k in ["norm.weight", "norm.bias"]:
                    new_dict[k] = v
            
            # Load
            msg = self.load_state_dict(new_dict, strict=False)
            print(f"Loaded DeiT Pretrained Weights: {msg}")
            
        except Exception as e:
            print(f"Failed to load DeiT weights: {e}")
            print("Proceeding with random initialization for ViT.")

    def forward(self, x):
        B = x.shape[0]
        # Patch Embed
        x = self.patch_embed(x).flatten(2).transpose(1, 2) # B, N, C
        
        # Add Tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        
        # Pos Embed
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer
        x = self.blocks(x)
        x = self.norm(x)
        
        # Remove tokens and reshape
        x = x[:, 2:] # Skip cls and dist
        H = W = int(math.sqrt(self.num_patches))
        x = x.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)
        
        return x

# ------------------------------------------------------------------------------
# 4. Decoders & Smoothing
# ------------------------------------------------------------------------------

class SmoothBlock(nn.Module):
    """Smoothes skip connections with 1x1 then 3x3 to reduce noise."""
    def __init__(self, in_channels, out_channels):
        super(SmoothBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.reduce(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.smooth(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Upsample previous feature
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Smooth the skip connection before fusion
        self.skip_smooth = SmoothBlock(skip_channels, 48) # Heuristic: reduce skip influence/dim
        
        # Main Block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 48, out_channels, 3, padding=1, bias=False),
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
        
        skip = self.skip_smooth(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.att(x)
        return x

# ------------------------------------------------------------------------------
# 5. Main Model
# ------------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, in_channels=2, num_classes=1):
        super(Model, self).__init__()
        
        # Encoders
        self.cnn = CNNEncoder(in_channels) # ResNet50
        self.vit = DeiTTiny(in_channels, depth=4)   # DeiT-Tiny (Depth=4)
        
        self.vit_down = nn.Conv2d(192, 192, 3, stride=2, padding=1) # 1/16 -> 1/32
        
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(2048 + 192, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # ASPP with Global Pooling
        self.aspp = ASPP(512, atrous_rates=[6, 12, 18], out_channels=256)
        
        # Decoder
        # ASPP Out: 256ch at 1/32 (or roughly thereabouts)
        
        # Dec4: Input(256) + Skip(C3=1024) -> 256
        self.dec4 = DecoderBlock(256, 1024, 256)
        
        # Dec3: Input(256) + Skip(C2=512) -> 128
        self.dec3 = DecoderBlock(256, 512, 128)
        
        # Dec2: Input(128) + Skip(C1=256) -> 64
        self.dec2 = DecoderBlock(128, 256, 64)
        
        # Final Upsample x4 to get to original resolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
        
    def forward(self, x):
        # Encoders
        c1, c2, c3, c4 = self.cnn(x)
        v = self.vit(x) # 1/16
        
        # Fusion at Bottleneck
        # Align ViT to C4 (1/32)
        v_down = self.vit_down(v)
        if c4.shape[2:] != v_down.shape[2:]:
            v_down = F.interpolate(v_down, size=c4.shape[2:], mode='bilinear', align_corners=False)
            
        f = torch.cat([c4, v_down], dim=1)
        f = self.bottleneck_fusion(f)
        
        # ASPP
        f = self.aspp(f) # 1/32, 256ch
        
        # Decoder
        # Skip C3 (1/16), C2 (1/8), C1 (1/4)
        
        d4 = self.dec4(f, c3) # -> 1/16
        d3 = self.dec3(d4, c2) # -> 1/8
        d2 = self.dec2(d3, c1) # -> 1/4
        
        # Final Output
        out = F.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.final_conv(out)
        
        return out

if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tiny test
    try:
        model = Model(in_channels=2, num_classes=1).to(device)
        dummy_input = torch.randn(2, 2, 256, 256).to(device)
        output = model(dummy_input)
        print(f"Input: {dummy_input.shape}")
        print(f"Output: {output.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
    except Exception as e:
        print(e)
