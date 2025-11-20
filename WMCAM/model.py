import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet101_Weights


class WaterMatchingNetwork(nn.Module):
    """
    ResNet-101 backbone classification network for weakly supervised water segmentation.
    Returns both classification logits and the penultimate feature vector needed by NTC loss.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        local_weight_path: Optional[str] = None,
        strict_load: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        # Resolve local checkpoint path (optional).
        if local_weight_path is None:
            local_weight_path = os.path.expanduser(
                "~/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth"
            )

        if os.path.exists(local_weight_path):
            resnet = models.resnet101(weights=None)
            state = torch.load(local_weight_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            resnet.load_state_dict(state, strict=strict_load)
        else:
            try:
                resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
            except TypeError:
                resnet = models.resnet101(pretrained=True)

        # Feature hierarchy (f_b^0 ... f_b^3 in the paper).
        conv1 = resnet.conv1
        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=conv1.bias is not None,
            )
            with torch.no_grad():
                mean_weight = conv1.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
                if conv1.bias is not None:
                    new_conv.bias.copy_(conv1.bias)
            conv1 = new_conv

        self.layer0 = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # feat0
        self.layer2 = resnet.layer2  # feat1
        self.layer3 = resnet.layer3  # feat2
        self.layer4 = resnet.layer4  # feat3

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        self.gradients: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks_registered = False

    def hook_layers(self) -> None:
        """Register forward/backward hooks for Grad-CAM extraction."""
        if self._hooks_registered:
            return

        def forward_hook(name: str):
            def hook(module, _input, output):
                self.activations[name] = output.detach()

            return hook

        def backward_hook(name: str):
            def hook(module, _grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()

            return hook

        for name, module in zip(
            ("feat0", "feat1", "feat2", "feat3"),
            (self.layer1, self.layer2, self.layer3, self.layer4),
        ):
            module.register_forward_hook(forward_hook(name))
            module.register_full_backward_hook(backward_hook(name))

        self._hooks_registered = True

    def forward(self, x: torch.Tensor):
        x = self.layer0(x)

        f0 = self.layer1(x)
        f1 = self.layer2(f0)
        f2 = self.layer3(f1)
        f3 = self.layer4(f2)

        pooled = self.avgpool(f3)
        feat_vec = pooled.flatten(1)

        logits = self.fc(feat_vec)
        return logits, feat_vec


class Model(WaterMatchingNetwork):
    """Project-standard alias so training scripts can import `Model`."""

    def __init__(self, num_classes: int = 2, in_channels: int = 3, **kwargs) -> None:
        super().__init__(num_classes=num_classes, in_channels=in_channels, **kwargs)


def calculate_ntc_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Non-Water Targets Consistency (NTC) loss, Eq. (8)(9) in paper.

    Args:
        features: (B, D) batch-level feature vectors.
        labels:   (B,) image-level class labels.

    Returns:
        Scalar tensor representing L_ntc.
    """
    batch_size = features.size(0)
    loss_ntc = features.new_tensor(0.0)

    for idx in range(1, batch_size):
        if labels[idx] == labels[idx - 1]:
            continue
        loss_ntc = loss_ntc + F.mse_loss(features[idx], features[idx - 1])

    return loss_ntc / max(batch_size, 1)


def generate_mwgm_cam(
    model: WaterMatchingNetwork,
    input_image: torch.Tensor,
    target_class: int = 1,
) -> torch.Tensor:
    """
    Multilevel Water-backscatter Guided CAM generation (MWGM).

    Args:
        model:        Trained WaterMatchingNetwork with hooks registered.
        input_image:  Tensor of shape (B, C, H, W); device must match model.
        target_class: Class index representing water.

    Returns:
        Tensor CAM of shape (B, 1, H, W) normalized to [0, 1].
    """
    if not model._hooks_registered:
        model.hook_layers()

    model.gradients.clear()
    model.activations.clear()

    was_training = model.training
    model.eval()
    with torch.enable_grad():
        img_inverted = 1.0 - input_image
        min_val = img_inverted.amin(dim=(-2, -1), keepdim=True)
        max_val = img_inverted.amax(dim=(-2, -1), keepdim=True)
        omega_wh = (img_inverted - min_val) / (max_val - min_val + 1e-8)
        if omega_wh.size(1) > 1:
            omega_wh = omega_wh.mean(dim=1, keepdim=True)

        model.zero_grad(set_to_none=True)
        logits, _ = model(input_image)

        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        cams = []
        spatial_size = input_image.shape[2:]

        for name in ("feat0", "feat1", "feat2", "feat3"):
            gradients = model.gradients.get(name)
            activations = model.activations.get(name)
            if gradients is None or activations is None:
                raise RuntimeError(
                    f"Hook outputs for {name} not found. Ensure a forward pass ran with hooks."
                )

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            omega_resized = F.interpolate(
                omega_wh, size=activations.shape[2:], mode="bilinear", align_corners=False
            )
            activations_modified = activations * omega_resized
            cam = torch.sum(weights * activations_modified, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam, size=spatial_size, mode="bilinear", align_corners=False
            )
            if torch.any(cam > 0):
                cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + 1e-8)
            cams.append(cam)

        cams_tensor = torch.cat(cams, dim=1)
        final_cam, _ = torch.max(cams_tensor, dim=1, keepdim=True)

    if was_training:
        model.train()

    return final_cam


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    in_channels = 2
    height = width = 256

    dummy_images = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    dummy_masks = torch.randint(0, 2, (batch_size, 1, height, width))
    dummy_labels = dummy_masks.flatten(1).float().mean(dim=1).gt(0).long()

    model = Model(num_classes=2, in_channels=in_channels)
    model.hook_layers()

    logits, features = model(dummy_images)
    ce_loss = nn.CrossEntropyLoss()(logits, dummy_labels)
    ntc_loss = calculate_ntc_loss(features, dummy_labels)
    total_loss = ce_loss + 0.3 * ntc_loss

    total_loss.backward()

    print(
        f"Forward OK: logits={tuple(logits.shape)}, "
        f"features={tuple(features.shape)}, "
        f"loss={total_loss.item():.4f}"
    )

    with torch.no_grad():
        cams = generate_mwgm_cam(model, dummy_images, target_class=1)
        print(f"CAM OK: shape={tuple(cams.shape)}, min={cams.min():.4f}, max={cams.max():.4f}")