"""
tests/test_models.py
====================
Unit tests verifying architecture, shape integrity, and gradient flow for:
1. ConvUNet (Stage 1 Continuous Despeckler)
2. ConvAutoencoder (Stage 2 Continuous Tissue Baseline Autoencoder)
"""

import pytest
import torch
from models.despeckler.model import ConvUNet, get_despeckler
from models.anomaly_detector.autoencoder import ConvAutoencoder


def test_conv_unet_shape_and_range():
    model = ConvUNet(in_channels=1, base_ch=16)
    x = torch.rand(2, 1, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output must be bounded in [0, 1] by sigmoid"


def test_conv_autoencoder_shape_and_range():
    model = ConvAutoencoder(in_channels=1)
    x = torch.rand(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output must be bounded in [0, 1] by sigmoid"


def test_conv_unet_gradient_flow():
    model = get_despeckler(in_channels=1, base_ch=16)
    x = torch.rand(1, 1, 64, 64)
    target = torch.rand(1, 1, 64, 64)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    assert model.enc1.conv.weight.grad is not None, "Gradient must flow back to ConvUNet encoder"
    assert torch.isfinite(model.enc1.conv.weight.grad).all(), "Gradients must be finite"


def test_conv_autoencoder_gradient_flow():
    model = ConvAutoencoder(in_channels=1)
    x = torch.rand(1, 1, 64, 64)
    target = torch.rand(1, 1, 64, 64)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    assert model.enc[0].weight.grad is not None, "Gradient must flow back to autoencoder encoder"
    assert torch.isfinite(model.enc[0].weight.grad).all(), "Gradients must be finite"
