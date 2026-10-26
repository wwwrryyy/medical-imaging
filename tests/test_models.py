"""
tests/test_models.py
====================
Unit tests verifying architecture, shape integrity, and gradient flow for:
1. ConvUNet (CNN Default)
2. SpikingUNet (SNN Baseline, T=4)
"""

import pytest
import torch
from models.despeckler.model import ConvUNet, SpikingUNet, get_despeckler


def test_conv_unet_shape_and_range():
    model = ConvUNet(in_channels=1, base_ch=16)
    x = torch.rand(2, 1, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output must be bounded in [0, 1] by sigmoid"


def test_spiking_unet_shape_and_range():
    model = SpikingUNet(in_channels=1, base_ch=16, num_steps=4)
    x = torch.rand(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output must be bounded in [0, 1] by sigmoid"


def test_gradient_flow_both_models():
    # Test CNN
    cnn = get_despeckler("cnn", in_channels=1, base_ch=16)
    x_cnn = torch.rand(1, 1, 64, 64)
    target = torch.rand(1, 1, 64, 64)
    loss_cnn = torch.nn.functional.mse_loss(cnn(x_cnn), target)
    loss_cnn.backward()
    assert cnn.enc1.conv.weight.grad is not None, "Gradient must flow back to CNN encoder"
    assert torch.isfinite(cnn.enc1.conv.weight.grad).all(), "Gradients must be finite"

    # Test SNN
    snn = get_despeckler("snn", in_channels=1, base_ch=16, num_steps=3)
    x_snn = torch.rand(1, 1, 64, 64)
    loss_snn = torch.nn.functional.mse_loss(snn(x_snn), target)
    loss_snn.backward()
    assert snn.enc1.conv.weight.grad is not None, "Gradient must flow back to SNN encoder"
    assert torch.isfinite(snn.enc1.conv.weight.grad).all(), "Gradients must be finite"


def test_factory_error_handling():
    with pytest.raises(ValueError):
        get_despeckler("transformer")
