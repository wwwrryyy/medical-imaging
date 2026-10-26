"""
model.py
========
Dual Despeckle Filtering Architectures for Point-of-Care OCT/Ultrasound:
1. ConvUNet (Default): Classical continuous U-Net, T=1, SiLU activations.
2. SpikingUNet (Baseline): Neuromorphic Leaky Integrate-and-Fire (LIF) U-Net, T=4 timesteps.

Both models share identical channel progression and skip connections for rigorous comparison.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ---------------------------------------------------------------------------
# Classical Continuous CNN Blocks (Default)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → SiLU activation block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TransConvBlock(nn.Module):
    """ConvTranspose2d → BatchNorm → SiLU activation block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.tconv(x)))


class ConvUNet(nn.Module):
    """Classical Continuous U-Net for OCT Despeckle Filtering.

    Mirrors the exact topology of SpikingUNet without temporal unrolling (T=1),
    eliminating surrogate gradient approximations and providing high-fidelity
    continuous edge reconstruction with ~4x lower compute on ARM CPUs.

    Args:
        in_channels: Number of input channels (1 for grayscale OCT).
        base_ch: Base channel count (32).
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, base_ch, stride=2)          # [B, base, H/2, W/2]
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)         # [B, base*2, H/4, W/4]
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=1)     # [B, base*4, H/4, W/4] bottleneck

        # --- Decoder ---
        self.dec1 = TransConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)  # [B, base*2, H/2, W/2]
        self.dec2 = TransConvBlock(base_ch * 2 + base_ch, base_ch)          # [B, base, H, W]

        # --- Readout ---
        self.out_conv = nn.Conv2d(base_ch, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Decoder with skip connections
        d1_in = torch.cat([e3, e2], dim=1)
        d1 = self.dec1(d1_in)

        d2_in = torch.cat([d1, e1], dim=1)
        d2 = self.dec2(d2_in)

        return self.sigmoid(self.out_conv(d2))


# ---------------------------------------------------------------------------
# Neuromorphic Spiking Blocks (Baseline)
# ---------------------------------------------------------------------------

class SpikingConvBlock(nn.Module):
    """Conv2d → BatchNorm → LIF neuron block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, beta: float = 0.85):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def init_mem(self):
        return self.lif.init_leaky()

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        cur = self.bn(self.conv(x))
        spk, mem = self.lif(cur, mem)
        return spk, mem


class SpikingTransConvBlock(nn.Module):
    """ConvTranspose2d → BatchNorm → LIF neuron block."""

    def __init__(self, in_ch: int, out_ch: int, beta: float = 0.85):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.tconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def init_mem(self):
        return self.lif.init_leaky()

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        cur = self.bn(self.tconv(x))
        spk, mem = self.lif(cur, mem)
        return spk, mem


class SpikingUNet(nn.Module):
    """Spiking U-Net (S-UNet) for OCT despeckle filtering.

    Encoder:  3 spiking conv blocks with spatial downsampling (stride=2).
    Decoder:  2 spiking transposed-conv blocks with skip connections,
              plus a final 1x1 conv whose V_mem is the continuous output.
    Readout:  Raw membrane potential at the final timestep T.

    Args:
        in_channels: Number of input channels (1 for grayscale OCT).
        base_ch: Base channel count (32).
        num_steps: Number of spiking timesteps T (default: 4).
        beta: LIF membrane decay rate (default: 0.85).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 32,
        num_steps: int = 4,
        beta: float = 0.85,
    ):
        super().__init__()
        self.num_steps = num_steps

        # --- Encoder ---
        self.enc1 = SpikingConvBlock(in_channels, base_ch, stride=2, beta=beta)
        self.enc2 = SpikingConvBlock(base_ch, base_ch * 2, stride=2, beta=beta)
        self.enc3 = SpikingConvBlock(base_ch * 2, base_ch * 4, stride=1, beta=beta)

        # --- Decoder ---
        self.dec1 = SpikingTransConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2, beta=beta)
        self.dec2 = SpikingTransConvBlock(base_ch * 2 + base_ch, base_ch, beta=beta)

        # Final readout conv — V_mem accumulation
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.out_conv = nn.Conv2d(base_ch, in_channels, kernel_size=1)
        self.out_lif = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=False, output=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem_e1 = self.enc1.init_mem()
        mem_e2 = self.enc2.init_mem()
        mem_e3 = self.enc3.init_mem()
        mem_d1 = self.dec1.init_mem()
        mem_d2 = self.dec2.init_mem()
        mem_out = self.out_lif.init_leaky()

        for t in range(self.num_steps):
            # Encoder
            spk_e1, mem_e1 = self.enc1(x, mem_e1)
            spk_e2, mem_e2 = self.enc2(spk_e1, mem_e2)
            spk_e3, mem_e3 = self.enc3(spk_e2, mem_e3)

            # Decoder with skip connections
            d1_in = torch.cat([spk_e3, spk_e2], dim=1)
            spk_d1, mem_d1 = self.dec1(d1_in, mem_d1)

            d2_in = torch.cat([spk_d1, spk_e1], dim=1)
            spk_d2, mem_d2 = self.dec2(d2_in, mem_d2)

            # Readout: accumulate V_mem
            cur_out = self.out_conv(spk_d2)
            spk_out, mem_out = self.out_lif(cur_out, mem_out)

        return torch.sigmoid(mem_out)

    def forward_unrolled(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


# ---------------------------------------------------------------------------
# Factory Helper
# ---------------------------------------------------------------------------

def get_despeckler(
    model_type: str = "cnn",
    in_channels: int = 1,
    base_ch: int = 32,
    num_steps: int = 4,
    beta: float = 0.85,
) -> nn.Module:
    """Factory helper returning either ConvUNet (default) or SpikingUNet (baseline)."""
    if model_type.lower() == "cnn":
        return ConvUNet(in_channels=in_channels, base_ch=base_ch)
    elif model_type.lower() == "snn":
        return SpikingUNet(
            in_channels=in_channels,
            base_ch=base_ch,
            num_steps=num_steps,
            beta=beta,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'cnn' or 'snn'.")
