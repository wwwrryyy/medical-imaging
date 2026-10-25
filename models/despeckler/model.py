"""
model.py
========
Spiking U-Net (S-UNet) for SAR despeckle filtering.

Architecture: Encoder-decoder with skip connections, LIF neurons at every block,
and raw V_mem readout at the final layer. Designed for static temporal unrolling
during ONNX export.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


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
        """Returns (spike, mem)."""
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
        """Returns (spike, mem)."""
        cur = self.bn(self.tconv(x))
        spk, mem = self.lif(cur, mem)
        return spk, mem


class SpikingUNet(nn.Module):
    """Spiking U-Net (S-UNet) for SAR despeckle filtering.

    Encoder:  3 spiking conv blocks with spatial downsampling (stride=2).
    Decoder:  2 spiking transposed-conv blocks with skip connections,
              plus a final 1x1 conv whose V_mem is the continuous output.
    Readout:  Raw membrane potential at the final timestep T (no spike count).

    Args:
        in_channels: Number of input channels (1 for grayscale SAR).
        base_ch: Base channel count (doubled at each encoder stage).
        num_steps: Number of spiking timesteps T.
        beta: LIF membrane decay rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 32,
        num_steps: int = 8,
        beta: float = 0.85,
    ):
        super().__init__()
        self.num_steps = num_steps

        # --- Encoder ---
        self.enc1 = SpikingConvBlock(in_channels, base_ch, stride=2, beta=beta)
        self.enc2 = SpikingConvBlock(base_ch, base_ch * 2, stride=2, beta=beta)
        self.enc3 = SpikingConvBlock(base_ch * 2, base_ch * 4, stride=1, beta=beta)  # bottleneck

        # --- Decoder ---
        # dec1 receives bottleneck + skip from enc2 via concatenation
        self.dec1 = SpikingTransConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2, beta=beta)
        # dec2 receives dec1 output + skip from enc1 via concatenation
        self.dec2 = SpikingTransConvBlock(base_ch * 2 + base_ch, base_ch, beta=beta)

        # Final readout conv — we accumulate V_mem here, no LIF
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.out_conv = nn.Conv2d(base_ch, in_channels, kernel_size=1)
        self.out_lif = snn.Leaky(
            beta=beta, spike_grad=spike_grad, init_hidden=False, output=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal integration.

        Args:
            x: Input tensor [B, C, H, W]. The same frame is presented
               at every timestep (static image denoising).

        Returns:
            Denoised output [B, C, H, W] from final-timestep V_mem.
        """
        # Initialize all membrane potentials
        mem_e1 = self.enc1.init_mem()
        mem_e2 = self.enc2.init_mem()
        mem_e3 = self.enc3.init_mem()
        mem_d1 = self.dec1.init_mem()
        mem_d2 = self.dec2.init_mem()
        mem_out = self.out_lif.init_leaky()

        for t in range(self.num_steps):
            # Encoder
            spk_e1, mem_e1 = self.enc1(x, mem_e1)        # [B, base, H/2, W/2]
            spk_e2, mem_e2 = self.enc2(spk_e1, mem_e2)   # [B, base*2, H/4, W/4]
            spk_e3, mem_e3 = self.enc3(spk_e2, mem_e3)   # [B, base*4, H/4, W/4]

            # Decoder with skip connections (concatenation)
            d1_in = torch.cat([spk_e3, spk_e2], dim=1)   # [B, base*6, H/4, W/4]
            spk_d1, mem_d1 = self.dec1(d1_in, mem_d1)     # [B, base*2, H/2, W/2]

            d2_in = torch.cat([spk_d1, spk_e1], dim=1)   # [B, base*3, H/2, W/2]
            spk_d2, mem_d2 = self.dec2(d2_in, mem_d2)     # [B, base, H, W]

            # Readout: accumulate V_mem
            cur_out = self.out_conv(spk_d2)
            spk_out, mem_out = self.out_lif(cur_out, mem_out)

        # Return V_mem at final timestep — continuous pixel output
        return torch.sigmoid(mem_out)

    def forward_unrolled(self, x: torch.Tensor) -> torch.Tensor:
        """Identical to forward() but written without a Python loop.

        Used for ONNX export: the temporal dimension is statically unrolled
        so the graph contains no dynamic control flow.
        """
        return self.forward(x)
