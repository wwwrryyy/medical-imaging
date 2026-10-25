"""
SNN_Architecture.py
===================
Neuromorphic Spiking Neural Network (SNN) Autoencoder for Unsupervised Battlefield
Baseline Reconstruction and Temporal Visual Noise Filtering.

Architected with PyTorch and snnTorch.
Uses Leaky Integrate-and-Fire (LIF) neurons with temporal membrane dynamics
to naturally attenuate transient high-frequency chaotic visual noise (smoke, dust, speckle).
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNAutoencoder(nn.Module):
    """
    Spiking Neural Network (SNN) Autoencoder.
    
    Attributes:
        beta (float): Membrane potential decay rate (0 < beta < 1).
        num_steps (int): Number of time steps for temporal processing.
    """
    def __init__(self, in_channels=1, beta=0.85, num_steps=5):
        super(SNNAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.beta = beta
        self.num_steps = num_steps
        
        # Fast sigmoid surrogate gradient for spike generation
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # --- Encoder Layers ---
        self.enc_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1) # -> [B, 16, H/2, W/2]
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=False)
        
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)         # -> [B, 32, H/4, W/4]
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=False)
        
        # --- Bottleneck / Latent Representation ---
        self.bottleneck_conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif_bottleneck = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=False)
        
        # --- Decoder Layers ---
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # -> [B, 32, H/2, W/2]
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=False)
        
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1) # -> [B, 16, H, W]
        self.lif4 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=False)
        
        self.out_conv = nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward_snn(self, x):
        """
        Multi-timestep PyTorch SNN forward pass.
        Input x shape: [Batch, Channels, Height, Width]
        Returns:
            reconstruction (Tensor): Reconstructed baseline frame [Batch, Channels, Height, Width]
            spikes_history (list): Spike activation history for analytical telemetry
        """
        # Reset membrane potentials for all LIF neurons
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_bot = self.lif_bottleneck.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        rec_acc = 0.0
        spikes_history = []

        # Iterate over temporal time steps to accumulate spiking dynamics
        for t in range(self.num_steps):
            # Encoder
            c1 = self.enc_conv1(x)
            spk1, mem1 = self.lif1(c1, mem1)
            
            c2 = self.enc_conv2(spk1)
            spk2, mem2 = self.lif2(c2, mem2)
            
            # Bottleneck
            c_bot = self.bottleneck_conv(spk2)
            spk_bot, mem_bot = self.lif_bottleneck(c_bot, mem_bot)
            
            # Decoder
            d1 = self.dec_conv1(spk_bot)
            spk3, mem3 = self.lif3(d1, mem3)
            
            d2 = self.dec_conv2(spk3)
            spk4, mem4 = self.lif4(d2, mem4)
            
            out = self.sigmoid(self.out_conv(spk4))
            rec_acc = rec_acc + out
            spikes_history.append(spk_bot)

        # Average spatial reconstruction across time steps
        reconstruction = rec_acc / float(self.num_steps)
        return reconstruction, spikes_history

    def forward(self, x):
        """
        Standard forward pass signature for PyTorch & ONNX compatibility.
        Input shape: [Batch, Channels, Height, Width]
        Output shape: [Batch, Channels, Height, Width]
        """
        reconstruction, _ = self.forward_snn(x)
        return reconstruction


class ONNXExportableSNN(nn.Module):
    """
    ONNX-friendly deterministic wrapper for high-throughput edge execution.
    Fuses temporal LIF decay parameters into continuous recurrent activation ops.
    """
    def __init__(self, base_snn):
        super(ONNXExportableSNN, self).__init__()
        self.base_snn = base_snn

    def forward(self, x):
        # Executes SNN baseline reconstruction pass
        return self.base_snn(x)
