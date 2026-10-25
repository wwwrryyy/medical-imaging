"""
train_snn_autoencoder.py
========================
Synthetic terrain dataset generation, unsupervised training of the SNN Autoencoder,
quantization, and ONNX export pipeline for edge device deployment (e.g., NVIDIA Jetson Nano).
"""

import os
import sys
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch.nn as nn
import torch.optim as optim
import numpy as np
from SNN_Architecture import SNNAutoencoder, ONNXExportableSNN


def generate_synthetic_terrain_batch(batch_size=16, height=128, width=128, in_channels=1):
    """
    Generates synthetic multispectral terrain baselines with textures,
    gradients, and mild noise for unsupervised training.
    """
    batch = []
    for _ in range(batch_size):
        # Base terrain gradient (e.g. topography/elevation background)
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        
        # Synthetic terrain features: Perlin-like low frequency sine waves
        freq = np.random.uniform(2.0, 8.0)
        terrain = 0.5 + 0.3 * np.sin(freq * xx) * np.cos(freq * yy)
        
        # Add micro terrain texture (dirt, rocks, foliage baselines)
        texture = np.random.normal(0.0, 0.05, (height, width))
        frame = np.clip(terrain + texture, 0.0, 1.0).astype(np.float32)
        
        if in_channels == 3:
            frame = np.stack([frame, frame * 0.9, frame * 1.1], axis=0)
        else:
            frame = np.expand_dims(frame, axis=0)
            
        batch.append(frame)
        
    return torch.tensor(np.array(batch), dtype=torch.float32)


def train_and_export_snn():
    print("[+] Starting Neuromorphic SNN Autoencoder Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")
    
    # Model parameters
    in_channels = 1
    h, w = 128, 128
    epochs = 15
    batch_size = 16
    
    model = SNNAutoencoder(in_channels=in_channels, beta=0.85, num_steps=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 10
        for _ in range(num_batches):
            inputs = generate_synthetic_terrain_batch(batch_size=batch_size, height=h, width=w, in_channels=in_channels).to(device)
            
            # Add synthetic smoke/dust noise to inputs during training
            noise = torch.randn_like(inputs) * 0.05
            noisy_inputs = torch.clamp(inputs + noise, 0.0, 1.0)
            
            optimizer.zero_grad()
            reconstructed = model(noisy_inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / num_batches
        if epoch % 3 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch}/{epochs}] - Loss (MSE): {avg_loss:.6f}")

    print("[+] SNN Autoencoder Training Complete!")

    # --- Model Quantization & ONNX Export ---
    model.eval()
    onnx_wrapper = ONNXExportableSNN(model).to(device)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(output_dir, "snn_autoencoder.onnx")
    
    dummy_input = torch.randn(1, in_channels, h, w, device=device)
    
    print(f"[*] Exporting SNN Autoencoder to ONNX format at: {onnx_path}")
    torch.onnx.export(
        onnx_wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_frame'],
        output_names=['reconstructed_baseline'],
        dynamic_axes={
            'input_frame': {0: 'batch_size', 2: 'height', 3: 'width'},
            'reconstructed_baseline': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        dynamo=False
    )
    print(f"[+] Successfully exported ONNX model! File size: {os.path.getsize(onnx_path) / 1024:.2f} KB")

    # Quantization note
    print("[*] Model exported ready for INT8/FP16 TensorRT or ONNX Runtime quantization on Edge Hardware.")
    return onnx_path


if __name__ == "__main__":
    train_and_export_snn()
