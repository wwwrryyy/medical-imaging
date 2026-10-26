"""
train_autoencoder.py
====================
Training and ONNX export pipeline for the Stage 2 ConvAutoencoder tissue baseline.
Learns healthy normal tissue representation and exports to ONNX for the C++ AnomalyCalculus engine.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    from autoencoder import ConvAutoencoder
except ImportError:
    from models.anomaly_detector.autoencoder import ConvAutoencoder


def generate_synthetic_tissue_batch(batch_size=16, height=128, width=128, in_channels=1):
    """Generates synthetic healthy baseline tissue slices for training."""
    batch = []
    for _ in range(batch_size):
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        
        freq = np.random.uniform(2.0, 6.0)
        tissue = 0.5 + 0.3 * np.sin(freq * yy) * np.cos(1.0 * xx)
        texture = np.random.normal(0.0, 0.03, (height, width))
        frame = np.clip(tissue + texture, 0.0, 1.0).astype(np.float32)
        
        frame = np.expand_dims(frame, axis=0)
        batch.append(frame)
        
    return torch.tensor(np.array(batch), dtype=torch.float32)


def train_and_export_autoencoder():
    print("[+] Starting ConvAutoencoder Training Pipeline...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[*] Training Device: {device}")
    
    in_channels = 1
    h, w = 128, 128
    epochs = 10
    batch_size = 16
    
    model = ConvAutoencoder(in_channels=in_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 10
        for _ in range(num_batches):
            inputs = generate_synthetic_tissue_batch(batch_size=batch_size, height=h, width=w, in_channels=in_channels).to(device)
            noise = torch.randn_like(inputs) * 0.03
            noisy_inputs = torch.clamp(inputs + noise, 0.0, 1.0)
            
            optimizer.zero_grad()
            reconstructed = model(noisy_inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / num_batches
        if epoch % 2 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch}/{epochs}] - Loss (MSE): {avg_loss:.6f}")

    print("[+] ConvAutoencoder Training Complete!")

    # --- ONNX Export ---
    model.eval()
    cpu_model = model.to("cpu")
    
    output_dir = "exports"
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "conv_autoencoder_tissue_baseline.onnx")
    
    dummy_input = torch.randn(1, in_channels, h, w, device="cpu")
    
    print(f"[*] Exporting ConvAutoencoder to ONNX: {onnx_path}")
    torch.onnx.export(
        cpu_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
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
    return onnx_path


if __name__ == "__main__":
    train_and_export_autoencoder()
