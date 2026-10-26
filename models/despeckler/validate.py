"""
validate.py
===========
PyTorch ↔ ONNX Runtime equivalence validation and latency benchmarking for:
1. ConvUNet (CNN Default)
2. SpikingUNet (SNN Baseline)

Asserts that ONNX outputs match native PyTorch within 10⁻⁴ tolerance and measures
mean edge inference latency against the ≥15 FPS (≤66.6 ms) target.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import onnxruntime as ort

try:
    from model import get_despeckler
except ImportError:
    from models.despeckler.model import get_despeckler


def validate(args):
    device = torch.device("cpu")

    # --- Load PyTorch model ---
    model = get_despeckler(
        model_type=args.model_type,
        in_channels=1,
        base_ch=args.base_ch,
        num_steps=args.num_steps,
        beta=args.beta,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded PyTorch checkpoint: {args.checkpoint}")

    model.eval()

    # --- Load ONNX model ---
    if not os.path.exists(args.onnx):
        print(f"[!] Error: ONNX file not found at {args.onnx}")
        return 1

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[+] Loaded ONNX model: {args.onnx}")

    # --- Run Numerical Parity Comparison ---
    num_passed = 0
    max_diff_seen = 0.0
    rng = np.random.default_rng(args.seed)

    print(f"[*] Testing PyTorch vs ONNX Runtime numerical equivalence on {args.num_samples} samples...")
    for i in range(args.num_samples):
        x_np = rng.random((1, 1, args.height, args.width)).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        with torch.no_grad():
            y_pt = model(x_pt).numpy()

        y_ort = session.run([output_name], {input_name: x_np})[0]

        max_diff = float(np.max(np.abs(y_pt - y_ort)))
        max_diff_seen = max(max_diff_seen, max_diff)

        if max_diff <= args.tolerance:
            num_passed += 1
        else:
            print(f"    FAIL sample {i}: max_diff={max_diff:.6e} > tolerance={args.tolerance:.0e}")

    print(f"\n[*] Numerical Parity: {num_passed}/{args.num_samples} passed (tolerance={args.tolerance:.0e})")
    print(f"    Max absolute diff: {max_diff_seen:.6e}")

    # --- Latency Benchmarking ---
    print(f"[*] Running latency benchmark ({args.benchmark_iters} iterations on CPU)...")
    bench_input = rng.random((1, 1, args.height, args.width)).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run([output_name], {input_name: bench_input})

    latencies = []
    for _ in range(args.benchmark_iters):
        t0 = time.perf_counter()
        session.run([output_name], {input_name: bench_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    fps = 1000.0 / mean_lat

    print(f"[+] {args.model_type.upper()} ONNX Runtime CPU Latency ({args.height}x{args.width}):")
    print(f"    - Mean: {mean_lat:.2f} ms")
    print(f"    - P95:  {p95_lat:.2f} ms")
    print(f"    - Throughput: {fps:.1f} FPS (Target: ≥15.0 FPS)")

    if fps >= 15.0:
        print(f"    - Status: [PASS] Exceeds 15 FPS edge budget.")
    else:
        print(f"    - Status: [WARN] Below 15 FPS on this host.")

    if num_passed == args.num_samples:
        print(f"[+] VALIDATION PASSED for {args.model_type.upper()}!")
        return 0
    else:
        print(f"[!] VALIDATION FAILED for {args.model_type.upper()}: Numerical divergence.")
        return 1


def parse_args():
    p = argparse.ArgumentParser(description="Validate PyTorch ↔ ONNX equivalence and benchmark latency")
    p.add_argument("--model-type", type=str, choices=["cnn", "snn"], default="cnn")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--onnx", type=str, default="exports/unet_despeckler.onnx")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--benchmark-iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(validate(parse_args()))
