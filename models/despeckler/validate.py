"""
validate.py
============
PyTorch ↔ ONNX Runtime equivalence validation for the Spiking Despeckler.

Asserts that the ONNX model produces outputs within 10⁻⁴ tolerance of the
native PyTorch model, as required by the PRD.
"""

import argparse
import sys

import numpy as np
import torch
import onnxruntime as ort

from model import SpikingUNet


def validate(args):
    device = torch.device("cpu")

    # --- Load PyTorch model ---
    model = SpikingUNet(
        in_channels=1,
        base_ch=args.base_ch,
        num_steps=args.num_steps,
        beta=args.beta,
    ).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded checkpoint: {args.checkpoint}")

    model.eval()

    # --- Load ONNX model ---
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[+] Loaded ONNX model: {args.onnx}")

    # --- Run comparison ---
    num_passed = 0
    max_diff_seen = 0.0
    rng = np.random.default_rng(args.seed)

    for i in range(args.num_samples):
        # Random test input
        x_np = rng.random((1, 1, args.height, args.width)).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        # PyTorch inference
        with torch.no_grad():
            y_pt = model(x_pt).numpy()

        # ONNX Runtime inference
        y_ort = session.run([output_name], {input_name: x_np})[0]

        # Compare
        max_diff = np.max(np.abs(y_pt - y_ort))
        max_diff_seen = max(max_diff_seen, max_diff)

        if max_diff <= args.tolerance:
            num_passed += 1
        else:
            print(f"    FAIL sample {i}: max_diff={max_diff:.6e} > tolerance={args.tolerance:.0e}")

    # --- Report ---
    print(f"\n[*] Results: {num_passed}/{args.num_samples} passed "
          f"(tolerance={args.tolerance:.0e})")
    print(f"    Max absolute difference seen: {max_diff_seen:.6e}")

    if num_passed == args.num_samples:
        print("[+] VALIDATION PASSED: PyTorch and ONNX outputs are equivalent.")
        return 0
    else:
        print("[!] VALIDATION FAILED: Outputs diverge beyond tolerance.")
        return 1


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate PyTorch ↔ ONNX equivalence for Spiking Despeckler"
    )
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to trained .pt checkpoint.")
    p.add_argument("--onnx", type=str, default="../../exports/despeckler.onnx",
                    help="Path to exported ONNX model.")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--tolerance", type=float, default=1e-4,
                    help="Max allowed absolute difference.")
    p.add_argument("--num-samples", type=int, default=10,
                    help="Number of random test inputs.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(validate(parse_args()))
