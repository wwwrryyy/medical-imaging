"""
export.py
=========
PyTorch → ONNX export for the Spiking Despeckler.

Exports the SpikingUNet with static temporal unrolling: the T-step spiking
loop is traced into the computation graph so the ONNX file contains no
dynamic control flow and can be run by standard C++ ONNX Runtime.
"""

import os
import argparse

import torch

from model import SpikingUNet


def export_onnx(args):
    device = torch.device("cpu")  # Export on CPU for portability

    model = SpikingUNet(
        in_channels=1,
        base_ch=args.base_ch,
        num_steps=args.num_steps,
        beta=args.beta,
    ).to(device)

    # Load trained weights
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded checkpoint: {args.checkpoint}")

    model.eval()

    dummy = torch.randn(1, 1, args.height, args.width, device=device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"[*] Exporting ONNX with T={args.num_steps} static timesteps...")
    torch.onnx.export(
        model,
        dummy,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["corrupted_frame"],
        output_names=["clean_frame"],
        dynamic_axes={
            "corrupted_frame": {0: "batch_size"},
            "clean_frame": {0: "batch_size"},
        },
        dynamo=False,
    )

    size_kb = os.path.getsize(args.output) / 1024
    print(f"[+] Exported: {args.output} ({size_kb:.1f} KB)")
    print(f"    T={args.num_steps} timesteps statically unrolled into graph.")


def parse_args():
    p = argparse.ArgumentParser(description="Export Spiking Despeckler to ONNX")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to trained .pt checkpoint.")
    p.add_argument("--output", type=str, default="../../exports/despeckler.onnx",
                    help="Output ONNX file path.")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
