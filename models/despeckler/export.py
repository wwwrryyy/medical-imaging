"""
export.py
=========
PyTorch → ONNX export for Stage 1 Despeckler:
1. ConvUNet (CNN Default, T=1)
2. SpikingUNet (SNN Baseline, T=4 static temporal unrolling)

Produces production-ready ONNX models compatible with ONNX Runtime CPU / ARM.
"""

import os
import argparse
import torch

try:
    from model import get_despeckler
except ImportError:
    from models.despeckler.model import get_despeckler


def export_onnx(args):
    device = torch.device("cpu")  # Export on CPU for embedded portability

    model = get_despeckler(
        model_type=args.model_type,
        in_channels=1,
        base_ch=args.base_ch,
        num_steps=args.num_steps,
        beta=args.beta,
    ).to(device)

    # Load trained weights if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded weights checkpoint: {args.checkpoint}")
    elif args.checkpoint:
        print(f"[!] Warning: checkpoint {args.checkpoint} not found. Exporting initialized architecture.")

    model.eval()

    dummy = torch.randn(1, 1, args.height, args.width, device=device)
    output_path = args.output
    if not output_path:
        default_name = "unet_despeckler.onnx" if args.model_type == "cnn" else f"sunet_despeckler_static_t{args.num_steps}.onnx"
        output_path = os.path.join("exports", default_name)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"[*] Exporting {args.model_type.upper()} model to ONNX: {output_path}")
    if args.model_type == "snn":
        print(f"    Static temporal unrolling with T={args.num_steps} timesteps...")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["corrupted_frame"],
        output_names=["clean_frame"],
        dynamic_axes={
            "corrupted_frame": {0: "batch_size"},
            "clean_frame": {0: "batch_size"},
        },
        dynamo=False,
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[+] Successfully exported {args.model_type.upper()} to: {output_path} ({size_kb:.1f} KB)")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Export Despeckler to ONNX")
    p.add_argument("--model-type", type=str, choices=["cnn", "snn"], default="cnn",
                    help="Model architecture ('cnn' default, 'snn' baseline)")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to trained .pt checkpoint")
    p.add_argument("--output", type=str, default=None,
                    help="Output ONNX file path (defaults to exports/unet_despeckler.onnx or exports/sunet_despeckler_static_t4.onnx)")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=4,
                    help="Number of spiking timesteps T (SNN only)")
    p.add_argument("--beta", type=float, default=0.85,
                    help="LIF membrane decay rate (SNN only)")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
