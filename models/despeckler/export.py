"""
export.py
=========
PyTorch → ONNX export for Stage 1 Despeckler (ConvUNet).

Produces a clean, single-pass continuous computation graph (T=1) compatible with
ONNX Runtime CPU / ARM for embedded point-of-care deployment.
"""

import os
import argparse
import torch

try:
    from model import ConvUNet
except ImportError:
    from models.despeckler.model import ConvUNet


def export_onnx(args):
    device = torch.device("cpu")  # Export on CPU for portability

    model = ConvUNet(
        in_channels=1,
        base_ch=args.base_ch,
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
    output_path = args.output or os.path.join("exports", "unet_despeckler.onnx")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"[*] Exporting ConvUNet model to ONNX: {output_path}")

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
    print(f"[+] Successfully exported ConvUNet to: {output_path} ({size_kb:.1f} KB)")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Export ConvUNet Despeckler to ONNX")
    p.add_argument("--checkpoint", type=str, default="exports/despeckler_cnn_best.pt",
                    help="Path to trained .pt checkpoint")
    p.add_argument("--output", type=str, default="exports/unet_despeckler.onnx",
                    help="Output ONNX file path")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
