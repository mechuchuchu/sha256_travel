"""Plot target-digest MSE along a line between two 64-byte messages."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from sha256_nn import bits_to_bytes, bytes_to_bits
from smooth_nn.network import SmoothSHA256NN
from smooth_nn.optimize import DEFAULT_MESSAGE, parse_hex_bytes


HERE = Path(__file__).resolve().parent


def save_plot(ts, losses, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    axes[0].plot(ts, losses, color="#2563eb", linewidth=1.7, marker=".", markersize=4)
    axes[0].set(title="Smooth SHA-256: input interpolation MSE (smoothness 0.009)",
                ylabel="MSE", xlim=(0, 1))
    axes[1].plot(ts[1:-1], losses[1:-1], color="#2563eb", linewidth=1.2,
                 marker=".", markersize=4)
    axes[1].set(title="Interior samples, enlarged", xlabel="Interpolation t: (1 - t) × start + t × end",
                ylabel="MSE", xlim=(0, 1))
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-hex", help="starting 64-byte message")
    parser.add_argument("--end-hex", help="ending 64-byte message")
    parser.add_argument("--target-hex", help="target 32-byte digest; defaults to SHA-256 of the start message")
    parser.add_argument("--seed", type=int, default=72631, help="seed for the default end message")
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "smooth_gates_sweep_0p009.pt")
    parser.add_argument("--output", type=Path, default=HERE / "input_interpolation_0p009.png")
    args = parser.parse_args()
    if args.samples < 3 or args.batch_size < 1:
        parser.error("samples must be at least 3 and batch-size must be positive")
    try:
        start = parse_hex_bytes(args.start_hex, expected_length=64, label="start message") if args.start_hex else DEFAULT_MESSAGE
        end = parse_hex_bytes(args.end_hex, expected_length=64, label="end message") if args.end_hex else random.Random(args.seed).randbytes(64)
        target = parse_hex_bytes(args.target_hex, expected_length=32, label="target digest") if args.target_hex else hashlib.sha256(start).digest()
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    torch.set_num_threads(1)
    model = SmoothSHA256NN(checkpoint=args.checkpoint, dtype=torch.float64)
    x0, x1 = bytes_to_bits(start).double(), bytes_to_bits(end).double()
    target_bits = bytes_to_bits(target).double()
    ts = torch.linspace(0, 1, args.samples, dtype=torch.float64)
    losses = []
    with torch.inference_mode():
        for offset in range(0, args.samples, args.batch_size):
            batch_t = ts[offset:offset + args.batch_size]
            x = x0 + batch_t[:, None] * (x1 - x0)
            output = model(x)
            loss = (output - target_bits).square().mean(dim=-1)
            if not torch.isfinite(loss).all():
                raise RuntimeError("non-finite loss during interpolation")
            losses.extend(loss.tolist())
            if offset == 0 or offset + args.batch_size >= args.samples or offset % (args.batch_size * 10) == 0:
                print(f"evaluated {min(offset + args.batch_size, args.samples)}/{args.samples}", flush=True)
        start_digest = bits_to_bytes(model(x0))
        end_digest = bits_to_bytes(model(x1))
    if start_digest != hashlib.sha256(start).digest() or end_digest != hashlib.sha256(end).digest():
        raise RuntimeError("smooth model does not match hashlib at an endpoint")

    output_path = args.output
    data_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "start_message": start.hex(), "end_message": end.hex(),
        "target_digest": target.hex(),
        "start_sha256": start_digest.hex(), "end_sha256": end_digest.hex(),
        "different_input_bits": int((x0 != x1).sum().item()),
        "samples": [{"t": t, "mse": loss} for t, loss in zip(ts.tolist(), losses)],
    }
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    save_plot(ts.tolist(), losses, output_path)
    print(f"start message: {start.hex()}\nend message:   {end.hex()}\ntarget digest: {target.hex()}")
    print(f"different input bits: {payload['different_input_bits']}")
    print(f"endpoint MSE: {losses[0]:.9g}, {losses[-1]:.9g}")
    print(f"saved plot: {output_path.resolve()}\nsaved data: {data_path.resolve()}")


if __name__ == "__main__":
    main()
