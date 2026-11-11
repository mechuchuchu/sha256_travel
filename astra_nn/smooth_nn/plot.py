"""Compare source ReLU gates with the trained smooth surrogates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from sha256_nn import SHA256NN
from smooth_nn import GATE_NAMES, build_smooth_gate, gate_paths, source_gate_map


HERE = Path(__file__).resolve().parent


def plot_path(ax, name, source, smooth, start, end, path_label, samples):
    dtype = torch.float64
    ts = torch.linspace(0, 1, samples, dtype=dtype)
    a = torch.tensor(start, dtype=dtype)
    b = torch.tensor(end, dtype=dtype)
    points = a + ts[:, None] * (b - a)
    with torch.no_grad():
        reference = source(*points.unbind(-1)).reshape(samples, -1)
        prediction = smooth(points)
    output_labels = ("XOR", "AND") if name == "xor_and" else \
                    ("G", "P") if name == "prefix" else \
                    ("XOR3",) if name == "xor3" else \
                    ("Ch",) if name == "choose" else ("Maj",)
    for index, output_name in enumerate(output_labels):
        ax.plot(ts.numpy(), reference[:, index].numpy(), linestyle="--", linewidth=1.7,
                label=f"source {output_name} ({path_label})")
        ax.plot(ts.numpy(), prediction[:, index].numpy(), linewidth=2,
                label=f"smooth {output_name} ({path_label})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "smooth_gates.pt")
    parser.add_argument("--samples", type=int, default=501)
    parser.add_argument("--output", type=Path, default=HERE / "smooth_gate_comparison.png")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        parser.error("unsupported smooth-gate checkpoint")

    source_model = SHA256NN().double().eval()
    sources = source_gate_map(source_model)
    smooth_models = {}
    for name in GATE_NAMES:
        smooth = build_smooth_gate(name, sources[name],
                                   width=checkpoint["width"], depth=checkpoint["depth"])
        smooth.load_state_dict(checkpoint["state_dicts"][name], strict=True)
        smooth.cache_anchor_correction()
        smooth.eval()
        smooth_models[name] = smooth

    fig = plt.figure(figsize=(13, 13), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    axes.append(fig.add_subplot(grid[2, :]))
    titles = {
        "xor_and": "XOR and AND",
        "xor3": "XOR3",
        "choose": "Choose",
        "majority": "Majority",
        "prefix": "Carry prefix",
    }
    for ax, name in zip(axes, GATE_NAMES):
        for start, end, label in gate_paths(name):
            plot_path(ax, name, sources[name], smooth_models[name], start, end, label,
                      args.samples)
        ax.set_title(titles[name])
        ax.set_xlabel("linear input interpolation t")
        ax.set_ylabel("gate output")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Source ReLU gate vs smooth anchored MLP", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
