"""Plot each fixed SHA gate along selected linear input interpolations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from sha256_nn_mpmath import DEFAULT_WEIGHTS, FractionalSHA256MP
from sha256_nn_mpmath import _Tape


def evaluate(model, gate_name, inputs):
    tape = _Tape(model.ctx, recording=False)
    nodes = [tape.push(model.ctx.mpf(value)) for value in inputs]
    output_nodes = model.gates[gate_name](tape, *nodes)
    return [tape.values[node] for node in output_nodes]


def interpolate(start, end, t):
    return [a + t * (b - a) for a, b in zip(start, end)]


def add_path(ax, model, gate_name, start, end, output_names, ts, linestyle="-",
             path_name=None, output_indices=None):
    start_out = evaluate(model, gate_name, start)
    end_out = evaluate(model, gate_name, end)
    if output_indices is None:
        output_indices = range(len(output_names))
    curves = [[] for _ in output_names]
    chords = [[] for _ in output_names]
    for t in ts:
        output = evaluate(model, gate_name, interpolate(start, end, t))
        for index, output_index in enumerate(output_indices):
            curves[index].append(float(output[output_index]))
            chords[index].append(float((1-t)*start_out[output_index] + t*end_out[output_index]))
    for index, name in enumerate(output_names):
        label = f"{path_name}: {name}" if path_name else name
        line, = ax.plot([float(t) for t in ts], curves[index], linestyle=linestyle,
                        linewidth=2, label=f"gate {label}")
        ax.plot([float(t) for t in ts], chords[index], "--", color=line.get_color(),
                alpha=0.65, label=f"endpoint chord {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=237,
                        help="mpmath significand bits; defaults to binary256 precision")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--samples", type=int, default=501)
    parser.add_argument("--output", type=Path, default=Path("gate_interpolation_237.png"))
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    model = FractionalSHA256MP("compression", args.precision, args.weights)
    ctx = model.ctx
    ts = [ctx.mpf(i) / (args.samples - 1) for i in range(args.samples)]
    fig = plt.figure(figsize=(13, 13), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    axes.append(fig.add_subplot(grid[2, :]))

    axes[0].set_title("XOR and AND   (0, 0) → (1, 1)")
    add_path(axes[0], model, "xor_and", (0, 0), (1, 1), ("XOR", "AND"), ts)

    axes[1].set_title("XOR3   (0, 0, 0) → (1, 1, 1)")
    add_path(axes[1], model, "xor3", (0, 0, 0), (1, 1, 1), ("XOR3",), ts)

    axes[2].set_title("Choose   (0, 0, 0) → (1, 1, 1)")
    add_path(axes[2], model, "choose", (0, 0, 0), (1, 1, 1), ("Ch",), ts)

    axes[3].set_title("Majority   (0, 0, 0) → (1, 1, 1)")
    add_path(axes[3], model, "majority", (0, 0, 0), (1, 1, 1), ("Maj",), ts)

    axes[4].set_title("Carry prefix   separate generate and propagate paths")
    add_path(axes[4], model, "prefix", (0, 0, 0, 0), (0, 1, 1, 0), ("G",), ts,
             path_name="generate: (0,0,0,0)→(0,1,1,0)", output_indices=(0,))
    add_path(axes[4], model, "prefix", (0, 0, 0, 0), (0, 1, 0, 1), ("P",), ts,
             linestyle=":", path_name="propagate: (0,0,0,0)→(0,1,0,1)", output_indices=(1,))
    for ax in axes[:5]:
        ax.set_xlabel("linear input interpolation t")
        ax.set_ylabel("gate output")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Fixed SHA-256 gates on fractional inputs ({args.precision}-bit mpmath)",
                 fontsize=15)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
