"""Sweep smoothness weights and compare full SHA input gradients.

Each setting gets its own checkpoint. The existing default checkpoint is used
only as a baseline and is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from sha256_nn import SHA256NN, bytes_to_bits
from smooth_nn import GATE_NAMES, source_gate_map
from smooth_nn.network import DEFAULT_CHECKPOINT
from smooth_nn.stability_experiment import measure_gradients, summarize
from smooth_nn.train import train_gate


HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (0.004, 0.005, 0.006, 0.007, 0.008, 0.009)
DEFAULT_MESSAGE = bytes(range(64))


def weight_tag(weight: float) -> str:
    return f"{weight:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--report", type=Path,
                        default=HERE / "smooth_gates_sweep.gradients.json")
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=72631)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()

    if args.steps < 1 or args.width < 1 or args.depth < 1 or args.batch_size < 2:
        parser.error("steps, width, depth must be positive and batch-size at least 2")
    if args.gradient_samples < 1 or args.lr <= 0 or any(w < 0 for w in args.weights):
        parser.error("gradient-samples and lr must be positive; weights nonnegative")
    if len(set(args.weights)) != len(args.weights):
        parser.error("weights must not contain duplicates")
    if not args.baseline_checkpoint.is_file():
        parser.error(f"baseline checkpoint not found: {args.baseline_checkpoint}")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    target_digest = hashlib.sha256(DEFAULT_MESSAGE).digest()
    target = bytes_to_bits(target_digest).to(dtype=torch.float64)
    generator = torch.Generator().manual_seed(args.seed + 10000)
    inputs = torch.rand(args.gradient_samples, 512, generator=generator,
                        dtype=torch.float64)

    print(f"target digest: {target_digest.hex()}", flush=True)
    baseline_rows = measure_gradients(
        args.baseline_checkpoint, inputs, target, "baseline"
    )
    baseline_summary = summarize(baseline_rows)
    print(
        f"baseline median ||grad||2={baseline_summary['median_gradient_l2_norm']:.6e} "
        f"CV={baseline_summary['gradient_l2_coefficient_of_variation']:.3%}",
        flush=True,
    )

    source_model = SHA256NN().double().eval()
    sources = source_gate_map(source_model)
    results = {}
    for weight in args.weights:
        tag = weight_tag(weight)
        output = args.output_dir / f"smooth_gates_sweep_{tag}.pt"
        print(
            f"\ntraining lambda={weight:g}, width={args.width}, depth={args.depth}, "
            f"steps={args.steps}",
            flush=True,
        )
        state_dicts = {}
        gate_metrics = {}
        for index, name in enumerate(GATE_NAMES):
            model, gate_metrics[name] = train_gate(
                name, sources[name], steps=args.steps, width=args.width,
                depth=args.depth, lr=args.lr, smoothness_weight=weight,
                batch_size=args.batch_size, seed=args.seed + index,
            )
            state_dicts[name] = model.state_dict()
            del model

        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1,
            "dtype": "float64",
            "activation": "tanh",
            "width": args.width,
            "depth": args.depth,
            "metrics": gate_metrics,
            "state_dicts": state_dicts,
        }, output)
        output.with_suffix(".metrics.json").write_text(
            json.dumps(gate_metrics, indent=2) + "\n"
        )
        del state_dicts
        print(f"saved checkpoint: {output.resolve()}", flush=True)

        gradient_rows = measure_gradients(output, inputs, target, f"lambda={weight:g}")
        gradient_summary = summarize(gradient_rows)
        norm_ratio = (
            gradient_summary["median_gradient_l2_norm"] /
            baseline_summary["median_gradient_l2_norm"]
            if baseline_summary["median_gradient_l2_norm"] else None
        )
        results[tag] = {
            "smoothness_weight": weight,
            "checkpoint": str(output),
            "gate_metrics": gate_metrics,
            "gradient_samples": gradient_rows,
            "gradient_summary": gradient_summary,
            "median_gradient_norm_ratio_to_baseline": norm_ratio,
        }
        print(
            f"lambda={weight:g}: median ||grad||2="
            f"{gradient_summary['median_gradient_l2_norm']:.6e}, "
            f"ratio={norm_ratio:.6e}, "
            f"CV={gradient_summary['gradient_l2_coefficient_of_variation']:.3%}",
            flush=True,
        )

        report = {
            "target_message": DEFAULT_MESSAGE.hex(),
            "target_digest": target_digest.hex(),
            "training": {
                "width": args.width,
                "depth": args.depth,
                "steps": args.steps,
                "learning_rate": args.lr,
                "weights": list(args.weights),
            },
            "gradient_inputs_seed": args.seed + 10000,
            "baseline": {
                "checkpoint": str(args.baseline_checkpoint),
                "samples": baseline_rows,
                "summary": baseline_summary,
            },
            "sweep": results,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    print(f"saved sweep report to {args.report.resolve()}")


if __name__ == "__main__":
    main()
