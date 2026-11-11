"""Train a larger, smoother gate set and compare full-network gradients.

This experiment writes a separate checkpoint so it does not replace the
default smooth gates used by ``SmoothSHA256NN``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from sha256_nn import SHA256NN, bytes_to_bits
from smooth_nn import GATE_NAMES, source_gate_map
from smooth_nn.network import DEFAULT_CHECKPOINT, SmoothSHA256NN
from smooth_nn.train import train_gate


HERE = Path(__file__).resolve().parent
DEFAULT_MESSAGE = bytes(range(64))


def measure_gradients(checkpoint: Path, inputs: torch.Tensor,
                      target: torch.Tensor, label: str) -> list[dict]:
    model = SmoothSHA256NN(checkpoint=checkpoint, dtype=torch.float64)
    rows = []
    for index, values in enumerate(inputs):
        x = values.detach().clone().unsqueeze(0).requires_grad_(True)
        prediction = model(x)[0]
        loss = F.mse_loss(prediction, target)
        gradient, = torch.autograd.grad(loss, x)
        finite = torch.isfinite(gradient)
        finite_gradient = gradient[finite]
        row = {
            "sample": index,
            "output_mse": loss.item(),
            "gradient_l2_norm": torch.linalg.vector_norm(finite_gradient).item(),
            "gradient_rms": finite_gradient.square().mean().sqrt().item(),
            "gradient_max_abs": finite_gradient.abs().max().item(),
            "gradient_finite_fraction": finite.double().mean().item(),
        }
        rows.append(row)
        print(
            f"{label} sample {index}: mse={row['output_mse']:.6e} "
            f"||grad||2={row['gradient_l2_norm']:.6e} "
            f"max|grad|={row['gradient_max_abs']:.6e} "
            f"finite={row['gradient_finite_fraction']:.1%}",
            flush=True,
        )
        del x, prediction, loss, gradient
    del model
    return rows


def summarize(rows: list[dict]) -> dict:
    norms = [row["gradient_l2_norm"] for row in rows]
    losses = [row["output_mse"] for row in rows]
    mean_norm = statistics.mean(norms)
    min_norm = min(norms)
    return {
        "median_gradient_l2_norm": statistics.median(norms),
        "gradient_l2_coefficient_of_variation": (
            statistics.pstdev(norms) / mean_norm if mean_norm else 0.0
        ),
        "gradient_l2_max_min_ratio": max(norms) / min_norm if min_norm else float("inf"),
        "max_gradient_l2_norm": max(norms),
        "median_output_mse": statistics.median(losses),
        "all_gradients_finite": all(row["gradient_finite_fraction"] == 1.0
                                     for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path,
                        default=HERE / "smooth_gates_stability.pt")
    parser.add_argument("--report", type=Path,
                        default=HERE / "smooth_gates_stability.gradients.json")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=72631)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--target-message-hex",
                        help="64-byte message used to derive the target digest")
    args = parser.parse_args()

    if args.steps < 1 or args.width < 1 or args.depth < 1 or args.batch_size < 2:
        parser.error("steps, width, depth must be positive and batch-size at least 2")
    if args.gradient_samples < 1 or args.lr <= 0 or args.smoothness_weight < 0:
        parser.error("gradient-samples and lr must be positive; smoothness weight nonnegative")
    if not args.baseline_checkpoint.is_file():
        parser.error(f"baseline checkpoint not found: {args.baseline_checkpoint}")

    if args.target_message_hex is None:
        target_message = DEFAULT_MESSAGE
    else:
        try:
            target_message = bytes.fromhex(args.target_message_hex)
        except ValueError:
            parser.error("target-message-hex must be valid hexadecimal")
        if len(target_message) != 64:
            parser.error("target-message-hex must encode exactly 64 bytes")
    target_digest = hashlib.sha256(target_message).digest()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source_model = SHA256NN().double().eval()
    sources = source_gate_map(source_model)
    state_dicts = {}
    gate_metrics = {}
    print(
        f"training width={args.width}, depth={args.depth}, "
        f"smoothness_weight={args.smoothness_weight}, steps={args.steps}",
        flush=True,
    )
    for index, name in enumerate(GATE_NAMES):
        model, gate_metrics[name] = train_gate(
            name, sources[name], steps=args.steps, width=args.width,
            depth=args.depth, lr=args.lr,
            smoothness_weight=args.smoothness_weight,
            batch_size=args.batch_size, seed=args.seed + index,
        )
        state_dicts[name] = model.state_dict()
        print(f"{name} metrics: {json.dumps(gate_metrics[name], sort_keys=True)}",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "dtype": "float64",
        "activation": "tanh",
        "width": args.width,
        "depth": args.depth,
        "metrics": gate_metrics,
        "state_dicts": state_dicts,
    }
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(gate_metrics, indent=2) + "\n"
    )
    print(f"saved separate checkpoint to {args.output.resolve()}", flush=True)

    generator = torch.Generator().manual_seed(args.seed + 10000)
    inputs = torch.rand(args.gradient_samples, 512, generator=generator,
                        dtype=torch.float64)
    target = bytes_to_bits(target_digest).to(dtype=torch.float64)
    print(f"target digest: {target_digest.hex()}", flush=True)
    baseline_rows = measure_gradients(
        args.baseline_checkpoint, inputs, target, "baseline"
    )
    experiment_rows = measure_gradients(args.output, inputs, target, "larger/smoother")

    baseline_summary = summarize(baseline_rows)
    experiment_summary = summarize(experiment_rows)
    baseline_norm = baseline_summary["median_gradient_l2_norm"]
    experiment_norm = experiment_summary["median_gradient_l2_norm"]
    ratio = experiment_norm / baseline_norm if baseline_norm else float("inf")
    print(f"median gradient norm ratio (new / baseline): {ratio:.6e}", flush=True)
    print(
        "gradient norm CV (baseline / new): "
        f"{baseline_summary['gradient_l2_coefficient_of_variation']:.3%} / "
        f"{experiment_summary['gradient_l2_coefficient_of_variation']:.3%}",
        flush=True,
    )

    report = {
        "target_message": target_message.hex(),
        "target_digest": target_digest.hex(),
        "training": {
            "width": args.width,
            "depth": args.depth,
            "steps": args.steps,
            "smoothness_weight": args.smoothness_weight,
            "learning_rate": args.lr,
            "gate_metrics": gate_metrics,
        },
        "gradient_inputs_seed": args.seed + 10000,
        "baseline": {"checkpoint": str(args.baseline_checkpoint),
                     "samples": baseline_rows, "summary": baseline_summary},
        "larger_smoother": {"checkpoint": str(args.output),
                             "samples": experiment_rows,
                             "summary": experiment_summary},
        "median_gradient_norm_ratio_new_over_baseline": ratio,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved gradient comparison to {args.report.resolve()}")


if __name__ == "__main__":
    main()
