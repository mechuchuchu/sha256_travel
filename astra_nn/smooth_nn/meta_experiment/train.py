"""Fine-tune a copy of the 0.009 gates through the full SHA graph.

The gradient regularizer uses a central directional difference: it is a
finite-difference approximation to the mixed parameter/input derivative,
not an exact second-order autograd calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from sha256_nn import bits_to_bytes, bytes_to_bits
from smooth_nn import GATE_NAMES
from smooth_nn.network import SmoothSHA256NN
from smooth_nn.train import curvature_penalty

HERE = Path(__file__).resolve().parent


class TrainableSHA(SmoothSHA256NN):
    def __init__(self, checkpoint_path):
        super().__init__(checkpoint=checkpoint_path, dtype=torch.float64)
        for name in GATE_NAMES:
            for parameter in getattr(self, name).gate.parameters():
                parameter.requires_grad_(True)

    def prepare_anchors(self):
        # The correction must participate in autograd. Compute it once per
        # full forward, not once per invocation of each shared SHA gate.
        for name in GATE_NAMES:
            gate = getattr(self, name).gate
            gate.anchor_correction = gate.truth_table - gate.net(gate.vertices)
            gate._anchors_cached = True

    def add_words(self, x, y):
        if torch.is_grad_enabled():
            return checkpoint(super().add_words, x, y, use_reentrant=False)
        return super().add_words(x, y)

    def forward(self, bits):
        self.prepare_anchors()
        return super().forward(bits)


def pair(rng):
    start, end = rng.randbytes(64), rng.randbytes(64)
    x0, x1 = bytes_to_bits(start).double(), bytes_to_bits(end).double()
    midpoint = (x0 + x1) / 2
    return start, end, midpoint, bytes_to_bits(hashlib.sha256(start).digest()).double()


def evaluate(model, pairs):
    losses, binary_matches = [], []
    with torch.no_grad():
        for start, end, midpoint, target in pairs:
            losses.append(F.mse_loss(model(midpoint), target).item())
            for message in (start, end):
                digest = bits_to_bytes(model(bytes_to_bits(message).double()))
                binary_matches.append(digest == hashlib.sha256(message).digest())
    return {"midpoint_mse": losses, "mean_midpoint_mse": sum(losses) / len(losses),
            "all_binary_hashes_match": all(binary_matches), "binary_cases": len(binary_matches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "original_0p009.pt")
    parser.add_argument("--output", type=Path, default=HERE / "trained_0p009.pt")
    parser.add_argument("--report", type=Path, default=HERE / "training_report.json")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=53291)
    parser.add_argument("--validation-pairs", type=int, default=3)
    parser.add_argument("--target-mse", type=float, default=0.23)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--smoothness-weight", type=float, default=1e-4)
    parser.add_argument("--binary-weight", type=float, default=1e-3)
    parser.add_argument("--meta-weight", type=float, default=1e-4)
    parser.add_argument("--target-gradient-norm", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    args = parser.parse_args()
    if args.steps < 1 or args.validation_pairs < 1 or args.lr <= 0 or not 0 < args.epsilon < 0.5:
        parser.error("steps, validation-pairs and lr must be positive; epsilon must be in (0, 0.5)")
    if min(args.smoothness_weight, args.binary_weight, args.meta_weight, args.target_gradient_norm) < 0:
        parser.error("loss weights and gradient target must be nonnegative")
    if args.output.resolve() == args.source.resolve() or args.output.exists() or args.report.exists():
        parser.error("output and report must be new files; the source cannot be overwritten")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source_sha256 = hashlib.sha256(args.source.read_bytes()).hexdigest()
    model = TrainableSHA(args.source)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    validation = [pair(random.Random(args.seed + 10000 + i)) for i in range(args.validation_pairs)]
    print("validating original checkpoint", flush=True)
    before = evaluate(model, validation)
    print(f"before: {before}", flush=True)
    if not before["all_binary_hashes_match"]:
        raise RuntimeError("baseline does not match standard SHA-256")
    rng = random.Random(args.seed)
    history = []
    for step in range(1, args.steps + 1):
        start, end, midpoint, target = pair(rng)
        # Perturb only coordinates at 0.5, keeping both finite-difference inputs
        # inside [0, 1]. A Rademacher direction estimates the gradient norm in
        # this varying-bit subspace, rather than the full 512-dimensional norm.
        active = midpoint == 0.5
        signs = torch.randint(0, 2, (512,), dtype=torch.int64).double() * 2 - 1
        direction = active.double() * signs
        active_count = int(active.sum().item())
        if not active_count:
            raise RuntimeError("random endpoints happened to be identical")
        plus = midpoint + args.epsilon * direction
        minus = midpoint - args.epsilon * direction
        with torch.no_grad():
            fp = F.mse_loss(model(plus), target).item()
            fm = F.mse_loss(model(minus), target).item()
        directional = (fp - fm) / (2 * args.epsilon)
        # E[(d dot gradient)^2] = ||gradient||^2 for Rademacher d.
        # A smooth absolute value makes the target norm a differentiable loss.
        magnitude = math.sqrt(directional * directional + 1e-20)
        meta_loss = (magnitude - args.target_gradient_norm) ** 2
        coefficient = args.meta_weight * 2 * (magnitude - args.target_gradient_norm) * directional / magnitude / (2 * args.epsilon)
        optimizer.zero_grad(set_to_none=True)
        mid_loss = F.mse_loss(model(midpoint), target)
        objective = (mid_loss - args.target_mse).square()
        print(f"step {step}: midpoint forward MSE={mid_loss.item():.9g}; backpropagating", flush=True)
        objective.backward()
        if args.meta_weight:
            # Separate forwards/backwards avoid retaining three full SHA graphs.
            (F.mse_loss(model(plus), target) * coefficient).backward()
            (F.mse_loss(model(minus), target) * -coefficient).backward()
        smoothness = torch.zeros((), dtype=torch.float64)
        binary_loss = torch.zeros((), dtype=torch.float64)
        for name in GATE_NAMES:
            gate = getattr(model, name).gate
            # The anchored output is exactly binary by construction; supervise
            # the uncorrected MLP at vertices to make this a nontrivial loss.
            binary_loss = binary_loss + F.mse_loss(gate.net(gate.vertices), gate.truth_table)
            x = torch.rand(8, gate.input_dim, dtype=torch.float64)
            gate._anchors_cached = False
            smoothness = smoothness + curvature_penalty(gate, x)
            gate._anchors_cached = True
        regularizer = args.binary_weight * binary_loss + args.smoothness_weight * smoothness
        regularizer.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True).item()
        optimizer.step()
        row = {"step": step, "start_message": start.hex(), "end_message": end.hex(),
               "midpoint_mse": mid_loss.item(), "mse_objective": objective.item(),
               "estimated_directional_gradient": directional, "meta_loss": meta_loss,
               "active_bits": active_count, "gate_binary_loss": binary_loss.item(),
               "gate_curvature_loss": smoothness.item(), "parameter_gradient_norm": gradient_norm}
        history.append(row)
        print(f"step {step}: gradient={directional:.4g} curvature={smoothness.item():.4g} gate_binary={binary_loss.item():.4g}", flush=True)
    print("validating trained checkpoint", flush=True)
    after = evaluate(model, validation)
    print(f"after: {after}", flush=True)
    if not after["all_binary_hashes_match"]:
        raise RuntimeError("trained gates do not match standard SHA-256 on binary validation cases")
    if not all(torch.isfinite(p.detach()).all().item() for p in params):
        raise RuntimeError("non-finite trained parameter")
    original = torch.load(args.source, map_location="cpu", weights_only=True)
    original["source_gate_metrics"] = original.pop("metrics", {})
    for name in GATE_NAMES:
        gate = getattr(model, name).gate
        gate.cache_anchor_correction()
        original["state_dicts"][name] = gate.state_dict()
    original["fine_tuning"] = {"source_sha256": source_sha256, "steps": args.steps, "seed": args.seed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(original, args.output)
    args.report.write_text(json.dumps({"source": str(args.source), "source_sha256": source_sha256,
        "output": str(args.output), "target_mse": args.target_mse, "seed": args.seed,
        "learning_rate": args.lr, "smoothness_weight": args.smoothness_weight,
        "binary_weight": args.binary_weight, "meta_weight": args.meta_weight,
        "target_gradient_norm": args.target_gradient_norm, "epsilon": args.epsilon,
        "gradient_definition": "central finite difference in random signs on differing input bits",
        "validation_seed_start": args.seed + 10000, "before": before, "after": after,
        "history": history}, indent=2) + "\n")
    print(f"saved {args.output} and {args.report}", flush=True)


if __name__ == "__main__":
    main()
