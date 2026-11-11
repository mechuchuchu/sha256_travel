"""Optimize a relaxed 64-byte message with L-BFGS and a bounded input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from sha256_nn import bits_to_bytes, bytes_to_bits
from smooth_nn.network import SmoothSHA256NN
from smooth_nn.optimize import DEFAULT_MESSAGE, parse_hex_bytes


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=HERE / "smooth_gates_sweep_0p009.pt")
    parser.add_argument("--target-hex", help="desired 32-byte digest")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--loss-scale", type=float, default=10000.0)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=72631)
    parser.add_argument("--output", type=Path,
                        default=HERE / "optimization_0p009_lbfgs.json")
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0 or args.loss_scale <= 0 or args.history_size < 1:
        parser.error("steps, lr, loss-scale and history-size must be positive")
    if args.target_hex is None:
        target_digest = hashlib.sha256(DEFAULT_MESSAGE).digest()
    else:
        try:
            target_digest = parse_hex_bytes(args.target_hex, expected_length=32,
                                            label="target digest")
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = SmoothSHA256NN(checkpoint=args.checkpoint, dtype=torch.float64)
    target = bytes_to_bits(target_digest).double()
    initial = torch.rand(512, dtype=torch.float64)
    latent = torch.nn.Parameter(torch.logit(initial.clamp(1e-8, 1 - 1e-8)))
    optimizer = torch.optim.LBFGS(
        [latent], lr=args.lr, max_iter=1, history_size=args.history_size,
        line_search_fn="strong_wolfe", tolerance_grad=0.0, tolerance_change=0.0,
    )

    with torch.no_grad():
        initial_prediction = model(torch.sigmoid(latent).unsqueeze(0))[0]
        initial_mse = F.mse_loss(initial_prediction, target).item()
        initial_accuracy = ((initial_prediction >= 0.5) == target.bool()).double().mean().item()
    best_mse = initial_mse
    best_input = initial.clone()
    best_step = 0
    evaluations = 0
    history = []
    print(f"target digest: {target_digest.hex()}", flush=True)
    print(f"initial relaxed MSE: {initial_mse:.9e}", flush=True)

    for step in range(1, args.steps + 1):
        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.sigmoid(latent).unsqueeze(0))[0]
            mse = F.mse_loss(prediction, target)
            (mse * args.loss_scale).backward()
            evaluations += 1
            print(f"  evaluation {evaluations}: mse={mse.item():.9e}", flush=True)
            return mse * args.loss_scale

        optimizer.step(closure)
        with torch.no_grad():
            current = torch.sigmoid(latent)
            prediction = model(current.unsqueeze(0))[0]
            mse = F.mse_loss(prediction, target).item()
            accuracy = ((prediction >= 0.5) == target.bool()).double().mean().item()
            if mse < best_mse:
                best_mse = mse
                best_input = current.clone()
                best_step = step
        history.append({"step": step, "output_mse": mse,
                        "relaxed_output_threshold_accuracy": accuracy,
                        "closure_evaluations": evaluations})
        print(f"step {step}/{args.steps}: accepted mse={mse:.9e}", flush=True)

    with torch.no_grad():
        best_prediction = model(best_input.unsqueeze(0))[0]
        best_accuracy = ((best_prediction >= 0.5) == target.bool()).double().mean().item()
        candidate_message = bits_to_bytes((best_input >= 0.5).double())
        candidate_model_digest = bits_to_bytes(model((best_input >= 0.5).double().unsqueeze(0))[0])
    initial_message = bits_to_bytes((initial >= 0.5).double())
    initial_digest = hashlib.sha256(initial_message).digest()
    candidate_digest = hashlib.sha256(candidate_message).digest()

    def matching_bits(digest: bytes) -> int:
        return 256 - sum((a ^ b).bit_count() for a, b in zip(digest, target_digest))

    report = {
        "optimizer": "LBFGS",
        "line_search": "strong_wolfe",
        "input_parameterization": "sigmoid",
        "checkpoint": str(args.checkpoint),
        "target_digest": target_digest.hex(),
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.lr,
        "loss_scale": args.loss_scale,
        "closure_evaluations": evaluations,
        "initial_output_mse": initial_mse,
        "best_output_mse": best_mse,
        "best_step": best_step,
        "initial_relaxed_output_threshold_accuracy": initial_accuracy,
        "best_relaxed_output_threshold_accuracy": best_accuracy,
        "initial_rounded_sha256": initial_digest.hex(),
        "initial_rounded_digest_matching_bits": matching_bits(initial_digest),
        "candidate_message": candidate_message.hex(),
        "candidate_model_digest": candidate_model_digest.hex(),
        "candidate_sha256": candidate_digest.hex(),
        "candidate_matches_target": candidate_digest == target_digest,
        "candidate_digest_matching_bits": matching_bits(candidate_digest),
        "continuous_input": best_input.tolist(),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"best relaxed MSE: {best_mse:.9e} at step {best_step}")
    print(f"rounded SHA-256: {candidate_digest.hex()}")
    print(f"rounded SHA target matching bits: {matching_bits(initial_digest)} -> "
          f"{matching_bits(candidate_digest)}/256")
    print(f"candidate matches target: {candidate_digest == target_digest}")
    print(f"saved result to {args.output.resolve()}")


if __name__ == "__main__":
    main()
