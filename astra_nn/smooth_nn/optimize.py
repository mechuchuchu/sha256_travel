"""Optimize a relaxed 64-byte message toward a chosen SHA-256 digest.

The input remains continuous during gradient descent. The reported candidate
message is made by rounding those 512 values at 0.5 and hashing the resulting
64 bytes with hashlib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from sha256_nn import bits_to_bytes, bytes_to_bits
from smooth_nn.network import DEFAULT_CHECKPOINT, SmoothSHA256NN


DEFAULT_MESSAGE = bytes(range(64))


def parse_hex_bytes(value: str, *, expected_length: int, label: str) -> bytes:
    try:
        result = bytes.fromhex(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be valid hexadecimal") from error
    if len(result) != expected_length:
        raise argparse.ArgumentTypeError(
            f"{label} must encode exactly {expected_length} bytes"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-hex",
        help="desired 32-byte SHA-256 digest in hexadecimal",
    )
    target_group.add_argument(
        "--target-message-hex",
        help="64-byte message whose SHA-256 digest becomes the target",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--binary-weight", type=float, default=0.01,
        help="weight for mean(x * (1 - x)), which encourages binary inputs",
    )
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=72631)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--initial-message-hex",
        help="optional 64-byte starting message; otherwise start randomly",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("steps must be positive")
    if args.lr <= 0 or args.binary_weight < 0 or args.grad_clip < 0:
        parser.error("lr must be positive and penalty/gradient clip nonnegative")

    if args.target_hex:
        try:
            target_digest = parse_hex_bytes(
                args.target_hex, expected_length=32, label="target digest"
            )
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        target_source_message = None
    else:
        source_hex = args.target_message_hex
        if source_hex is None:
            target_source_message = DEFAULT_MESSAGE
        else:
            try:
                target_source_message = parse_hex_bytes(
                    source_hex, expected_length=64, label="target message"
                )
            except argparse.ArgumentTypeError as error:
                parser.error(str(error))
        target_digest = hashlib.sha256(target_source_message).digest()

    initial_message = None
    if args.initial_message_hex:
        try:
            initial_message = parse_hex_bytes(
                args.initial_message_hex, expected_length=64, label="initial message"
            )
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dtype = torch.float64
    model = SmoothSHA256NN(checkpoint=args.checkpoint, device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    target = bytes_to_bits(target_digest).to(device=device, dtype=dtype)
    if initial_message is None:
        initial = torch.rand(512, device=device, dtype=dtype)
    else:
        initial = bytes_to_bits(initial_message).to(device=device, dtype=dtype)
    x = torch.nn.Parameter(initial.unsqueeze(0))
    optimizer = torch.optim.Adam([x], lr=args.lr)

    report_every = max(1, args.steps // 10)
    history = []
    best_mse = float("inf")
    best_input = x.detach().clone()
    print(f"target digest: {target_digest.hex()}", flush=True)
    if target_source_message is not None:
        print(f"target source message: {target_source_message.hex()}", flush=True)
    print(f"initial input: {'random' if initial_message is None else initial_message.hex()}",
          flush=True)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)[0]
        output_loss = F.mse_loss(prediction, target)
        binary_loss = (x * (1 - x)).mean()
        loss = output_loss + args.binary_weight * binary_loss
        loss.backward()
        grad_norm = x.grad.norm().item()
        if output_loss.item() < best_mse:
            best_mse = output_loss.item()
            best_input.copy_(x.detach())
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_([x], args.grad_clip)
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0, 1)

        if step == 1 or step % report_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": loss.item(),
                "output_mse": output_loss.item(),
                "binary_penalty": binary_loss.item(),
                "gradient_norm": grad_norm,
            }
            history.append(row)
            print(
                f"step {step:4d}/{args.steps}: mse={output_loss.item():.6e} "
                f"binary={binary_loss.item():.6e} |grad|={grad_norm:.6e}",
                flush=True,
            )

    with torch.no_grad():
        soft_prediction = model(x)[0]
        soft_mse = F.mse_loss(soft_prediction, target).item()
        if soft_mse < best_mse:
            best_input.copy_(x)
            best_mse = soft_mse
        x.copy_(best_input)
        soft_prediction = model(x)[0]
        soft_mse = F.mse_loss(soft_prediction, target).item()
        soft_match = ((soft_prediction >= 0.5) == target.bool()).double().mean().item()
        candidate_bits = (x[0] >= 0.5).to(dtype=dtype)
        candidate_message = bits_to_bytes(candidate_bits)
        candidate_prediction = model(candidate_bits.unsqueeze(0))[0]
        candidate_digest = bits_to_bytes(candidate_prediction)
        candidate_matches_target = candidate_digest == target_digest
        hashlib_digest = hashlib.sha256(candidate_message).digest()
        candidate_hash_matches_target = hashlib_digest == target_digest
        binary_fraction = ((x[0] < 0.1) | (x[0] > 0.9)).double().mean().item()

    print(f"final relaxed-output MSE: {soft_mse:.8e}")
    print(f"relaxed output threshold accuracy: {soft_match:.2%}")
    print(f"input values within 0.1 of a bit: {binary_fraction:.2%}")
    print(f"rounded candidate message: {candidate_message.hex()}")
    print(f"rounded candidate SHA-256: {hashlib_digest.hex()}")
    print(f"candidate matches target: {candidate_hash_matches_target}")

    if args.output:
        payload = {
            "target_digest": target_digest.hex(),
            "target_source_message": (
                target_source_message.hex() if target_source_message is not None else None
            ),
            "initial_message": initial_message.hex() if initial_message is not None else None,
            "seed": args.seed,
            "steps": args.steps,
            "learning_rate": args.lr,
            "binary_weight": args.binary_weight,
            "final_output_mse": soft_mse,
            "relaxed_output_threshold_accuracy": soft_match,
            "input_near_binary_fraction": binary_fraction,
            "candidate_message": candidate_message.hex(),
            "candidate_model_digest": candidate_digest.hex(),
            "candidate_sha256": hashlib_digest.hex(),
            "candidate_matches_target": candidate_hash_matches_target,
            "candidate_model_matches_target": candidate_matches_target,
            "continuous_input": x[0].tolist(),
            "history": history,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"saved result to {args.output.resolve()}")


if __name__ == "__main__":
    main()
