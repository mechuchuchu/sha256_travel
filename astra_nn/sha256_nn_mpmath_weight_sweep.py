"""Measure input-loss gradients as one shared SHA gate weight is perturbed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from sha256_nn_mpmath import DEFAULT_WEIGHTS, FractionalSHA256MP, bytes_to_bits, bits_to_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--precision", type=int, default=113,
                           help="mpmath significand bits (113 for binary128, 237 for binary256)")
    precision.add_argument("--float256", dest="precision", action="store_const", const=237,
                           help="use binary256's 237-bit significand with mpmath")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--message-hex", default="616263",
                        help="0..55-byte message; the default is abc")
    parser.add_argument("--start-exponent", type=int, default=1)
    parser.add_argument("--end-exponent", type=int, default=34)
    parser.add_argument("--output", type=Path, help="defaults to weight_perturbation_<precision>.json")
    args = parser.parse_args()
    if args.start_exponent < 1 or args.end_exponent < args.start_exponent:
        parser.error("exponents must satisfy 1 <= start <= end")
    if args.output is None:
        args.output = Path(f"weight_perturbation_{args.precision}.json")
    try:
        message = bytes.fromhex(args.message_hex)
        if len(message) > 55:
            raise ValueError("compression mode supports messages of at most 55 bytes")
        model = FractionalSHA256MP("compression", args.precision, args.weights)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))

    ctx = model.ctx
    digits = (args.precision * 30103 + 99999) // 100000 + 3
    def encode(value):
        return ctx.nstr(value, digits)
    padded = (message + b"\x80" + bytes(55-len(message)) +
              (8*len(message)).to_bytes(8, "big"))
    inputs = bytes_to_bits(padded)
    target_bytes = hashlib.sha256(message).digest()
    target = bytes_to_bits(target_bytes)
    baseline = model.evaluate(inputs, target)
    if baseline.loss != 0 or any(gradient != 0 for gradient in baseline.gradient):
        raise RuntimeError("reference input must have exactly zero loss and input gradient")
    if bits_to_bytes(baseline.output) != target_bytes:
        raise RuntimeError("binary circuit output does not match hashlib")
    print(f"baseline: loss={baseline.loss}; max|gradient|=0; "
          f"input SHA-256={target_bytes.hex()}", flush=True)

    gate = model.gates["xor_and"]
    row, column = 0, 0
    original_weight = ctx.mpf(gate.w_in[row][column])
    rows = []
    started = time.perf_counter()
    for exponent in range(args.start_exponent, args.end_exponent + 1):
        epsilon = ctx.mpf(f"1e-{exponent}")
        perturbed_weight = original_weight + epsilon
        actual_delta = perturbed_weight - original_weight
        gate.w_in[row][column] = perturbed_weight
        result = model.evaluate(inputs, target)
        gradient_max = max(abs(value) for value in result.gradient)
        gradient_l2 = ctx.sqrt(ctx.fsum(value*value for value in result.gradient))
        output_max = max(abs(value) for value in result.output)
        row_data = {
            "epsilon_requested": f"1e-{exponent}",
            "epsilon_represented": encode(actual_delta),
            "perturbed_weight": encode(perturbed_weight),
            "mse": encode(result.loss),
            "gradient_max_abs": encode(gradient_max),
            "gradient_l2": encode(gradient_l2),
            "output_max_abs": encode(output_max),
            "nonzero_gradient_count": sum(value != 0 for value in result.gradient),
            "forward_seconds": result.forward_seconds,
            "backward_seconds": result.backward_seconds,
        }
        rows.append(row_data)
        print(f"eps=1e-{exponent:02d}: MSE={ctx.nstr(result.loss, 9)}; "
              f"max|grad|={ctx.nstr(gradient_max, 9)}; "
              f"grad L2={ctx.nstr(gradient_l2, 9)}", flush=True)
    gate.w_in[row][column] = int(original_weight)

    payload = {
        "model": "FractionalSHA256MP",
        "precision_bits": args.precision,
        "precision_kind": "mpmath significand bits",
        "decimal_digits_saved": digits,
        "mode": "compression",
        "message_hex": message.hex(),
        "padded_block_hex": padded.hex(),
        "target_hex": target_bytes.hex(),
        "input": inputs,
        "perturbed_parameter": "xor_and.weight_in[0][0]",
        "original_weight": encode(original_weight),
        "perturbation": "positive additive epsilon",
        "baseline_mse": encode(baseline.loss),
        "baseline_gradient_max_abs": "0",
        "results": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {args.output.resolve()} in {payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
