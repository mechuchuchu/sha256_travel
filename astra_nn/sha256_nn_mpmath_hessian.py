"""Full local MSE Hessian for FractionalSHA256MP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from sha256_nn_mpmath import DEFAULT_WEIGHTS, FractionalSHA256MP, bytes_to_bits


DEFAULT_TARGET = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=113,
                        help="mpmath significand bits (113 matches binary128 precision)")
    parser.add_argument("--mode", choices=("message", "compression"), default="message")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--input-json", type=Path,
                        default=Path("fractional_mpmath_train_113_linesearch.json"))
    parser.add_argument("--target-hex", default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=Path("fractional_mpmath_hessian_113.json"))
    args = parser.parse_args()
    try:
        model = FractionalSHA256MP(args.mode, args.precision, args.weights)
        ctx = model.ctx
        raw = json.loads(args.input_json.read_text(), parse_float=str)
        inputs = model._vector(raw["input"] if isinstance(raw, dict) else raw, 512, "inputs")
        target_bytes = bytes.fromhex(args.target_hex)
        if len(target_bytes) != 32:
            raise ValueError("target must encode exactly 32 bytes")
        target = bytes_to_bits(target_bytes)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))

    started = time.perf_counter()
    print(f"Building the {args.mode} computation graph at {args.precision}-bit precision...",
          flush=True)
    tape, input_nodes, output_nodes = model._build(inputs, recording=True)
    output = [tape.values[node] for node in output_nodes]
    residual = [value - ctx.mpf(bit) for value, bit in zip(output, target)]
    loss = ctx.fsum(value*value for value in residual) / 256

    jacobian = []
    jacobian_start = time.perf_counter()
    for row, output_node in enumerate(output_nodes, start=1):
        adjoints = tape.backward([(output_node, ctx.one)])
        jacobian.append([adjoints[node] for node in input_nodes])
        if row % 16 == 0 or row == len(output_nodes):
            print(f"Jacobian rows {row}/{len(output_nodes)}; "
                  f"elapsed {time.perf_counter()-jacobian_start:.1f}s", flush=True)
        del adjoints

    factor = ctx.mpf(2) / 256
    gradient = [factor * ctx.fsum(residual[k] * jacobian[k][i] for k in range(256))
                for i in range(512)]
    hessian = [[ctx.zero for _ in range(512)] for _ in range(512)]
    gram_start = time.perf_counter()
    for i in range(512):
        for j in range(i, 512):
            value = factor * ctx.fsum(jacobian[k][i] * jacobian[k][j] for k in range(256))
            hessian[i][j] = value
            hessian[j][i] = value
        if (i + 1) % 32 == 0 or i == 511:
            print(f"Hessian rows {i+1}/512; elapsed {time.perf_counter()-gram_start:.1f}s",
                  flush=True)

    digits = (args.precision * 30103 + 99999) // 100000 + 3
    encode = lambda value: ctx.nstr(value, digits)
    diagonal = [hessian[i][i] for i in range(512)]
    trace = ctx.fsum(diagonal)
    payload = {
        "model": "FractionalSHA256MP",
        "quantity": "local Hessian of mean squared error",
        "hessian_formula": "(2/256) * J.T * J",
        "relu_derivative_at_zero": 0,
        "precision_bits": args.precision,
        "precision_kind": "mpmath significand bits",
        "mode": args.mode,
        "input_source": str(args.input_json),
        "target_hex": target_bytes.hex(),
        "loss": encode(loss),
        "gradient_max_abs": encode(max(abs(value) for value in gradient)),
        "gradient": list(map(encode, gradient)),
        "hessian_shape": [512, 512],
        "hessian_rank_upper_bound": 256,
        "hessian_trace": encode(trace),
        "diagonal_min": encode(min(diagonal)),
        "diagonal_max": encode(max(diagonal)),
        "max_abs_entry": encode(max(abs(value) for row in hessian for value in row)),
        "jacobian_rows": len(jacobian),
        "computation_graph_nodes": len(tape.values),
        "jacobian_seconds": time.perf_counter() - jacobian_start,
        "gram_seconds": time.perf_counter() - gram_start,
        "total_seconds": time.perf_counter() - started,
        "hessian": [[encode(value) for value in row] for row in hessian],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"MSE={payload['loss']}")
    print(f"max |gradient|={payload['gradient_max_abs']}")
    print(f"trace(H)={payload['hessian_trace']}; max |H_ij|={payload['max_abs_entry']}")
    print(f"Saved {args.output.resolve()} ({payload['total_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
