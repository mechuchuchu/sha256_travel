"""Projected mpmath optimization of the fractional SHA circuit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from sha256_nn_mpmath import DEFAULT_WEIGHTS, FractionalSHA256MP, bytes_to_bits


DEFAULT_TARGET = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def load_inputs(path, model, seed):
    ctx = model.ctx
    if path:
        data = json.loads(Path(path).read_text(), parse_float=str)
        values = data["input"] if isinstance(data, dict) else data
        return model._vector(values, 512, "inputs")
    rng = random.Random(seed)
    return [ctx.ldexp(ctx.mpf(rng.getrandbits(model.ctx.prec)), -model.ctx.prec)
            for _ in range(512)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=113,
                        help="mpmath significand bits (113 matches binary128 precision)")
    parser.add_argument("--mode", choices=("message", "compression"), default="message")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--target-hex", default=DEFAULT_TARGET)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", default="1e-16")
    parser.add_argument("--optimizer", choices=("sign-line-search", "adam"),
                        default="sign-line-search")
    parser.add_argument("--backtracks", type=int, default=6,
                        help="step-size reductions allowed per line-search step")
    parser.add_argument("--beta1", default="0.9")
    parser.add_argument("--beta2", default="0.999")
    parser.add_argument("--epsilon", default="1e-30")
    parser.add_argument("--output", type=Path, default=Path("fractional_mpmath_train.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.backtracks < 0:
        parser.error("steps must be positive and backtracks nonnegative")
    try:
        model = FractionalSHA256MP(args.mode, args.precision, args.weights)
        ctx = model.ctx
        target_bytes = bytes.fromhex(args.target_hex)
        if len(target_bytes) != 32:
            raise ValueError("target must encode exactly 32 bytes")
        target = bytes_to_bits(target_bytes)
        x = load_inputs(args.input_json, model, args.seed)
        lr, beta1, beta2, epsilon = map(ctx.mpf, (args.lr, args.beta1, args.beta2, args.epsilon))
        if lr <= 0 or not (0 <= beta1 < 1 and 0 <= beta2 < 1) or epsilon <= 0:
            raise ValueError("lr and epsilon must be positive; beta values must be in [0, 1)")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))

    first_moment = [ctx.zero] * 512
    second_moment = [ctx.zero] * 512
    history = []
    best_loss = None
    best_input = None
    best_output = None
    started = time.perf_counter()
    print(f"Training {args.mode} at {args.precision}-bit mpmath precision with {args.optimizer} "
          f"for up to {args.steps} steps",
          flush=True)
    for step in range(1, args.steps + 1):
        result = model.evaluate(x, target)
        if best_loss is None or result.loss < best_loss:
            best_loss, best_input, best_output = result.loss, list(x), list(result.output)
        grad_max = max(abs(g) for g in result.gradient)
        if not all(ctx.isfinite(g) for g in result.gradient):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        lr_used = None
        loss_after = None
        retries = 0
        accepted = True
        line_search_trials = []
        if args.optimizer == "adam":
            correction1 = 1 - beta1 ** step
            correction2 = 1 - beta2 ** step
            updated = []
            for i, (value, grad) in enumerate(zip(x, result.gradient)):
                first_moment[i] = beta1 * first_moment[i] + (1 - beta1) * grad
                second_moment[i] = beta2 * second_moment[i] + (1 - beta2) * grad * grad
                m_hat = first_moment[i] / correction1
                v_hat = second_moment[i] / correction2
                candidate = value - lr * m_hat / (ctx.sqrt(v_hat) + epsilon)
                updated.append(min(ctx.one, max(ctx.zero, candidate)))
            x = updated
            lr_used = lr
        else:
            accepted = False
            trial_lr = lr
            for retries in range(args.backtracks + 1):
                candidate = [min(ctx.one, max(ctx.zero, value - trial_lr *
                                               (1 if grad > 0 else -1 if grad < 0 else 0)))
                             for value, grad in zip(x, result.gradient)]
                if candidate == x:
                    break
                candidate_output = model.forward(candidate)
                residual = [y - ctx.mpf(t) for y, t in zip(candidate_output, target)]
                candidate_loss = ctx.fsum(value*value for value in residual) / 256
                line_search_trials.append({
                    "learning_rate": ctx.nstr(trial_lr, 12),
                    "loss": ctx.nstr(candidate_loss, 40),
                })
                if candidate_loss < result.loss:
                    x = candidate
                    lr_used = trial_lr
                    loss_after = candidate_loss
                    accepted = True
                    if best_loss is None or candidate_loss < best_loss:
                        best_loss = candidate_loss
                        best_input, best_output = list(candidate), list(candidate_output)
                    break
                trial_lr /= 10
        row = {
            "step": step,
            "loss_before": ctx.nstr(result.loss, 40),
            "gradient_max_abs": ctx.nstr(grad_max, 12),
            "nonzero_gradients": sum(g != 0 for g in result.gradient),
            "accepted": accepted,
            "learning_rate_used": ctx.nstr(lr_used, 12) if lr_used is not None else None,
            "loss_after": ctx.nstr(loss_after, 40) if loss_after is not None else None,
            "backtracks": retries,
            "line_search_trials": line_search_trials,
            "forward_seconds": result.forward_seconds,
            "backward_seconds": result.backward_seconds,
        }
        history.append(row)
        print(f"step {step}/{args.steps}: MSE={row['loss_before']}; "
              f"max|grad|={row['gradient_max_abs']}; accepted={accepted}; "
              f"lr={row['learning_rate_used']}; forward={result.forward_seconds:.2f}s "
              f"backward={result.backward_seconds:.2f}s",
              flush=True)
        if args.optimizer == "sign-line-search" and not accepted:
            print("No lower-loss step found; stopping.", flush=True)
            break

    final_output = model.forward(x)
    residual = [y - ctx.mpf(t) for y, t in zip(final_output, target)]
    final_loss = ctx.fsum(v*v for v in residual) / 256
    if final_loss < best_loss:
        best_loss, best_input, best_output = final_loss, list(x), list(final_output)
    threshold_bits = [int(value >= ctx.mpf("0.5")) for value in best_input]
    if args.mode == "message":
        threshold_message = bytes(sum(threshold_bits[i+j] << (7-j) for j in range(8))
                                  for i in range(0, 512, 8))
    else:
        threshold_message = bytes(sum(threshold_bits[i+j] << (7-j) for j in range(8))
                                  for i in range(0, 512, 8))
    import hashlib
    actual_digest = hashlib.sha256(threshold_message).hexdigest() if args.mode == "message" else None
    digits = (args.precision * 30103 + 99999) // 100000 + 3
    payload = {
        "model": "FractionalSHA256MP", "optimizer": args.optimizer,
        "precision_bits": args.precision, "precision_kind": "mpmath significand bits",
        "mode": args.mode, "learning_rate": args.lr, "steps": len(history),
        "beta1": args.beta1, "beta2": args.beta2, "epsilon": args.epsilon,
        "seed": args.seed, "input_source": str(args.input_json) if args.input_json else "random dyadic",
        "target_hex": target_bytes.hex(), "initial_loss": history[0]["loss_before"],
        "final_loss": ctx.nstr(final_loss, 40), "best_loss": ctx.nstr(best_loss, 40),
        "history": history,
        "threshold_rule": "input >= 0.5 becomes 1",
        "thresholded_message_hex": threshold_message.hex(),
        "thresholded_message_sha256": actual_digest,
        "input": [ctx.nstr(value, digits) for value in best_input],
        "output": [ctx.nstr(value, digits) for value in best_output],
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"final MSE={payload['final_loss']}")
    print(f"best MSE={payload['best_loss']}")
    if actual_digest is not None:
        print(f"thresholded input SHA-256={actual_digest}")
    print(f"elapsed={payload['elapsed_seconds']:.2f}s; saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
