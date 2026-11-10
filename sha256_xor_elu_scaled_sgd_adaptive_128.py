#!/usr/bin/env python3
"""128-bit mpmath version of the adaptive sign-SGD SHA-256 experiment.

Uses sha256_mpmath_backward.py for the learned-XOR model and input gradients.
NumPy is used only to reproduce the original script's seeded random bytes.
All model, gradient, learning-rate, update, and loss arithmetic uses mpmath.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from sha256_mpmath_backward import (
    DEFAULT_WEIGHTS, SHA256MP, bytes_to_bits, pad_message,
)


def candidate_lrs(ctx, min_exponent=-20):
    """Ascending 1, 3 decade grid, constructed without binary64 intermediates."""
    if min_exponent > -2:
        raise ValueError("minimum learning-rate exponent must be at most -2")
    return [ctx.mpf(f"{coefficient}e{exponent}")
            for exponent in range(min_exponent, -1) for coefficient in (1, 3)
            ] + [ctx.mpf("1e-1")]


def initial_problem(ctx, seed):
    rng = np.random.default_rng(seed)
    message = rng.bytes(32)
    target_bytes = rng.bytes(32)
    x = [ctx.mpf(v) for v in bytes_to_bits(pad_message(message))]
    target = bytes_to_bits(target_bytes)
    return message, target_bytes, x, target


def state_key(x):
    """Exact mpmath state, including changes too small to survive float64."""
    return tuple(value._mpf_ for value in x)


def projected_step(ctx, x, direction, lr):
    return [min(ctx.one, max(ctx.zero, value - lr * sign))
            for value, sign in zip(x, direction)]


@dataclass
class StepResult:
    chosen: list
    base_loss: object
    chosen_lr: object
    post_loss: object
    step_delta: object
    total_delta: object
    boundary_bits: int
    grad_max: object

    def row(self, step, ctx):
        fmt = lambda value: ctx.nstr(value, ctx.dps + 5)
        return ",".join([
            str(step), fmt(self.base_loss), fmt(self.chosen_lr), fmt(self.post_loss),
            fmt(self.post_loss - self.base_loss), fmt(self.step_delta),
            fmt(self.total_delta), str(self.boundary_bits), fmt(self.grad_max),
        ]) + "\n"


def adaptive_step(model, x, target, original, lrs, *, rounds=64, progress=None):
    """One gradient evaluation followed by a full learning-rate grid search.

    Like the float64 version, ties choose the first candidate; no zero-step
    candidate is added, and the winning candidate is accepted even if worse
    than the current point. This preserves the original update rule.
    """
    if not lrs:
        raise ValueError("at least one learning-rate candidate is required")
    ctx = model.ctx
    if any(not ctx.isfinite(lr) or lr <= 0 for lr in lrs):
        raise ValueError("learning-rate candidates must be finite and positive")
    base = model.evaluate(x, target, rounds=rounds)
    if not ctx.isfinite(base.loss) or not all(ctx.isfinite(g) for g in base.gradient):
        raise ArithmeticError("non-finite base loss or input gradient")
    grad_max = max(abs(g) for g in base.gradient)
    direction = [ctx.sign(g) for g in base.gradient]
    best_loss = best_lr = chosen = None
    for index, lr in enumerate(lrs, 1):
        candidate = projected_step(ctx, x, direction, lr)
        result = model.evaluate(candidate, target, backward=False, rounds=rounds)
        if not ctx.isfinite(result.loss):
            raise ArithmeticError(f"non-finite candidate loss at learning rate {lr}")
        if best_loss is None or result.loss < best_loss:
            best_loss, best_lr, chosen = result.loss, lr, candidate
        if progress is not None:
            progress(index, len(lrs), lr, result.loss, best_lr, best_loss)
    return StepResult(
        chosen=chosen, base_loss=base.loss, chosen_lr=best_lr, post_loss=best_loss,
        step_delta=max(abs(a-b) for a, b in zip(chosen, x)),
        total_delta=max(abs(a-b) for a, b in zip(chosen, original)),
        boundary_bits=sum(v == 0 or v == 1 for v in chosen), grad_max=grad_max,
    )


def run(args):
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 1 <= args.rounds <= 64:
        raise ValueError("--rounds must be between 1 and 64")
    model = SHA256MP(precision=args.precision, weights=args.weights)
    ctx = model.ctx
    lrs = candidate_lrs(ctx, args.min_lr_exponent)
    message, target_bytes, x, target = initial_problem(ctx, args.seed)
    original = list(x)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"precision_bits={ctx.prec} device=cpu rounds={args.rounds} "
          f"steps={args.steps} candidates={len(lrs)}", flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"optimizer=per-step sign-SGD steps={args.steps} device=cpu "
                  f"precision_bits={ctx.prec} rounds={args.rounds} seed={args.seed}\n")
        log.write(f"message={message.hex()}\ntarget={target_bytes.hex()}\n")
        log.write("target_encoding=big-endian bytes; bit_order=LSB-first within each word\n")
        log.write(f"weights={Path(args.weights)}\n")
        log.write(f"learning_rate_candidates={len(lrs)} min_lr={ctx.nstr(lrs[0], 5)} "
                  f"max_lr={ctx.nstr(lrs[-1], 5)}\n")
        log.write("step,base_loss,chosen_lr,post_loss,loss_change,step_delta,"
                  "total_delta,boundary_bits,grad_max\n")
        log.flush()
        seen_states = {}
        rows = []
        step = 1
        while step <= args.steps:
            key = state_key(x)
            if key in seen_states:
                cycle_start = seen_states[key]
                cycle_rows = rows[cycle_start - 1:step - 1]
                cycle_len = len(cycle_rows)
                marker = (f"cycle_detected start_step={cycle_start} repeat_at_step={step} "
                          f"period={cycle_len}; repeating exact rows without reevaluation\n")
                log.write(marker)
                print(marker.strip(), flush=True)
                while step <= args.steps:
                    row = cycle_rows[(step - cycle_start) % cycle_len]
                    repeated = f"{step},{row.split(',', 1)[1]}"
                    log.write(repeated)
                    if step % 10 == 0 or step == args.steps:
                        log.flush()
                        print(repeated.strip(), flush=True)
                    step += 1
                break
            seen_states[key] = step
            step_started = time.perf_counter()
            print(f"step={step}: computing forward/backward, then {len(lrs)} candidates...",
                  flush=True)

            def progress(index, count, lr, loss, best_lr, best_loss):
                if index == 1 or index % 5 == 0 or index == count:
                    print(f"step={step} candidate={index}/{count} "
                          f"best_lr={ctx.nstr(best_lr, 5)} best_loss={ctx.nstr(best_loss, ctx.dps+5)} "
                          f"elapsed_s={time.perf_counter()-step_started:.1f}", flush=True)

            result = adaptive_step(model, x, target, original, lrs,
                                   rounds=args.rounds, progress=progress)
            line = result.row(step, ctx)
            log.write(line)
            log.flush()
            rows.append(line)
            print(line.strip(), flush=True)
            x = result.chosen
            step += 1
    print(f"log saved: {log_path} (elapsed_s={time.perf_counter()-started:.1f})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto",
                        help="mpmath runs on CPU; auto also selects CPU")
    parser.add_argument("--precision", type=int, default=128,
                        help="binary significand precision, at least 113 (default: 128)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--rounds", type=int, default=64,
                        help="use fewer than 64 rounds only for debugging")
    parser.add_argument("--min-lr-exponent", type=int, default=-20,
                        help="grid starts at 10**exponent; default -20 gives 39 candidates")
    parser.add_argument("--log", type=Path, default=Path("sha256_xor_elu_scaled_sgd_adaptive_128.log"))
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
