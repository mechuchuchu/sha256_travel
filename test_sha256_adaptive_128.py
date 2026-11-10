"""Checks for high-precision adaptive selection and exact-state cycle replay."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mpmath import mp
import numpy as np

import sha256_xor_elu_scaled_sgd_adaptive_128 as adaptive


class Quadratic:
    def __init__(self):
        self.ctx = mp.clone()
        self.ctx.prec = 128
        self.calls = 0

    def evaluate(self, x, target, *, backward=True, rounds=64):
        self.calls += 1
        errors = [a-b for a, b in zip(x, target)]
        return SimpleNamespace(loss=self.ctx.fsum(e*e for e in errors)/len(x),
                               gradient=[2*e/len(x) for e in errors] if backward else None)


class Adaptive128Tests(unittest.TestCase):
    def test_grid_and_seed_match_original_problem(self):
        ctx = Quadratic().ctx
        lrs = adaptive.candidate_lrs(ctx)
        self.assertEqual(len(lrs), 39)
        self.assertEqual(lrs[0], ctx.mpf("1e-20"))
        self.assertEqual(lrs[-1], ctx.mpf("1e-1"))
        self.assertEqual(lrs, sorted(set(lrs)))
        message, target_bytes, x, target = adaptive.initial_problem(ctx, 20261110)
        self.assertEqual(message.hex(), "8a1587c509b446a0bf64f8d289d96124cf5af138c3b242a31049af10a2b64e86")
        rng = np.random.default_rng(20261110)
        self.assertEqual(message, rng.bytes(32))
        words = np.frombuffer(rng.bytes(32), dtype=">u4").astype(np.uint32)
        expected = [int((int(word) >> bit) & 1) for word in words for bit in range(32)]
        self.assertEqual(target, expected)
        self.assertEqual(target_bytes, words.astype(">u4").tobytes())
        self.assertEqual(len(x), 512)
        self.assertEqual(len(adaptive.candidate_lrs(ctx, -40)), 79)

    def test_selection_and_logging_keep_sub_float64_changes(self):
        model = Quadratic()
        ctx = model.ctx
        x = [ctx.mpf("0.5")]
        target = [x[0] - ctx.mpf("3e-20")]
        lrs = [ctx.mpf(s) for s in ("1e-20", "3e-20", "1e-19")]
        result = adaptive.adaptive_step(model, x, target, x, lrs)
        self.assertEqual(result.chosen_lr, lrs[1])
        self.assertEqual(result.chosen, target)
        self.assertEqual(result.post_loss, 0)
        self.assertNotEqual(adaptive.state_key(x), adaptive.state_key(result.chosen))
        self.assertEqual(float(x[0]), float(result.chosen[0]))
        row = result.row(1, ctx).strip().split(",")
        self.assertEqual(ctx.mpf(row[5]), result.step_delta)
        self.assertGreater(ctx.mpf(row[5]), 0)

    def test_clamping_zero_gradient_and_first_tie(self):
        model = Quadratic()
        ctx = model.ctx
        x = [ctx.zero, ctx.one, ctx.mpf("0.5")]
        target = [-ctx.one, 2*ctx.one, ctx.mpf("0.5")]
        lrs = [ctx.mpf("0.1"), ctx.mpf("0.3")]
        result = adaptive.adaptive_step(model, x, target, x, lrs)
        self.assertEqual(result.chosen, x)
        self.assertEqual(result.chosen_lr, lrs[0])
        self.assertEqual(result.boundary_bits, 2)
        self.assertEqual(result.step_delta, 0)

    def test_accepts_best_candidate_even_if_loss_increases(self):
        model = Quadratic()
        ctx = model.ctx
        x, target = [ctx.mpf("0.5")], [ctx.mpf("0.51")]
        result = adaptive.adaptive_step(model, x, target, x, [ctx.mpf("0.1")])
        self.assertGreater(result.post_loss, result.base_loss)
        self.assertEqual(result.chosen, [ctx.mpf("0.6")])

    def test_cycle_replays_correct_phase_without_evaluating_again(self):
        model = Quadratic()
        ctx = model.ctx
        x, target = [ctx.mpf("0.5")], [ctx.mpf("0.6")]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(steps=7, rounds=64, precision=128,
                                   weights=adaptive.DEFAULT_WEIGHTS, seed=1,
                                   min_lr_exponent=-20, log=Path(directory)/"nested"/"run.log")
            with patch.object(adaptive, "SHA256MP", return_value=model), \
                 patch.object(adaptive, "initial_problem", return_value=(b"a", b"b", x, target)), \
                 patch.object(adaptive, "candidate_lrs", return_value=[ctx.mpf("0.25")]), \
                 redirect_stdout(io.StringIO()):
                adaptive.run(args)
            lines = args.log.read_text().splitlines()
        self.assertEqual(model.calls, 4)  # two real steps, each base + one candidate
        self.assertTrue(any("start_step=1 repeat_at_step=3 period=2" in line for line in lines))
        rows = [line.split(",") for line in lines if line[0].isdigit()]
        self.assertEqual([int(row[0]) for row in rows], list(range(1, 8)))
        for i, row in enumerate(rows):
            self.assertEqual(row[1:], rows[i % 2][1:])


if __name__ == "__main__":
    unittest.main()
