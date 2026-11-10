"""Real spawn-process checks for precision, ordering, and optimizer parity."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import io
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sha256_xor_elu_scaled_sgd_adaptive_128 as serial
import sha256_xor_elu_scaled_sgd_adaptive_128_multiprocess as parallel
from sha256_mpmath_backward import DEFAULT_WEIGHTS, SHA256MP
from test_sha256_adaptive_128 import Quadratic


def process_pool(precision):
    return ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"),
        initializer=parallel.initialize_worker,
        initargs=(precision, str(DEFAULT_WEIGHTS)),
    )


class MultiprocessTests(unittest.TestCase):
    def test_spawn_matches_every_candidate_and_reuses_pool_across_steps(self):
        model = SHA256MP(precision=160)
        ctx = model.ctx
        x = [ctx.mpf(2 + i % 7) / 10 + ctx.mpf("1e-35") for i in range(512)]
        original = list(x)
        target = [i % 2 for i in range(256)]
        lrs = [ctx.mpf(value) for value in ("1e-30", "3e-30", "0.01")]
        with process_pool(ctx.prec) as executor:
            for step in (1, 2):
                serial_progress, parallel_progress = [], []
                expected = serial.adaptive_step(
                    model, x, target, original, lrs, rounds=2,
                    progress=lambda *row: serial_progress.append(row))
                actual = parallel.adaptive_step(
                    model, x, target, original, lrs, rounds=2, executor=executor,
                    progress=lambda *row: parallel_progress.append(row))
                self.assertEqual(parallel_progress, serial_progress)
                self.assertEqual(parallel.state_key(actual.chosen), serial.state_key(expected.chosen))
                self.assertEqual(actual.row(step, ctx), expected.row(step, ctx))
                x = actual.chosen

    def test_spawn_preserves_fractional_target_and_first_candidate_tie(self):
        model = SHA256MP(precision=128)
        ctx = model.ctx
        x = [ctx.mpf("0.3") + ctx.mpf("1e-30")] * 512
        target = model.evaluate(x, [0] * 256, rounds=1, backward=False).output
        # Exact target gives zero gradients: every learning rate ties at zero.
        # A later, smaller rate must not displace the first candidate.
        lrs = [ctx.mpf(value) for value in ("0.03", "1e-30", "0.1")]
        with process_pool(ctx.prec) as executor:
            actual = parallel.adaptive_step(model, x, target, x, lrs, rounds=1,
                                            executor=executor)
        self.assertEqual(actual.post_loss, 0)
        self.assertEqual(actual.chosen_lr, lrs[0])
        self.assertEqual(actual.chosen, x)

    def test_worker_error_propagates_and_pool_remains_usable(self):
        model = SHA256MP()
        x = [model.ctx.zero] * 512
        with process_pool(model.ctx.prec) as executor:
            with self.assertRaisesRegex(ValueError, "expected 512 input"):
                list(parallel.candidate_losses(model, [x[:-1]], [0] * 256, 1, executor))
            loss, = parallel.candidate_losses(model, [x], [0] * 256, 1, executor)
        self.assertTrue(model.ctx.isfinite(loss))

    def test_cli_log_rows_match_original(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            args = SimpleNamespace(steps=2, rounds=1, precision=128,
                                   weights=DEFAULT_WEIGHTS, seed=20261110,
                                   min_lr_exponent=-2, log=directory / "serial.log")
            with redirect_stdout(io.StringIO()):
                serial.run(args)
            log_path = directory / "parallel.log"
            result = subprocess.run(
                [sys.executable, str(Path(parallel.__file__)), "--steps", "2",
                 "--rounds", "1", "--min-lr-exponent", "-2", "--workers", "2",
                 "--log", str(log_path)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            serial_rows = [line for line in args.log.read_text().splitlines()
                           if line[:1].isdigit()]
            parallel_lines = log_path.read_text().splitlines()
            parallel_rows = [line for line in parallel_lines if line[:1].isdigit()]
            self.assertEqual(parallel_rows, serial_rows)
            self.assertIn("workers=2", parallel_lines[0])

    def test_single_worker_keeps_cycle_replay_without_creating_pool(self):
        model = Quadratic()
        ctx = model.ctx
        x, target = [ctx.mpf("0.5")], [ctx.mpf("0.6")]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(steps=7, rounds=64, precision=128, workers=1,
                                   weights=DEFAULT_WEIGHTS, seed=1,
                                   min_lr_exponent=-20, log=Path(directory) / "run.log")
            with patch.object(parallel, "SHA256MP", return_value=model), \
                 patch.object(parallel, "initial_problem", return_value=(b"a", b"b", x, target)), \
                 patch.object(parallel, "candidate_lrs", return_value=[ctx.mpf("0.25")]), \
                 patch.object(parallel, "ProcessPoolExecutor") as executor, \
                 redirect_stdout(io.StringIO()):
                parallel.run(args)
            executor.assert_not_called()
            lines = args.log.read_text().splitlines()
        self.assertEqual(model.calls, 4)
        self.assertTrue(any("start_step=1 repeat_at_step=3 period=2" in line for line in lines))
        rows = [line.split(",") for line in lines if line[:1].isdigit()]
        self.assertEqual([int(row[0]) for row in rows], list(range(1, 8)))
        for i, row in enumerate(rows):
            self.assertEqual(row[1:], rows[i % 2][1:])


if __name__ == "__main__":
    unittest.main()
