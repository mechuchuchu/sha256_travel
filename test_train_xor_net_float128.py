"""Regression checks for best-checkpoint export and its reported metrics."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mpmath import mp

from sha256_mpmath_backward import SHA256MP
import train_xor_net_float128 as trainer


class XORExportTests(unittest.TestCase):
    def check_training_export(self, steps, lr, expect_last):
        evaluations = []
        original = trainer.loss_and_gradient

        def record(params):
            loss, gradient = original(params)
            evaluations.append((loss, list(trainer.flatten(params))))
            return loss, gradient

        with tempfile.TemporaryDirectory() as directory, mp.workprec(128):
            output = Path(directory) / "weights.bin"
            argv = ["train_xor_net_float128.py", "--steps", str(steps),
                    "--lr", lr, "--output", str(output),
                    "--history", str(Path(directory) / "history.csv")]
            with patch("sys.argv", argv), \
                 patch.object(trainer, "loss_and_gradient", side_effect=record), \
                 redirect_stdout(io.StringIO()):
                trainer.main()

            self.assertEqual(len(evaluations), steps + 1)
            best_index = min(range(len(evaluations)), key=lambda i: evaluations[i][0])
            if expect_last:
                self.assertEqual(best_index, steps)
            else:
                self.assertLess(best_index, steps)
                self.assertLess(evaluations[best_index][0], evaluations[-1][0])
            expected = b"".join(trainer.to_binary128(v) for v in evaluations[best_index][1])
            self.assertEqual(output.read_bytes(), expected)

            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["checkpoint"], "best")
            self.assertEqual(metadata["best_step"], best_index)
            self.assertEqual(mp.mpf(metadata["best_mse"]), evaluations[best_index][0])

            # Use the consuming engine's independent decoder and XOR evaluator.
            model = SHA256MP(precision=metadata["evaluation_precision_bits"], weights=output)
            ctx = model.ctx
            outputs = [model.network.evaluate(ctx.mpf(a), ctx.mpf(b), False)[0]
                       for a, b in trainer.INPUTS]
            loss = ctx.fsum((v - t) ** 2 for v, t in zip(outputs, trainer.TARGETS)) / 4
            self.assertLess(abs(loss - ctx.mpf(metadata["final_mse"])), ctx.mpf("1e-48"))
            for actual, reported in zip(outputs, metadata["xor_outputs_00_01_10_11"]):
                self.assertLess(abs(actual - ctx.mpf(reported)), ctx.mpf("1e-48"))

    def test_exports_earlier_best_when_final_iterate_is_worse(self):
        self.check_training_export(8, "1", expect_last=False)

    def test_includes_improved_last_update(self):
        self.check_training_export(1, "0.001", expect_last=True)

    def test_embedded_weights_match_binary_file(self):
        module = ast.parse((trainer.HERE / "sha256_xor_elu_scaled_sgd_adaptive.py").read_text())
        embedded = next(ast.literal_eval(node.value) for node in module.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name)
                                and target.id == "_XOR_NET_F128_HEX"
                                for target in node.targets))
        self.assertEqual(bytes.fromhex(embedded), (trainer.HERE / "xor_net_f128.bin").read_bytes())


if __name__ == "__main__":
    unittest.main()
