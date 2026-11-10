"""Numerical and structural checks for the mpmath SHA-256 surrogate."""
import hashlib
import unittest

from mpmath import mp

from sha256_mpmath_backward import (
    SHA256MP, Tape, bytes_to_bits, decode_binary128, pad_message,
)


class PolynomialXOR:
    """Exact Boolean XOR oracle, used only to validate the SHA wiring."""

    @staticmethod
    def evaluate(a, b, derivatives=True):
        return a + b - 2 * a * b, 1 - 2 * b, 1 - 2 * a


class SHA256MPTests(unittest.TestCase):
    def test_binary128_preserves_bits_beyond_float64(self):
        ctx = mp.clone()
        ctx.prec = 128
        raw = ((16383 << 112) | 1).to_bytes(16, "big")
        value = decode_binary128(raw, ctx)
        self.assertEqual(value, 1 + ctx.ldexp(ctx.one, -112))
        self.assertNotEqual(value, ctx.mpf(float(value)))
        negative = ((1 << 127) | (16383 << 112) | 1).to_bytes(16, "big")
        self.assertEqual(decode_binary128(negative, ctx), -value)
        self.assertEqual(decode_binary128((1).to_bytes(16, "big"), ctx),
                         ctx.ldexp(ctx.one, -16494))
        with self.assertRaises(ValueError):
            decode_binary128((0x7FFF << 112).to_bytes(16, "big"), ctx)

    def test_xor_derivatives_against_high_precision_central_difference(self):
        model = SHA256MP(precision=180)
        ctx = model.ctx
        h = ctx.mpf("1e-20")
        for av, bv in (("0", "0"), ("0", "1"), ("1", "0"),
                       ("1", "1"), ("0.23", "0.71")):
            a, b = ctx.mpf(av), ctx.mpf(bv)
            _, da, db = model.network.evaluate(a, b)
            numerical_a = (model.network.evaluate(a+h, b, False)[0]
                           - model.network.evaluate(a-h, b, False)[0]) / (2*h)
            numerical_b = (model.network.evaluate(a, b+h, False)[0]
                           - model.network.evaluate(a, b-h, False)[0]) / (2*h)
            self.assertLess(abs(da - numerical_a), ctx.mpf("1e-30"))
            self.assertLess(abs(db - numerical_b), ctx.mpf("1e-30"))

    def test_sha_structure_matches_hashlib_with_boolean_xor(self):
        model = SHA256MP()
        model.network = PolynomialXOR()
        for message in (b"", b"abc", bytes(range(55))):
            expected = bytes_to_bits(hashlib.sha256(message).digest())
            result = model.evaluate(bytes_to_bits(pad_message(message)), expected,
                                    backward=False)
            self.assertEqual(result.output, expected)
            self.assertEqual(result.loss, 0)
            self.assertIsNone(result.gradient)

    def test_reverse_accumulates_shared_parents_and_seeds(self):
        model = SHA256MP()
        tape = Tape(model.ctx, model.network)
        x = tape.push(model.ctx.mpf("0.23"))
        y = tape.xor(x, x)
        _, da, db = model.network.evaluate(tape.values[x], tape.values[x])
        gradients = tape.backward([(y, model.ctx.one), (y, model.ctx.one)])
        self.assertLess(abs(gradients[x] - 2 * (da + db)), model.ctx.mpf("1e-36"))

    def test_sha_gradient_against_directional_finite_difference(self):
        model = SHA256MP()
        ctx = model.ctx
        # Fractional points avoid Boolean boundaries; two rounds exercise the
        # carry recurrence, both sigma functions, choose/majority, and feedforward.
        x = [ctx.mpf(2 + i % 7) / 10 for i in range(512)]
        direction = [ctx.mpf(i % 5 - 2) / 3 for i in range(512)]
        target = [i % 2 for i in range(256)]
        h = ctx.mpf("1e-12")
        result = model.evaluate(x, target, rounds=2)
        plus = model.evaluate([v + h*d for v, d in zip(x, direction)], target,
                              rounds=2, backward=False)
        minus = model.evaluate([v - h*d for v, d in zip(x, direction)], target,
                               rounds=2, backward=False)
        numerical = (plus.loss - minus.loss) / (2*h)
        analytic = ctx.fsum(g*d for g, d in zip(result.gradient, direction))
        self.assertTrue(all(ctx.isfinite(g) for g in result.gradient))
        self.assertGreater(abs(analytic), ctx.mpf("1e-8"))
        self.assertLess(abs(analytic - numerical) / abs(analytic), ctx.mpf("1e-16"))
        # Unused input words in a two-round compression must have zero gradient.
        self.assertTrue(all(g == 0 for g in result.gradient[64:]))

    def test_validation_and_context_isolation(self):
        original_precision = mp.prec
        model = SHA256MP()
        self.assertEqual(model.ctx.prec, 128)
        self.assertEqual(mp.prec, original_precision)
        with self.assertRaises(ValueError):
            SHA256MP(precision=64)
        with self.assertRaises(ValueError):
            pad_message(b"x" * 56)
        with self.assertRaises(ValueError):
            model.evaluate([0] * 511, [0] * 256)
        with self.assertRaises(ValueError):
            model.evaluate([0] * 512, [0] * 256, rounds=0)


if __name__ == "__main__":
    unittest.main()
