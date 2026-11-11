"""Independent hash, PyTorch autograd, and finite-difference comparisons."""

import hashlib
import itertools
import random
import unittest

from mpmath import mp
import torch

from sha256_nn import SHA256NN
from sha256_nn_fractional import FractionalSHA256NN
from sha256_nn_mpmath import FractionalSHA256MP, _Tape, bits_to_bytes, bytes_to_bits


class FractionalSHA256MPTests(unittest.TestCase):
    def test_gate_values_and_jacobians_including_relu_boundaries(self):
        model = FractionalSHA256MP()
        reference = SHA256NN().double()
        for name, arity in (("xor_and", 2), ("prefix", 4), ("xor3", 3),
                            ("choose", 3), ("majority", 3)):
            samples = list(itertools.product((0, 0.25, 0.5, 1), repeat=arity))
            x = torch.tensor(samples, dtype=torch.float64, requires_grad=True)
            expected = getattr(reference, name)(*x.unbind(-1))
            seeds = torch.arange(1, expected.shape[-1]+1, dtype=torch.float64)
            (expected * seeds).sum().backward()
            for index, sample in enumerate(samples):
                tape = _Tape(model.ctx)
                inputs = [tape.push(model.ctx.mpf(v)) for v in sample]
                outputs = model.gates[name](tape, *inputs)
                actual = [tape.values[i] for i in outputs]
                gradients = tape.backward((node, model.ctx.mpf(j+1))
                                          for j, node in enumerate(outputs))
                self.assertEqual(actual, expected[index].tolist(), (name, sample))
                self.assertEqual([gradients[i] for i in inputs], x.grad[index].tolist(),
                                 (name, sample))

    def test_repeated_parent_accumulation(self):
        model = FractionalSHA256MP()
        tape = _Tape(model.ctx)
        node = tape.push(model.ctx.mpf("0.7"))
        outputs = model.gates["xor3"](tape, node, node, node)
        gradients = tape.backward([(outputs[0], model.ctx.one)])
        self.assertEqual(gradients[node], 3)

    def test_addition_gradient_finite_difference(self):
        model = FractionalSHA256MP()
        ctx = model.ctx
        rng = random.Random(41)
        values = [ctx.mpf(rng.randrange(1, 2**20)) / 2**20 for _ in range(64)]

        def compute(x, recording=False):
            tape = _Tape(ctx, recording)
            nodes = [tape.push(v) for v in x]
            outputs = model._add_words(tape, nodes[:32], nodes[32:])
            seeds = [ctx.mpf(i+1) / 32 for i in range(32)]
            value = ctx.fsum(tape.values[node] * seed for node, seed in zip(outputs, seeds))
            if recording:
                grad = tape.backward(zip(outputs, seeds))
                return value, [grad[i] for i in nodes]
            return value

        _, gradient = compute(values, True)
        epsilon = ctx.ldexp(ctx.one, -60)
        for index in (0, 7, 15, 31, 32, 48, 63):
            plus, minus = list(values), list(values)
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical = (compute(plus) - compute(minus)) / (2 * epsilon)
            self.assertLess(abs(gradient[index] - numerical), ctx.mpf("1e-16"), index)

    def test_full_binary_hashes_both_modes(self):
        compression = FractionalSHA256MP("compression")
        message = b"abc"
        padded = message + b"\x80" + bytes(55-len(message)) + (8*len(message)).to_bytes(8, "big")
        self.assertEqual(bits_to_bytes(compression.forward(bytes_to_bits(padded))),
                         hashlib.sha256(message).digest())
        raw = bytes(range(64))
        model = FractionalSHA256MP("message")
        self.assertEqual(bits_to_bytes(model.forward(bytes_to_bits(raw))), hashlib.sha256(raw).digest())

    def test_full_fractional_forward_and_backward_against_pytorch(self):
        # Dyadic inputs with 20 fraction bits keep all small integer-affine
        # forward intermediates exact in float64. This avoids comparing different
        # ReLU regions caused by rounding a high-precision input down to float64.
        rng = random.Random(123)
        inputs = [rng.randrange(2**20) / 2**20 for _ in range(512)]
        target = bytes_to_bits(hashlib.sha256(b"abc").digest())
        x = torch.tensor(inputs, dtype=torch.float64, requires_grad=True)
        output = FractionalSHA256NN().double()(x)
        loss = ((output - torch.tensor(target, dtype=torch.float64)) ** 2).mean()
        loss.backward()
        model = FractionalSHA256MP()
        result = model.evaluate(inputs, target)
        self.assertEqual(result.output, output.tolist())
        self.assertAlmostEqual(float(result.loss), loss.item(), places=14)
        gradient = torch.tensor([float(g) for g in result.gradient], dtype=torch.float64)
        self.assertTrue(torch.isfinite(gradient).all().item())
        torch.testing.assert_close(gradient, x.grad, rtol=1e-10, atol=0)

    def test_precision_and_invalid_inputs(self):
        before = mp.prec
        model = FractionalSHA256MP(precision=128)
        other = FractionalSHA256MP(precision=113)
        self.assertEqual(model.ctx.prec, 128)
        self.assertEqual(other.ctx.prec, 113)
        self.assertEqual(mp.prec, before)
        for bad in ([0]*511, ["nan"]*512, ["inf"]*512, ["-0.1"]*512,
                    ["1.1"]*512, [1j]*512):
            with self.assertRaises(ValueError):
                model.forward(bad)
        with self.assertRaises(ValueError):
            model.backward([0]*512, [0]*255)
        with self.assertRaises(ValueError):
            model.evaluate([0]*512, ["nan"]*256)
        with self.assertRaises(ValueError):
            FractionalSHA256MP(precision=1)
        with self.assertRaises(ValueError):
            FractionalSHA256MP("invalid")


if __name__ == "__main__":
    unittest.main()
