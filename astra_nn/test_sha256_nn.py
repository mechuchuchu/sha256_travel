"""Independent integer/hashlib reference checks for the exact neural circuit."""

import hashlib
import itertools
from pathlib import Path
import random
import tempfile
import unittest

import torch

from sha256_nn import SHA256NN, bits_to_bytes, bytes_to_bits, pad_single_block


class SHA256NNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.message = SHA256NN().eval()
        cls.compression = SHA256NN("compression").eval()

    def assert_hashes(self, model, inputs, messages):
        with torch.inference_mode():
            output = model(torch.stack([bytes_to_bits(data) for data in inputs]))
        self.assertEqual(tuple(output.shape), (len(inputs), 256))
        for bits, message in zip(output, messages, strict=True):
            # The decoder rejects fractional outputs; no thresholding hides errors.
            self.assertEqual(bits_to_bytes(bits), hashlib.sha256(message).digest())

    def test_gate_truth_tables(self):
        triples = list(itertools.product((0, 1), repeat=3))
        a, b, c = torch.tensor(triples, dtype=torch.float32).T
        with torch.inference_mode():
            results = {
                "xor": self.message.xor_and(a, b)[..., 0],
                "and": self.message.xor_and(a, b)[..., 1],
                "xor3": self.message.xor3(a, b, c)[..., 0],
                "choose": self.message.choose(a, b, c)[..., 0],
                "majority": self.message.majority(a, b, c)[..., 0],
            }
        expected = {
            "xor": [x ^ y for x, y, z in triples],
            "and": [x & y for x, y, z in triples],
            "xor3": [x ^ y ^ z for x, y, z in triples],
            "choose": [y if x else z for x, y, z in triples],
            "majority": [int(x + y + z >= 2) for x, y, z in triples],
        }
        for name, actual in results.items():
            self.assertEqual(actual.tolist(), expected[name], name)

    def test_modular_addition_and_carry_chains(self):
        rng = random.Random(1701)
        pairs = [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(1024)]
        pairs += [(2**length - 1, 1) for length in range(33)]
        pairs += [(0xFFFFFFFF, 0xFFFFFFFF), (0, 0), (0xAAAAAAAA, 0x55555555)]
        x, y = (
            torch.tensor([[(pair[index] >> i) & 1 for i in range(32)] for pair in pairs], dtype=torch.float32)
            for index in (0, 1)
        )
        with torch.inference_mode():
            actual = self.message.add_words(x, y)
        self.assertTrue(torch.all((actual == 0) | (actual == 1)).item())
        weights = 2 ** torch.arange(32, dtype=torch.int64)
        integers = (actual.to(torch.int64) * weights).sum(-1).tolist()
        self.assertEqual(integers, [(a + b) % 2**32 for a, b in pairs])

    def test_raw_messages(self):
        rng = random.Random(20261111)
        messages = [bytes(64), b"\xff" * 64, b"\xaa" * 64, b"\x55" * 64,
                    bytes(range(64)), bytes(reversed(range(64)))]
        messages += [rng.randbytes(64) for _ in range(64)]
        for start in range(0, len(messages), 16):
            batch = messages[start:start + 16]
            self.assert_hashes(self.message, batch, batch)

    def test_every_input_bit_position(self):
        # Each of the 512 input positions is exercised in isolation, including
        # the first and last bit of every byte and every 32-bit word.
        messages = [(1 << bit).to_bytes(64, "big") for bit in range(512)]
        for start in range(0, len(messages), 32):
            batch = messages[start:start + 32]
            self.assert_hashes(self.message, batch, batch)

    def test_padded_single_blocks(self):
        rng = random.Random(123)
        messages = [b"", b"abc", b"a" * 55] + [rng.randbytes(size) for size in range(56)]
        for start in range(0, len(messages), 16):
            batch = messages[start:start + 16]
            self.assert_hashes(self.compression, [pad_single_block(data) for data in batch], batch)

    def test_shapes_boolean_input_and_bit_order(self):
        data = bytes(range(64))
        bits = bytes_to_bits(data)
        self.assertEqual(bits_to_bytes(bits), data)
        self.assertEqual(bytes_to_bits(b"\x80\x01").tolist(), [1] + [0] * 14 + [1])
        with torch.inference_mode():
            output = self.message(bits.bool().expand(2, 2, 512))
        self.assertEqual(tuple(output.shape), (2, 2, 256))
        for digest in output.reshape(-1, 256):
            self.assertEqual(bits_to_bytes(digest), hashlib.sha256(data).digest())

    def test_checkpoint_round_trip_and_float64(self):
        model = SHA256NN("compression").double()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            loaded = SHA256NN.load(path)
        self.assertEqual(loaded.mode, "compression")
        self.assertEqual(loaded.initial_state.dtype, torch.float64)
        self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)
        self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
        self.assert_hashes(loaded, [pad_single_block(b"abc")], [b"abc"])

    def test_exported_checkpoints(self):
        directory = Path(__file__).resolve().parent
        model = SHA256NN.load(directory / "sha256_512_256.pt")
        self.assertEqual(model.mode, "message")
        self.assert_hashes(model, [bytes(64)], [bytes(64)])
        model = SHA256NN.load(directory / "sha256_compression_512_256.pt")
        self.assertEqual(model.mode, "compression")
        self.assert_hashes(model, [pad_single_block(b"abc")], [b"abc"])

    def test_invalid_inputs(self):
        for value in (0.5, -1.0, 2.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.message(torch.full((512,), value))
        for shape in ((), (511,), (1, 513), (16, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.message(torch.zeros(shape))
        with self.assertRaises(ValueError):
            SHA256NN("invalid")
        with self.assertRaises(ValueError):
            pad_single_block(bytes(56))
        with self.assertRaises(ValueError):
            bits_to_bytes(torch.full((256,), 0.5))
        with self.assertRaises(ValueError):
            bits_to_bytes(torch.zeros(7))


if __name__ == "__main__":
    unittest.main()
