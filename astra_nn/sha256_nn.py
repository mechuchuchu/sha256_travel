"""Exact SHA-256 on binary inputs using fixed affine/ReLU neural gates.

External bits are MSB first (the same order as bytes written in binary).
Internal 32-bit words are LSB first. No integer hash/bitwise operations are
used on the input in the model's forward path. Rotations are fixed wiring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


K = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)
IV = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)


def _word_bits(words: tuple[int, ...]) -> Tensor:
    """Encode compile-time constants, not model inputs."""
    return torch.tensor(
        [[(word >> bit) & 1 for bit in range(32)] for word in words],
        dtype=torch.float32,
    )


def bytes_to_bits(data: bytes) -> Tensor:
    """Bytes to a flat float32 tensor, most significant bit first per byte."""
    return torch.tensor(
        [(byte >> bit) & 1 for byte in data for bit in range(7, -1, -1)],
        dtype=torch.float32,
    )


def bits_to_bytes(bits: Tensor) -> bytes:
    """Decode a single binary vector without rounding or thresholding."""
    if bits.ndim != 1 or bits.numel() % 8:
        raise ValueError("expected a flat bit vector with a multiple of 8 bits")
    if not torch.all((bits == 0) | (bits == 1)).item():
        raise ValueError("expected exact binary values (0 or 1)")
    values = bits.detach().cpu().tolist()
    return bytes(
        sum(int(values[start + offset]) << (7 - offset) for offset in range(8))
        for start in range(0, len(values), 8)
    )


def pad_single_block(message: bytes) -> bytes:
    """Standard padding for a message of 0..55 bytes, for compression mode."""
    if len(message) > 55:
        raise ValueError("one-block padding requires a message of at most 55 bytes")
    return message + b"\x80" + bytes(55 - len(message)) + (8 * len(message)).to_bytes(8, "big")


class ReLUGate(nn.Module):
    """A two-affine-layer MLP with explicit, shared, frozen parameters."""

    def __init__(self, weight_in, bias_in, weight_out):
        super().__init__()
        self.weight_in = nn.Parameter(torch.tensor(weight_in, dtype=torch.float32), requires_grad=False)
        self.bias_in = nn.Parameter(torch.tensor(bias_in, dtype=torch.float32), requires_grad=False)
        self.weight_out = nn.Parameter(torch.tensor(weight_out, dtype=torch.float32), requires_grad=False)
        self.bias_out = nn.Parameter(torch.zeros(len(weight_out)), requires_grad=False)

    def forward(self, *inputs: Tensor) -> Tensor:
        x = torch.stack(torch.broadcast_tensors(*inputs), dim=-1)
        return F.linear(F.relu(F.linear(x, self.weight_in, self.bias_in)), self.weight_out, self.bias_out)


class SHA256NN(nn.Module):
    """Fixed 512 -> 256 bit network with tied weights and fixed SHA wiring.

    mode="message": the input is a raw 64-byte message; applies two blocks.
    mode="compression": the input is one 512-bit block; applies the standard
    IV and one compression including feed-forward, without adding padding.

    forward accepts (..., 512) and returns (..., 256), both MSB first.
    Exactness is promised for binary inputs with the provided weights only.
    """

    def __init__(self, mode: str = "message"):
        super().__init__()
        if mode not in ("message", "compression"):
            raise ValueError("mode must be 'message' or 'compression'")
        self.mode = mode
        # For s=x+y, XOR=s-2*ReLU(s-1), AND=ReLU(s-1).
        self.xor_and = ReLUGate([[1, 1], [1, 1]], [0, -1], [[1, -2], [0, 1]])
        # Combine carry intervals: G=G_hi+P_hi*G_lo, P=P_hi*P_lo.
        # The two contributions to G are mutually exclusive on valid bits.
        self.prefix = ReLUGate(
            [[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 1]],
            [0, -1, -1],
            [[1, 1, 0], [0, 0, 1]],
        )
        self.xor3 = ReLUGate([[1, 1, 1]] * 3, [0, -1, -2], [[1, -2, 2]])
        self.choose = ReLUGate([[1, 1, 0], [-1, 0, 1]], [-1, 0], [[1, 1]])
        self.majority = ReLUGate([[1, 1, 1]] * 2, [-1, -2], [[1, -1]])
        self.register_buffer("initial_state", _word_bits(IV))
        self.register_buffer("round_constants", _word_bits(K))
        # A 64-byte message has a second, constant padding block:
        # 0x80, 55 zero bytes, and the 64-bit big-endian bit length 512.
        self.register_buffer("padding_block", _word_bits((0x80000000,) + (0,) * 14 + (512,)))

    def add_words(self, x: Tensor, y: Tensor) -> Tensor:
        """32-bit modular addition, with five parallel carry-prefix layers."""
        pg = self.xor_and(x, y)
        propagate, generate = pg.unbind(-1)
        original_propagate = propagate
        for distance in (1, 2, 4, 8, 16):
            combined = self.prefix(
                generate[..., distance:], propagate[..., distance:],
                generate[..., :-distance], propagate[..., :-distance],
            )
            generate = torch.cat((generate[..., :distance], combined[..., 0]), dim=-1)
            propagate = torch.cat((propagate[..., :distance], combined[..., 1]), dim=-1)
        carry = F.pad(generate[..., :-1], (1, 0))
        return self.xor_and(original_propagate, carry)[..., 0]

    def _add_many(self, *words: Tensor) -> Tensor:
        total = words[0]
        for word in words[1:]:
            total = self.add_words(total, word)
        return total

    def _sigma(self, x: Tensor, rotations: tuple[int, int, int], shift: bool = False) -> Tensor:
        a, b, c = rotations
        third = F.pad(x[..., c:], (0, c)) if shift else x.roll(-c, dims=-1)
        return self.xor3(x.roll(-a, dims=-1), x.roll(-b, dims=-1), third)[..., 0]

    def _compress(self, block: Tensor, state: Tensor) -> Tensor:
        schedule = list(block.unbind(-2))
        for t in range(16, 64):
            schedule.append(self._add_many(
                schedule[t - 16], self._sigma(schedule[t - 15], (7, 18, 3), shift=True),
                schedule[t - 7], self._sigma(schedule[t - 2], (17, 19, 10), shift=True),
            ))
        a, b, c, d, e, f, g, h = state.unbind(-2)
        for t in range(64):
            t1 = self._add_many(
                h, self._sigma(e, (6, 11, 25)), self.choose(e, f, g)[..., 0],
                self.round_constants[t], schedule[t],
            )
            t2 = self.add_words(self._sigma(a, (2, 13, 22)), self.majority(a, b, c)[..., 0])
            a, b, c, d, e, f, g, h = self.add_words(t1, t2), a, b, c, self.add_words(d, t1), e, f, g
        return self.add_words(state, torch.stack((a, b, c, d, e, f, g, h), dim=-2))

    def _validate_input(self, bits: Tensor) -> None:
        if bits.ndim < 1 or bits.shape[-1] != 512:
            raise ValueError("expected shape (..., 512)")
        if bits.is_complex() or not torch.all((bits == 0) | (bits == 1)).item():
            raise ValueError("input must contain only exact binary values (0 or 1)")

    def forward(self, bits: Tensor) -> Tensor:
        self._validate_input(bits)
        # Casting remains differentiable for floating input tensors.
        bits = bits.to(device=self.initial_state.device, dtype=self.initial_state.dtype)
        batch_shape = bits.shape[:-1]
        block = bits.reshape(*batch_shape, 16, 32).flip(-1)
        state = self.initial_state.expand(*batch_shape, 8, 32)
        state = self._compress(block, state)
        if self.mode == "message":
            padding = self.padding_block.expand(*batch_shape, 16, 32)
            state = self._compress(padding, state)
        return state.flip(-1).reshape(*batch_shape, 256)

    def save(self, path: str | Path) -> None:
        torch.save({
            "format_version": 1,
            "mode": self.mode,
            "bit_order": "msb_first",
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "SHA256NN":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("bit_order") != "msb_first":
            raise ValueError("unsupported SHA256NN checkpoint format")
        dtype = checkpoint["state_dict"]["initial_state"].dtype
        model = cls(mode=checkpoint["mode"]).to(device=device, dtype=dtype)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.eval()


def export_weights(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model = SHA256NN()
    model.save(directory / "sha256_512_256.pt")
    SHA256NN(mode="compression").save(directory / "sha256_compression_512_256.pt")
    # The JSON contains every gate parameter and every constant buffer.
    metadata = {
        "format_version": 1,
        "input_bits": 512,
        "output_bits": 256,
        "bit_order": "msb_first",
        "architecture": "shared affine-ReLU-affine gates with fixed SHA-256 wiring",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "modes": {
            "message": "64 raw bytes, standard padding, two compression blocks",
            "compression": "one supplied block, standard IV, no added padding",
        },
        "state_dict": {name: tensor.tolist() for name, tensor in model.state_dict().items()},
    }
    (directory / "sha256_weights.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved both checkpoints and JSON weights to {directory.resolve()}")
    print(f"Shared scalar parameters: {metadata['parameter_count']}; all frozen")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, metavar="DIRECTORY")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--message-hex", help="exactly 64 input bytes as 128 hexadecimal characters")
    args = parser.parse_args()
    if args.export is not None:
        export_weights(args.export)
    if args.message_hex is not None:
        data = bytes.fromhex(args.message_hex)
        if len(data) != 64:
            parser.error("--message-hex must encode exactly 64 bytes")
        model = SHA256NN.load(args.weights) if args.weights else SHA256NN()
        with torch.inference_mode():
            print(bits_to_bytes(model(bytes_to_bits(data))).hex())
    if args.export is None and args.message_hex is None:
        parser.error("provide --export DIRECTORY or --message-hex HEX")


if __name__ == "__main__":
    main()
