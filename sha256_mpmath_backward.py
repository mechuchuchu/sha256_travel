#!/usr/bin/env python3
"""mpmath forward/backward for the learned-XOR SHA-256 relaxation.

All model values and derivatives use an isolated mpmath context (128 binary
significand bits by default). Stored IEEE binary128 weights have 113 significand
bits and are decoded directly, without a Python float conversion. This models
one compression block; fractional inputs, including padding, are differentiable.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

from mpmath import mp


HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "xor_net_f128.bin"
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
IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)


def decode_binary128(raw, ctx):
    """Decode one finite, big-endian IEEE-754 binary128 value into ctx."""
    if len(raw) != 16:
        raise ValueError("one binary128 value requires 16 bytes")
    word = int.from_bytes(raw, "big")
    exponent = (word >> 112) & 0x7FFF
    fraction = word & ((1 << 112) - 1)
    if exponent == 0x7FFF:
        raise ValueError("XOR weights must be finite")
    if exponent == 0:
        value = ctx.ldexp(ctx.mpf(fraction), -16494)
    else:
        value = ctx.ldexp(ctx.mpf((1 << 112) | fraction), exponent - 16495)
    return -value if word >> 127 else value


class XORNetwork:
    """Frozen 2-8-1 ELU network with analytic derivatives w.r.t. its inputs."""

    def __init__(self, ctx, path=DEFAULT_WEIGHTS):
        self.ctx = ctx
        raw = Path(path).read_bytes()
        if len(raw) != 528:
            raise ValueError("expected 33 binary128 XOR parameters (528 bytes)")
        values = [decode_binary128(raw[i:i + 16], ctx)
                  for i in range(0, len(raw), 16)]
        self.neurons = [(values[2*j], values[2*j+1], values[16+j], values[24+j])
                        for j in range(8)]
        self.bias = values[32]
        self.scale, self.offset = ctx.mpf("1.0001"), ctx.mpf("0.00005")

    def evaluate(self, a, b, derivatives=True):
        ctx = self.ctx
        logit = self.bias
        da = db = ctx.zero
        for wa, wb, bias, weight in self.neurons:
            pre = wa * a + wb * b + bias
            if pre > 0:
                hidden = pre
                slope = ctx.one
            else:
                hidden = ctx.expm1(pre)
                slope = ctx.exp(pre) if derivatives else ctx.zero
            logit += weight * hidden
            if derivatives:
                da += weight * slope * wa
                db += weight * slope * wb
        # Compute the sigmoid slope without subtracting a rounded probability
        # from 1, preserving tiny gradients even at large positive logits.
        ex = ctx.exp(-abs(logit))
        denominator = 1 + ex
        probability = 1 / denominator if logit >= 0 else ex / denominator
        output = self.scale * probability - self.offset
        if not derivatives:
            return output, ctx.zero, ctx.zero
        slope = self.scale * ex / denominator ** 2
        return output, slope * da, slope * db


class Tape:
    """Scalar reverse-mode tape; fused operations store only local Jacobians."""

    def __init__(self, ctx, network, recording=True):
        self.ctx, self.network, self.recording = ctx, network, recording
        self.values = []
        self.parents = []
        self.zero = self.push(ctx.zero)
        self.one = self.push(ctx.one)

    def push(self, value, parents=()):
        index = len(self.values)
        self.values.append(value)
        self.parents.append(parents if self.recording else ())
        return index

    def xor(self, a, b):
        value, da, db = self.network.evaluate(
            self.values[a], self.values[b], self.recording)
        return self.push(value, ((a, da), (b, db)))

    def xor_words(self, a, b):
        return [self.xor(x, y) for x, y in zip(a, b)]

    def xor3(self, a, b, c):
        return self.xor_words(self.xor_words(a, b), c)

    def add2(self, a, b):
        carry = self.zero
        result = []
        for bit in range(32):
            ai, bi = a[bit], b[bit]
            propagate = self.xor(ai, bi)
            result.append(self.xor(propagate, carry))
            if bit != 31:  # discard overflow beyond the 32-bit word
                av, bv = self.values[ai], self.values[bi]
                pv, cv = self.values[propagate], self.values[carry]
                carry = self.push(av * bv + pv * cv,
                                  ((ai, bv), (bi, av), (propagate, cv), (carry, pv)))
        return result

    def add(self, *words):
        total = words[0]
        for word in words[1:]:
            total = self.add2(total, word)
        return total

    def choose(self, a, b, c):
        result = []
        for ai, bi, ci in zip(a, b, c):
            av, bv, cv = self.values[ai], self.values[bi], self.values[ci]
            result.append(self.push(av * bv + (1 - av) * cv,
                                    ((ai, bv - cv), (bi, av), (ci, 1 - av))))
        return result

    def majority(self, a, b, c):
        result = []
        for ai, bi, ci in zip(a, b, c):
            av, bv, cv = self.values[ai], self.values[bi], self.values[ci]
            value = av * bv + av * cv + bv * cv - 2 * av * bv * cv
            result.append(self.push(value, (
                (ai, bv + cv - 2 * bv * cv),
                (bi, av + cv - 2 * av * cv),
                (ci, av + bv - 2 * av * bv))))
        return result

    def word(self, value):
        return [self.one if (value >> bit) & 1 else self.zero for bit in range(32)]

    @staticmethod
    def rotr(word, n):
        return word[n:] + word[:n]

    def shr(self, word, n):
        return word[n:] + [self.zero] * n

    def big_sigma0(self, x):
        return self.xor3(self.rotr(x, 2), self.rotr(x, 13), self.rotr(x, 22))

    def big_sigma1(self, x):
        return self.xor3(self.rotr(x, 6), self.rotr(x, 11), self.rotr(x, 25))

    def small_sigma0(self, x):
        return self.xor3(self.rotr(x, 7), self.rotr(x, 18), self.shr(x, 3))

    def small_sigma1(self, x):
        return self.xor3(self.rotr(x, 17), self.rotr(x, 19), self.shr(x, 10))

    def backward(self, seeds):
        """Accumulate a vector-Jacobian product from (node, adjoint) pairs."""
        if not self.recording:
            raise RuntimeError("cannot backpropagate through a forward-only tape")
        gradients = [self.ctx.zero] * len(self.values)
        for index, value in seeds:
            gradients[index] += value
        for index in range(len(self.values) - 1, -1, -1):
            grad = gradients[index]
            if grad:
                for parent, derivative in self.parents[index]:
                    gradients[parent] += grad * derivative
        return gradients


def bytes_to_bits(data):
    """Big-endian words, least-significant bit first within each 32-bit word."""
    if len(data) % 4:
        raise ValueError("byte count must be a multiple of four")
    return [(int.from_bytes(data[i:i+4], "big") >> bit) & 1
            for i in range(0, len(data), 4) for bit in range(32)]


def pad_message(message):
    if len(message) > 55:
        raise ValueError("single-block messages must contain at most 55 bytes")
    return (message + b"\x80" + b"\x00" * (55 - len(message))
            + (len(message) * 8).to_bytes(8, "big"))


@dataclass
class Evaluation:
    loss: object
    output: list
    gradient: list | None
    nodes: int
    forward_seconds: float
    backward_seconds: float


class SHA256MP:
    """Differentiable SHA-256 surrogate; weights are fixed, inputs are variables."""

    def __init__(self, precision=128, weights=DEFAULT_WEIGHTS):
        if precision < 113:
            raise ValueError("precision must be at least 113 bits to load binary128 exactly")
        self.ctx = mp.clone()
        self.ctx.prec = precision
        self.network = XORNetwork(self.ctx, weights)

    def evaluate(self, block_bits, target_bits, *, backward=True, rounds=64):
        """Return MSE, 256 output values, and optionally dMSE/d(block_bits).

        Inputs and target are flat sequences of 512 and 256 values. Use strings
        or mpf values for fractional inputs to avoid prior float64 rounding.
        Reduced rounds are for diagnostics; the default includes all 64 rounds.
        """
        if len(block_bits) != 512 or len(target_bits) != 256:
            raise ValueError("expected 512 input values and 256 target values")
        if not 1 <= rounds <= 64:
            raise ValueError("rounds must be between 1 and 64")
        ctx = self.ctx
        x = [ctx.mpf(v) for v in block_bits]
        target = [ctx.mpf(v) for v in target_bits]
        if not all(ctx.isfinite(v) for v in x + target):
            raise ValueError("input and target values must be finite")
        started = time.perf_counter()
        tape = Tape(ctx, self.network, recording=backward)
        inputs = [tape.push(v) for v in x]
        schedule = [inputs[i:i+32] for i in range(0, 512, 32)]
        for t in range(16, rounds):
            schedule.append(tape.add(
                tape.small_sigma1(schedule[t-2]), schedule[t-7],
                tape.small_sigma0(schedule[t-15]), schedule[t-16]))
        initial = [tape.word(v) for v in IV]
        state = initial
        for t in range(rounds):
            a, b, c, d, e, f, g, h = state
            t1 = tape.add(h, tape.big_sigma1(e), tape.choose(e, f, g),
                          tape.word(K[t]), schedule[t])
            t2 = tape.add(tape.big_sigma0(a), tape.majority(a, b, c))
            state = [tape.add2(t1, t2), a, b, c, tape.add2(d, t1), e, f, g]
        outputs = [bit for word, iv in zip(state, initial)
                   for bit in tape.add2(word, iv)]
        values = [tape.values[i] for i in outputs]
        errors = [v - y for v, y in zip(values, target)]
        loss = ctx.fsum(error * error for error in errors) / 256
        forward_seconds = time.perf_counter() - started
        gradient = None
        backward_seconds = 0.0
        if backward:
            started = time.perf_counter()
            adjoints = tape.backward((i, error / 128) for i, error in zip(outputs, errors))
            gradient = [adjoints[i] for i in inputs]
            backward_seconds = time.perf_counter() - started
        return Evaluation(loss, values, gradient, len(tape.values),
                          forward_seconds, backward_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=128,
                        help="mpmath significand precision in binary bits (default: 128)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--message-hex", help="message up to 55 bytes; otherwise seeded random")
    parser.add_argument("--target-hex", help="32 target bytes; otherwise seeded random")
    parser.add_argument("--rounds", type=int, default=64, help="use fewer rounds only for debugging")
    parser.add_argument("--steps", type=int, default=1, help="number of forward/backward evaluations")
    parser.add_argument("--lr", default="1e-20", help="sign-SGD step size between evaluations")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "sha256_mpmath_backward.json")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 1 <= args.rounds <= 64:
        parser.error("--rounds must be between 1 and 64")
    rng = random.Random(args.seed)
    try:
        model = SHA256MP(args.precision, args.weights)
        ctx = model.ctx
        lr = ctx.mpf(args.lr)
        if not ctx.isfinite(lr) or lr <= 0:
            raise ValueError("--lr must be finite and positive")
        message = rng.randbytes(32) if args.message_hex is None else bytes.fromhex(args.message_hex)
        target_bytes = rng.randbytes(32) if args.target_hex is None else bytes.fromhex(args.target_hex)
        if len(target_bytes) != 32:
            raise ValueError("--target-hex must encode exactly 32 bytes")
        block = pad_message(message)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    x = [ctx.mpf(v) for v in bytes_to_bits(block)]
    target = bytes_to_bits(target_bytes)
    digits = ctx.dps + 5
    fmt = lambda v: ctx.nstr(v, digits)
    history = []
    print(f"precision_bits={ctx.prec} rounds={args.rounds} steps={args.steps}", flush=True)
    print(f"message={message.hex()} target={target_bytes.hex()}", flush=True)
    for step in range(1, args.steps + 1):
        print(f"step={step}: computing forward and backward...", flush=True)
        result = model.evaluate(x, target, rounds=args.rounds)
        grad_max = max(abs(v) for v in result.gradient)
        if not all(ctx.isfinite(v) for v in [result.loss] + result.output + result.gradient):
            raise ArithmeticError("non-finite forward/backward result")
        row = {
            "step": step, "loss": fmt(result.loss), "grad_max": fmt(grad_max),
            "nonzero_gradients": sum(v != 0 for v in result.gradient),
            "nodes": result.nodes, "forward_seconds": result.forward_seconds,
            "backward_seconds": result.backward_seconds,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        # Save the evaluated input and its matching gradient, including partial
        # progress on longer runs. Timing fields alone use Python floats.
        payload = {
            "model": "learned-XOR SHA-256 relaxation", "precision_bits": ctx.prec,
            "rounds": args.rounds, "seed": args.seed,
            "rng": "random.Random", "weights_file": str(args.weights),
            "message_hex": message.hex(), "padded_block_hex": block.hex(),
            "target_hex": target_bytes.hex(), "bit_order": "big-endian words, LSB-first per word",
            "optimizer": "fixed-lr projected sign-SGD", "learning_rate": fmt(lr),
            "history": history, "input": [fmt(v) for v in x],
            "output": [fmt(v) for v in result.output],
            "gradient": [fmt(v) for v in result.gradient],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if step < args.steps:
            x = [min(ctx.one, max(ctx.zero, v - lr * ctx.sign(g)))
                 for v, g in zip(x, result.gradient)]
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
