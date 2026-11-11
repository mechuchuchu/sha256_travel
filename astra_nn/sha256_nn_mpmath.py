"""mpmath forward/backward for the fixed affine-ReLU SHA-256 circuit.

External vectors are MSB first, as in FractionalSHA256NN. Precision defaults
to 128 significand bits; this is not IEEE binary128 emulation (113 significand
bits, restricted exponents). ReLU's derivative at zero is zero, like PyTorch.
Only mpmath is required at runtime; weights/constants come from the JSON export.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

from mpmath import mp


DEFAULT_WEIGHTS = Path(__file__).with_name("sha256_weights.json")


def bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> bit) & 1 for byte in data for bit in range(7, -1, -1)]


def bits_to_bytes(bits) -> bytes:
    bits = list(bits)
    if len(bits) % 8 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("expected exact binary values with a multiple of 8 bits")
    return bytes(sum(int(bits[i+j]) << (7-j) for j in range(8))
                 for i in range(0, len(bits), 8))


class _Gate:
    """Fixed integer affine-ReLU-affine weights and their local Jacobian."""

    def __init__(self, ctx, state, name):
        def integer(value):
            value = ctx.mpf(value)
            if not ctx.isfinite(value) or value != int(value):
                raise ValueError("this circuit requires integer gate weights")
            return int(value)

        self.w_in = [[integer(v) for v in row] for row in state[name + ".weight_in"]]
        self.b_in = [integer(v) for v in state[name + ".bias_in"]]
        self.w_out = [[integer(v) for v in row] for row in state[name + ".weight_out"]]
        self.b_out = [integer(v) for v in state[name + ".bias_out"]]

    def __call__(self, tape, *inputs):
        ctx = tape.ctx
        hidden, active = [], []
        for row, bias in zip(self.w_in, self.b_in):
            value = ctx.mpf(bias)
            for node, weight in zip(inputs, row):
                if weight:
                    value += tape.values[node] * weight
            active.append(value > 0)
            hidden.append(value if value > 0 else ctx.zero)
        outputs = []
        for row, bias in zip(self.w_out, self.b_out):
            value = ctx.mpf(bias)
            for h, weight in zip(hidden, row):
                if weight:
                    value += h * weight
            parents = ()
            if tape.recording:
                # Local derivatives are constant within a ReLU region. They are
                # integers for the original weights but may be mpf when perturbed.
                # Keep repeated parent IDs: their adjoints must be accumulated.
                derivatives = [sum(wo * wi[i] for wo, wi, on in
                                   zip(row, self.w_in, active) if on)
                               for i in range(len(inputs))]
                parents = tuple((node, d) for node, d in zip(inputs, derivatives) if d)
            outputs.append(tape.push(value, parents))
        return outputs


class _Tape:
    def __init__(self, ctx, recording=True):
        self.ctx, self.recording = ctx, recording
        self.values, self.parents = [], []
        self.zero = self.push(ctx.zero)
        self.one = self.push(ctx.one)

    def push(self, value, parents=()):
        index = len(self.values)
        self.values.append(value)
        if self.recording:
            self.parents.append(parents)
        return index

    def backward(self, seeds):
        if not self.recording:
            raise RuntimeError("backward needs a recorded forward pass")
        adjoints = [self.ctx.zero] * len(self.values)
        for node, value in seeds:
            adjoints[node] += value
        for node in range(len(self.values) - 1, -1, -1):
            grad = adjoints[node]
            if grad:
                for parent, derivative in self.parents[node]:
                    adjoints[parent] += grad * derivative
        return adjoints


@dataclass
class Evaluation:
    output: list
    loss: object
    gradient: list
    nodes: int
    forward_seconds: float
    backward_seconds: float


class FractionalSHA256MP:
    """Single 512-value input -> 256 values, all computed in an isolated context.

    forward(inputs) returns output values. backward(inputs, grad_output)
    recomputes a recorded forward pass and returns J(inputs).T @ grad_output.
    evaluate(inputs, target) computes both MSE and its input gradient in one pass.
    Pass decimal strings or this model's ctx.mpf values to retain input precision.
    """

    def __init__(self, mode="message", precision=128, weights=DEFAULT_WEIGHTS):
        if mode not in ("message", "compression"):
            raise ValueError("mode must be 'message' or 'compression'")
        if not isinstance(precision, int) or precision < 2:
            raise ValueError("precision must be an integer >= 2")
        self.mode = mode
        self.ctx = mp.clone()
        self.ctx.prec = precision
        # Decimal JSON numbers go straight to mpmath, never through float64.
        metadata = json.loads(Path(weights).read_text(), parse_float=str)
        if metadata.get("format_version") != 1 or metadata.get("bit_order") != "msb_first":
            raise ValueError("unsupported weights format")
        state = metadata["state_dict"]
        self.gates = {name: _Gate(self.ctx, state, name)
                      for name in ("xor_and", "prefix", "xor3", "choose", "majority")}
        self.constants = {}
        for name, size in (("initial_state", 8), ("round_constants", 64), ("padding_block", 16)):
            rows = [[self.ctx.mpf(v) for v in row] for row in state[name]]
            if len(rows) != size or any(len(row) != 32 or any(v not in (0, 1) for v in row)
                                       for row in rows):
                raise ValueError(f"invalid {name} buffer")
            self.constants[name] = rows

    def _vector(self, values, length, name, unit_interval=True):
        try:
            result = [self.ctx.mpf(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a flat real vector") from exc
        if len(result) != length or any(not self.ctx.isfinite(v) for v in result):
            raise ValueError(f"{name} must contain {length} finite real values")
        if unit_interval and any(v < 0 or v > 1 for v in result):
            raise ValueError(f"{name} must be in [0, 1]")
        return result

    def _add_words(self, tape, x, y):
        pg = [self.gates["xor_and"](tape, a, b) for a, b in zip(x, y)]
        propagate, generate = [v[0] for v in pg], [v[1] for v in pg]
        original = propagate
        for distance in (1, 2, 4, 8, 16):
            combined = [self.gates["prefix"](tape, generate[i], propagate[i],
                                             generate[i-distance], propagate[i-distance])
                        for i in range(distance, 32)]
            generate = generate[:distance] + [v[0] for v in combined]
            propagate = propagate[:distance] + [v[1] for v in combined]
        carry = [tape.zero] + generate[:-1]
        return [self.gates["xor_and"](tape, p, c)[0] for p, c in zip(original, carry)]

    def _add_many(self, tape, *words):
        total = words[0]
        for word in words[1:]:
            total = self._add_words(tape, total, word)
        return total

    def _sigma(self, tape, word, rotations, shift=False):
        a, b, c = rotations
        third = word[c:] + ([tape.zero] * c if shift else word[:c])
        return [self.gates["xor3"](tape, x, y, z)[0]
                for x, y, z in zip(word[a:]+word[:a], word[b:]+word[:b], third)]

    def _compress(self, tape, block, state, constants):
        schedule = list(block)
        for t in range(16, 64):
            schedule.append(self._add_many(
                tape, schedule[t-16], self._sigma(tape, schedule[t-15], (7, 18, 3), True),
                schedule[t-7], self._sigma(tape, schedule[t-2], (17, 19, 10), True)))
        a, b, c, d, e, f, g, h = state
        for t in range(64):
            choose = [self.gates["choose"](tape, x, y, z)[0] for x, y, z in zip(e, f, g)]
            majority = [self.gates["majority"](tape, x, y, z)[0] for x, y, z in zip(a, b, c)]
            t1 = self._add_many(tape, h, self._sigma(tape, e, (6, 11, 25)),
                                choose, constants[t], schedule[t])
            t2 = self._add_words(tape, self._sigma(tape, a, (2, 13, 22)), majority)
            a, b, c, d, e, f, g, h = (self._add_words(tape, t1, t2), a, b, c,
                                      self._add_words(tape, d, t1), e, f, g)
        return [self._add_words(tape, old, new)
                for old, new in zip(state, (a, b, c, d, e, f, g, h))]

    def _build(self, inputs, recording):
        inputs = self._vector(inputs, 512, "inputs")
        tape = _Tape(self.ctx, recording)
        input_nodes = [tape.push(v) for v in inputs]
        buffers = {name: [[tape.one if v else tape.zero for v in row] for row in rows]
                   for name, rows in self.constants.items()}
        block = [input_nodes[i:i+32][::-1] for i in range(0, 512, 32)]
        state = self._compress(tape, block, buffers["initial_state"], buffers["round_constants"])
        if self.mode == "message":
            state = self._compress(tape, buffers["padding_block"], state, buffers["round_constants"])
        output_nodes = [node for word in state for node in reversed(word)]
        return tape, input_nodes, output_nodes

    def forward(self, inputs):
        tape, _, outputs = self._build(inputs, recording=False)
        return [tape.values[i] for i in outputs]

    def backward(self, inputs, grad_output):
        seeds = self._vector(grad_output, 256, "grad_output", unit_interval=False)
        tape, input_nodes, output_nodes = self._build(inputs, recording=True)
        adjoints = tape.backward(zip(output_nodes, seeds))
        return [adjoints[i] for i in input_nodes]

    def evaluate(self, inputs, target):
        target = self._vector(target, 256, "target")
        start = time.perf_counter()
        tape, input_nodes, output_nodes = self._build(inputs, recording=True)
        output = [tape.values[i] for i in output_nodes]
        residual = [y - t for y, t in zip(output, target)]
        loss = self.ctx.fsum(r*r for r in residual) / 256
        forward_seconds = time.perf_counter() - start
        start = time.perf_counter()
        adjoints = tape.backward((node, r / 128) for node, r in zip(output_nodes, residual))
        return Evaluation(output, loss, [adjoints[i] for i in input_nodes], len(tape.values),
                          forward_seconds, time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=128, help="binary significand bits")
    parser.add_argument("--mode", choices=("message", "compression"), default="message")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=20261111)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-json", type=Path, help="512 values or an object with an input field")
    source.add_argument("--message-hex", help="64 raw bytes in message mode; 0..55 bytes in compression mode")
    parser.add_argument("--target-hex", default="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    parser.add_argument("--output", type=Path, default=Path("fractional_mpmath_result.json"))
    args = parser.parse_args()
    try:
        model = FractionalSHA256MP(args.mode, args.precision, args.weights)
        ctx = model.ctx
        target_bytes = bytes.fromhex(args.target_hex)
        if len(target_bytes) != 32:
            raise ValueError("target must encode exactly 32 bytes")
        if args.input_json is not None:
            data = json.loads(args.input_json.read_text(), parse_float=str)
            inputs = data["input"] if isinstance(data, dict) else data
        elif args.message_hex is not None:
            message = bytes.fromhex(args.message_hex)
            if args.mode == "compression":
                if len(message) > 55:
                    raise ValueError("compression-mode message must be at most 55 bytes")
                message += b"\x80" + bytes(55-len(message)) + (8*len(message)).to_bytes(8, "big")
            elif len(message) != 64:
                raise ValueError("message mode requires exactly 64 bytes")
            inputs = bytes_to_bits(message)
        else:
            rng = random.Random(args.seed)
            inputs = [ctx.ldexp(ctx.mpf(rng.getrandbits(args.precision)), -args.precision)
                      for _ in range(512)]
        inputs = model._vector(inputs, 512, "inputs")
        print(f"Evaluating {args.mode}, {args.precision} significand bits...", flush=True)
        result = model.evaluate(inputs, bytes_to_bits(target_bytes))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    # Extra decimal digits preserve each stored binary mpf on re-import.
    digits = (args.precision * 30103 + 99999) // 100000 + 3
    def encode(value):
        return ctx.nstr(value, digits)
    payload = {
        "model": "FractionalSHA256MP", "mode": args.mode,
        "precision_bits": args.precision, "precision_kind": "mpmath significand bits",
        "bit_order": "msb_first", "relu_derivative_at_zero": 0,
        "weights_file": str(args.weights.resolve()), "seed": args.seed,
        "input_source": str(args.input_json) if args.input_json else
                        ("message_hex" if args.message_hex is not None else "random dyadic"),
        "target_hex": target_bytes.hex(), "loss": encode(result.loss),
        "gradient_max_abs": encode(max(abs(g) for g in result.gradient)),
        "finite_gradients": sum(ctx.isfinite(g) for g in result.gradient),
        "nonzero_gradients": sum(g != 0 for g in result.gradient),
        "nodes": result.nodes, "forward_seconds": result.forward_seconds,
        "backward_seconds": result.backward_seconds,
        "input": list(map(encode, inputs)), "output": list(map(encode, result.output)),
        "gradient": list(map(encode, result.gradient)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"MSE: {payload['loss']}")
    print(f"Finite gradients: {payload['finite_gradients']}/512; max |gradient|: {payload['gradient_max_abs']}")
    print(f"Forward: {result.forward_seconds:.3f}s; backward: {result.backward_seconds:.3f}s")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
