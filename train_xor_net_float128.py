#!/usr/bin/env python3
"""Train the tiny scaled-ELU XOR net with 128-bit mpmath backpropagation.

Exports 33 parameters as big-endian IEEE-754 binary128 values (16 bytes each),
plus a metadata JSON and a loss history CSV. No project-local Python imports.
"""
import argparse
import json
from pathlib import Path

from mpmath import mp


HERE = Path(__file__).resolve().parent
INITIAL_W1 = [
    ["0.353103310", "-1.574223638"],
    ["-2.798634052", "-2.563291311"],
    ["0.767613769", "-1.962423205"],
    ["2.440010786", "-3.430410147"],
    ["-4.616922379", "3.829748869"],
    ["1.821874022", "-2.935794115"],
    ["-3.508261681", "2.711204767"],
    ["2.786411762", "-3.034853697"],
]
INITIAL_B1 = ["0.177618653", "1.513626695", "-0.009045622", "-0.537469029",
              "-0.587478757", "-0.370821089", "-0.506456375", "-0.894154310"]
INITIAL_W2 = [["0.435043693", "-1.772849202", "0.773968756", "2.499436617",
               "2.217201471", "1.121317863", "2.691083908", "2.139189243"]]
INITIAL_B2 = ["-0.171298176"]
INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
TARGETS = (0, 1, 1, 0)


def elu(x):
    return x if x > 0 else mp.expm1(x)


def elu_grad(x):
    return mp.mpf(1) if x > 0 else mp.exp(x)


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + mp.exp(-x))
    ex = mp.exp(x)
    return ex / (1 + ex)


def forward(pair, params):
    w1, b1, w2, b2 = params
    pre = [w1[j][0] * pair[0] + w1[j][1] * pair[1] + b1[j]
           for j in range(8)]
    hidden = [elu(value) for value in pre]
    logit = sum((w2[0][j] * hidden[j] for j in range(8)), mp.mpf(0)) + b2[0]
    probability = sigmoid(logit)
    output = mp.mpf("1.0001") * probability - mp.mpf("0.00005")
    return output, (pre, hidden, probability)


def loss_and_gradient(params):
    w1, b1, w2, b2 = params
    scale = mp.mpf("1.0001")
    gw1 = [[mp.mpf(0), mp.mpf(0)] for _ in range(8)]
    gb1 = [mp.mpf(0) for _ in range(8)]
    gw2 = [[mp.mpf(0) for _ in range(8)]]
    gb2 = [mp.mpf(0)]
    loss = mp.mpf(0)

    for pair, target_int in zip(INPUTS, TARGETS):
        target = mp.mpf(target_int)
        output, (pre, hidden, probability) = forward(pair, params)
        error = output - target
        loss += error * error / 4

        # d/dlogit for mean squared error over the four XOR examples.
        dlogit = (error / 2) * scale * probability * (1 - probability)
        gb2[0] += dlogit
        for j in range(8):
            gw2[0][j] += dlogit * hidden[j]
            dpre = dlogit * w2[0][j] * elu_grad(pre[j])
            gb1[j] += dpre
            gw1[j][0] += dpre * pair[0]
            gw1[j][1] += dpre * pair[1]

    return loss, (gw1, gb1, gw2, gb2)


def flatten(params):
    w1, b1, w2, b2 = params
    return [v for row in w1 for v in row] + list(b1) + list(w2[0]) + list(b2)


def unflatten(values):
    offset = 0
    w1 = [values[offset + 2 * j:offset + 2 * j + 2] for j in range(8)]
    offset += 16
    b1 = values[offset:offset + 8]
    offset += 8
    w2 = [values[offset:offset + 8]]
    offset += 8
    b2 = values[offset:offset + 1]
    return w1, b1, w2, b2


def round_shift_even(value, shift):
    """Round a positive integer right by shift bits, ties to even."""
    if shift <= 0:
        return value << (-shift)
    quotient, remainder = divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        quotient += 1
    return quotient


def to_binary128(value):
    """Encode a finite mpf as one IEEE-754 binary128 big-endian value."""
    sign, mantissa, exponent, _ = value._mpf_
    sign_bit = int(sign) << 127
    if mantissa == 0:
        return sign_bit.to_bytes(16, "big")

    top_exponent = mantissa.bit_length() - 1 + exponent
    bias = 16383

    if top_exponent > 16383:
        raw = sign_bit | (0x7FFF << 112)
        return raw.to_bytes(16, "big")

    if top_exponent >= -16382:
        # Normalize to a 113-bit significand, rounding to nearest, ties even.
        significand = round_shift_even(mantissa, mantissa.bit_length() - 113)
        if significand >= (1 << 113):
            significand >>= 1
            top_exponent += 1
        if top_exponent > 16383:
            raw = sign_bit | (0x7FFF << 112)
            return raw.to_bytes(16, "big")
        exponent_field = top_exponent + bias
        fraction = significand - (1 << 112)
        raw = sign_bit | (exponent_field << 112) | fraction
        return raw.to_bytes(16, "big")

    # Subnormal values use units of 2**-16494.
    subnormal = round_shift_even(mantissa, -(exponent + 16494))
    if subnormal >= (1 << 112):
        raw = sign_bit | (1 << 112)  # rounded to the smallest normal
    else:
        raw = sign_bit | subnormal
    return raw.to_bytes(16, "big")


def from_binary128(raw):
    """Decode a finite exported parameter without going through float64."""
    if len(raw) != 16:
        raise ValueError("one binary128 parameter requires 16 bytes")
    word = int.from_bytes(raw, "big")
    exponent = (word >> 112) & 0x7FFF
    fraction = word & ((1 << 112) - 1)
    if exponent == 0x7FFF:
        raise ValueError("exported parameters must be finite")
    value = (mp.ldexp(mp.mpf(fraction), -16494) if exponent == 0 else
             mp.ldexp(mp.mpf((1 << 112) | fraction), exponent - 16495))
    return -value if word >> 127 else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--precision", type=int, default=128,
                        help="mpmath binary precision in bits")
    parser.add_argument("--lr", default="0.001")
    parser.add_argument("--eps", default="1e-30")
    parser.add_argument("--output", type=Path, default=HERE / "xor_net_f128.bin")
    parser.add_argument("--history", type=Path,
                        default=HERE / "results" / "xor_net_f128_training.csv")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.precision < 113:
        parser.error("--precision must be at least 113 bits")

    mp.prec = args.precision
    lr = mp.mpf(args.lr)
    eps = mp.mpf(args.eps)
    beta1, beta2 = mp.mpf("0.9"), mp.mpf("0.999")
    params = (
        [[mp.mpf(v) for v in row] for row in INITIAL_W1],
        [mp.mpf(v) for v in INITIAL_B1],
        [[mp.mpf(v) for v in INITIAL_W2[0]]],
        [mp.mpf(v) for v in INITIAL_B2],
    )
    values = flatten(params)
    first_moment = [mp.mpf(0) for _ in values]
    second_moment = [mp.mpf(0) for _ in values]
    best_loss = None
    best_values = None
    best_step = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.history.parent.mkdir(parents=True, exist_ok=True)

    with args.history.open("w", encoding="utf-8") as history:
        history.write("step,loss,grad_max\n")
        for step in range(1, args.steps + 1):
            params = unflatten(values)
            loss, gradients = loss_and_gradient(params)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_values = list(values)
                best_step = step - 1  # number of Adam updates already applied
            grad_values = flatten(gradients)
            first_correction = 1 - beta1 ** step
            second_correction = 1 - beta2 ** step
            for i, grad in enumerate(grad_values):
                first_moment[i] = beta1 * first_moment[i] + (1 - beta1) * grad
                second_moment[i] = beta2 * second_moment[i] + (1 - beta2) * grad * grad
                m_hat = first_moment[i] / first_correction
                v_hat = second_moment[i] / second_correction
                values[i] -= lr * m_hat / (mp.sqrt(v_hat) + eps)

            if step % 1000 == 0 or step == 1 or step == args.steps:
                grad_max = max(abs(g) for g in grad_values)
                line = f"{step},{mp.nstr(loss, 40)},{mp.nstr(grad_max, 12)}\n"
                history.write(line)
                history.flush()
                print(line.strip(), flush=True)

    # The loop evaluates before updating, so consider the last update too.
    last_loss, _ = loss_and_gradient(unflatten(values))
    if last_loss < best_loss:
        best_loss, best_values, best_step = last_loss, list(values), args.steps
    raw = b"".join(to_binary128(value) for value in best_values)
    if len(raw) != 33 * 16:
        raise RuntimeError(f"expected 528 parameter bytes, got {len(raw)}")
    args.output.write_bytes(raw)

    # Report the actual saved model separately from its training checkpoint:
    # binary128 retains 113 significand bits, less than the training precision.
    evaluation_precision = max(args.precision, 180)
    with mp.workprec(evaluation_precision):
        saved = args.output.read_bytes()
        params = unflatten([from_binary128(saved[i:i + 16])
                            for i in range(0, len(saved), 16)])
        outputs = [forward(pair, params)[0] for pair in INPUTS]
        exported_loss = mp.fsum((v - t) ** 2 for v, t in zip(outputs, TARGETS)) / 4

    metadata = {
        "format": "IEEE-754 binary128",
        "endianness": "big",
        "parameter_count": 33,
        "parameter_bytes": 16,
        "total_bytes": len(raw),
        "layout": ["W1[8][2] row-major", "B1[8]", "W2[1][8] row-major", "B2[1]"],
        "architecture": "Linear(2,8) -> ELU(alpha=1) -> Linear(8,1) -> sigmoid*1.0001-0.00005",
        "training_precision_bits": args.precision,
        "optimizer": "manual mpmath Adam",
        "steps": args.steps,
        "learning_rate": args.lr,
        "checkpoint": "best",
        "best_step": best_step,
        "best_step_unit": "completed Adam updates",
        "best_mse": mp.nstr(best_loss, 50),
        "evaluation_precision_bits": evaluation_precision,
        "final_mse": mp.nstr(exported_loss, 50),
        "xor_outputs_00_01_10_11": [mp.nstr(v, 50) for v in outputs],
        "weights_file": args.output.name,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"best checkpoint: {best_step} Adam updates, MSE: {mp.nstr(best_loss, 50)}")
    print(f"exported binary128 MSE: {mp.nstr(exported_loss, 50)}")
    print("XOR outputs:", ", ".join(mp.nstr(v, 30) for v in outputs))
    print(f"binary128 weights: {args.output} ({len(raw)} bytes)")
    print(f"metadata: {metadata_path}")
    print(f"history: {args.history}")


if __name__ == "__main__":
    main()
