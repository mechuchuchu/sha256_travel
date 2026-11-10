#!/usr/bin/env python3
"""Self-contained adaptive sign-SGD experiment for a differentiable SHA-256 model.

The only non-standard Python dependencies are PyTorch and NumPy. This file
contains the relaxed SHA-256 bit model and the hard-coded scaled-ELU XOR
network; it does not import any project-local Python modules.
"""
import argparse
import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


K = [
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
]
IV = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

# Scaled-ELU XOR MLP: Linear(2, 8) -> ELU -> Linear(8, 1) ->
# sigmoid * 1.0001 - 0.00005. The trained parameters are hard-coded below as
# 33 IEEE-754 binary128 values. PyTorch evaluates the SHA surrogate in float64,
# so this loader rounds the stored parameters to float64 when creating tensors.
W1 = []
B1 = []
W2 = []
B2 = []
_PARAM_CACHE = {}
_XOR_NET_F128_HEX = "3ffe7d8f70ebe01ad38172b874d510e5c000067425f0692c4dedf6e94c712a9ec0008f4c76e8229ae521a35459012481c0007583d362a623fad00f36e66b262d3fff1ac335e237a432448c01c391c776c00030e11cf5fd02ec8be8f4c23bc70740005e0e328ca2dea0d22c7b9d3807a9c000e82642ffdfc48dddb8c0be5384fec00142af9e612b9078f754443a6c5edc400109ad06d0f590e0e2a45c3cc076fa4000124494f7ac9da13d583dd32947bfc000aae61c66c22aa38f9e90f616d081c000f616eb8f6a882a91ec9bee5f347440008492816476df33ca3b2fcaa4dbae400088ea3acee02b4ade1888ddc4f7aac000b34f18931fff67dcf6412d9d162bbffc77e166abdb9f1e1469ecdcf960f93fffbd332bf51fa7be9ad6a1316eb87ebffd208ac08b965c6e4231e04a962d48bffe40aec3acd84fe2665b67eccd8c9cbffdababaf6a121f05966e97debc62b8bffe11de5256f26dbb86aa8b8dd75057bffda006eed415e4c4cf063dcfbdd184bfff2680a74bbc31950f0421f0a2ef903ffec328b9bd13b722179a8979188e4dbfffead13ffcd9dacc969a20f376013e3fff2df0d08d8017f4f86f564ed8ca8240006d82cc08d4cec3df23a8b77d048c40004c0c871d55089363599eb7bc3c873fff7cce8fe0f91fb39a2715bb601d60400089c5255c0fafeaabc108520d717040003ed611d6e8f23e06b667bbe8833bbffe24819410e0439eb55cb0730a7a23"


def _binary128_to_float(raw):
    word = int.from_bytes(raw, "big")
    sign = word >> 127
    exponent = (word >> 112) & 0x7FFF
    fraction = word & ((1 << 112) - 1)
    if exponent == 0:
        value = math.ldexp(float(fraction), -16382 - 112)
    elif exponent == 0x7FFF:
        value = math.inf if fraction == 0 else math.nan
    else:
        significand = (1 << 112) | fraction
        value = math.ldexp(float(significand), exponent - 16383 - 112)
    return -value if sign else value


def _load_hardcoded_binary128_xor():
    global W1, B1, W2, B2
    if _XOR_NET_F128_HEX == "__BINARY128_HEX__":
        raise RuntimeError("embedded binary128 XOR weights are missing")
    raw = bytes.fromhex(_XOR_NET_F128_HEX)
    if len(raw) != 33 * 16:
        raise RuntimeError(f"expected 528 embedded XOR bytes, got {len(raw)}")
    values = [_binary128_to_float(raw[i:i + 16]) for i in range(0, len(raw), 16)]
    W1 = [values[2 * j:2 * j + 2] for j in range(8)]
    B1 = values[16:24]
    W2 = [values[24:32]]
    B2 = [values[32]]


_load_hardcoded_binary128_xor()


def _params(like):
    key = (like.device, like.dtype)
    if key not in _PARAM_CACHE:
        _PARAM_CACHE[key] = (
            torch.tensor(W1, dtype=like.dtype, device=like.device),
            torch.tensor(B1, dtype=like.dtype, device=like.device),
            torch.tensor(W2, dtype=like.dtype, device=like.device),
            torch.tensor(B2, dtype=like.dtype, device=like.device),
        )
    return _PARAM_CACHE[key]


def learned_xor(a, b):
    a, b = torch.broadcast_tensors(a, b)
    pair = torch.stack((a, b), dim=-1)
    w1, b1, w2, b2 = _params(a)
    hidden = F.elu(F.linear(pair, w1, b1), alpha=1.0)
    probability = torch.sigmoid(F.linear(hidden, w2, b2))
    return (probability * 1.0001 - 0.00005).squeeze(-1)


def learned_add2(a, b):
    a, b = torch.broadcast_tensors(a, b)
    propagate = learned_xor(a, b)
    carry = torch.zeros_like(propagate[..., 0])
    carries = []
    for bit in range(32):
        carries.append(carry)
        carry = a[..., bit] * b[..., bit] + propagate[..., bit] * carry
    return learned_xor(propagate, torch.stack(carries, dim=-1))


def word_bits(value, device=None, dtype=torch.float32):
    values = torch.as_tensor(value, dtype=torch.int64, device=device)
    shifts = torch.arange(32, device=values.device, dtype=torch.int64)
    return ((values.unsqueeze(-1) >> shifts) & 1).to(dtype)


def xor(a, b):
    return learned_xor(a, b)


def xor3(a, b, c):
    return xor(xor(a, b), c)


def rotr(x, n):
    return torch.roll(x, shifts=-n, dims=-1)


def shr(x, n):
    return torch.cat((x[..., n:], torch.zeros_like(x[..., :n])), dim=-1)


def add2(x, y):
    propagate = learned_xor(x, y)
    carry = torch.zeros_like(propagate[..., 0])
    carries = []
    for bit in range(32):
        carries.append(carry)
        carry = x[..., bit] * y[..., bit] + propagate[..., bit] * carry
    return learned_xor(propagate, torch.stack(carries, dim=-1))


def add(*words):
    total = words[0]
    for word in words[1:]:
        total = add2(total, word)
    return total


def big_sigma0(x):
    return xor3(rotr(x, 2), rotr(x, 13), rotr(x, 22))


def big_sigma1(x):
    return xor3(rotr(x, 6), rotr(x, 11), rotr(x, 25))


def small_sigma0(x):
    return xor3(rotr(x, 7), rotr(x, 18), shr(x, 3))


def small_sigma1(x):
    return xor3(rotr(x, 17), rotr(x, 19), shr(x, 10))


def choose(x, y, z):
    return x * y + (1 - x) * z


def majority(x, y, z):
    return x * y + x * z + y * z - 2 * x * y * z


class SHA256Bits(nn.Module):
    """64-round relaxed SHA-256 compression for one padded 512-bit block."""

    def forward(self, block_bits):
        if block_bits.ndim == 2:
            block_bits = block_bits.reshape(-1, 16, 32)
        if block_bits.shape[1:] != (16, 32):
            raise ValueError("expected shape (batch, 512) or (batch, 16, 32)")

        schedule = [block_bits[:, i, :] for i in range(16)]
        for t in range(16, 64):
            schedule.append(add(
                small_sigma1(schedule[t - 2]), schedule[t - 7],
                small_sigma0(schedule[t - 15]), schedule[t - 16],
            ))

        state = [word_bits(v, device=block_bits.device, dtype=block_bits.dtype)
                 .expand(block_bits.shape[0], -1) for v in IV]
        for t in range(64):
            a, b, c, d, e, f, g, h = state
            t1 = add(h, big_sigma1(e), choose(e, f, g),
                     word_bits(K[t], device=block_bits.device, dtype=block_bits.dtype),
                     schedule[t])
            t2 = add(big_sigma0(a), majority(a, b, c))
            state = [add2(t1, t2), a, b, c, add2(d, t1), e, f, g]

        return torch.stack([
            add2(x, word_bits(IV[i], device=block_bits.device,
                              dtype=block_bits.dtype))
            for i, x in enumerate(state)
        ], dim=1)


def block_to_bits(block):
    if len(block) != 64:
        raise ValueError("SHA-256 block must be exactly 64 bytes")
    words = np.frombuffer(block, dtype=">u4").astype(np.uint32)
    return word_bits(words)


def candidate_lrs():
    values = [1e-20, 3e-20, 1e-19, 3e-19, 1e-18, 3e-18,
              1e-17, 3e-17, 1e-16, 3e-16]
    for exponent in range(-15, -1):
        values.extend((10.0 ** exponent, 3.0 * 10.0 ** exponent))
    values.append(1e-1)
    return values


def run(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cpu":
        torch.set_num_threads(1)

    rng = np.random.default_rng(args.seed)
    message = rng.bytes(32)
    padded = message + b"\x80" + b"\x00" * (55 - len(message))
    padded += (len(message) * 8).to_bytes(8, "big")
    x = block_to_bits(padded).unsqueeze(0).to(device=device, dtype=torch.float64)
    target_words = np.frombuffer(rng.bytes(32), dtype=">u4").astype(np.uint32)
    target = word_bits(target_words).reshape(1, 8, 32).to(
        device=device, dtype=torch.float64)
    lrs = candidate_lrs()
    lr_tensor = torch.tensor(lrs, dtype=torch.float64, device=device).reshape(-1, 1, 1)
    original = x.clone()
    model = SHA256Bits()

    with open(args.log, "w", encoding="utf-8") as log:
        log.write(f"optimizer=per-step sign-SGD steps={args.steps} device={device}\n")
        log.write(f"message={message.hex()}\ntarget={target_words.tobytes().hex()}\n")
        log.write(f"learning_rate_candidates={len(lrs)} min_lr={min(lrs):.1e} max_lr={max(lrs):.1e}\n")
        log.write("step,base_loss,chosen_lr,post_loss,loss_change,step_delta,total_delta,boundary_bits,grad_max\n")
        step = 1
        seen_states = {}
        rows = []
        while step <= args.steps:
            state_bytes = x.detach().cpu().numpy().tobytes()
            state_key = hashlib.sha256(state_bytes).digest()
            if state_key in seen_states:
                cycle_start = seen_states[state_key]
                cycle_rows = rows[cycle_start - 1:step - 1]
                cycle_len = len(cycle_rows)
                if cycle_len:
                    log.write(
                        f"cycle_detected start_step={cycle_start} "
                        f"repeat_at_step={step} period={cycle_len}; "
                        "repeating exact rows\n"
                    )
                    while step <= args.steps:
                        row = cycle_rows[(step - cycle_start) % cycle_len]
                        repeated = f"{step},{row.rstrip().split(',', 1)[1]}\n"
                        log.write(repeated)
                        rows.append(repeated)
                        step += 1
                    log.flush()
                    break
            else:
                seen_states[state_key] = step

            x = x.detach().requires_grad_(True)
            base_loss = F.mse_loss(model(x), target)
            base_loss.backward()
            grad_max = x.grad.abs().max().item()
            direction = x.grad.detach().sign()[0]

            candidates = x.detach()[0].unsqueeze(0) - lr_tensor * direction.unsqueeze(0)
            candidates.clamp_(0.0, 1.0)
            with torch.no_grad():
                outputs = model(candidates)
                losses = ((outputs - target) ** 2).mean(dim=(1, 2))
                best = int(torch.argmin(losses))
                post_loss = losses[best].item()
                chosen = candidates[best].unsqueeze(0).detach()
                step_delta = (chosen - x.detach()).abs().max().item()
                total_delta = (chosen - original).abs().max().item()
                boundary_bits = int(((chosen == 0.0) | (chosen == 1.0)).sum())

            line = (
                f"{step},{base_loss.item():.17e},{lrs[best]:.1e},"
                f"{post_loss:.17e},{post_loss-base_loss.item():.17e},"
                f"{step_delta:.8e},{total_delta:.8e},{boundary_bits},"
                f"{grad_max:.8e}\n"
            )
            log.write(line)
            log.flush()
            rows.append(line)
            if step % 10 == 0 or step == args.steps:
                print(line.strip(), flush=True)
            x = chosen
            step += 1

    print(f"log saved: {args.log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log", default="sha256_xor_elu_scaled_sgd_adaptive.log")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
