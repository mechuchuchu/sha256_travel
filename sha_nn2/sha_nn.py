"""SHA-256 as a hand-wired ReLU network (no training).

Input : 512 message bits (fixed length)  ->  Output: 256 hash bits.
Every unit is  y = ReLU(w . x + b).  Weights are set by construction.

Two wiring styles:
  bool : only 0/1 values flow; additions are ripple-carry adders made of neurons
  int  : 32-bit words are summed as plain numbers (linear, free), then bits are
         peeled back off with ReLU pairs. Needs float64 (values up to ~2^35).
"""
import sys, os, time, hashlib
from collections import defaultdict
import torch


# ---------- constants (derived, not typed in) ----------
def _primes(n):
    ps, x = [], 2
    while len(ps) < n:
        if all(x % p for p in ps):
            ps.append(x)
        x += 1
    return ps


def _iroot(n, k):
    lo, hi = 0, 1 << (n.bit_length() // k + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** k <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


PR = _primes(64)
IV = [_iroot(p << 64, 2) & 0xFFFFFFFF for p in PR[:8]]
KC = [_iroot(p << 96, 3) & 0xFFFFFFFF for p in PR]
M32 = 0xFFFFFFFF


# ---------- linear expressions over wires ----------
class E:
    __slots__ = ("t", "c")

    def __init__(self, t=None, c=0):
        self.t = t or {}
        self.c = c


ZERO = E()


def lin(pairs, const=0):
    t, c = {}, const
    for w, e in pairs:
        c += w * e.c
        for k, v in e.t.items():
            nv = t.get(k, 0) + w * v
            if nv == 0:
                t.pop(k, None)
            else:
                t[k] = nv
    return E(t, c)


def bitlin(pairs, const=0):
    """Linear combo of 0/1-valued exprs, plus its true value range."""
    e = lin(pairs, const)
    lo = hi = const
    for w, b in pairs:
        if b.t:
            lo += min(0, w)
            hi += max(0, w)
        else:
            lo += w * b.c
            hi += w * b.c
    return e, lo, hi


class Net:
    def __init__(self, n_in, fan=6):
        self.n_in, self.rows, self.depth, self.fan = n_in, [], [0] * n_in, fan

    def neuron(self, e):
        idx = self.n_in + len(self.rows)
        self.rows.append((e.t, e.c))
        self.depth.append(1 + max(self.depth[k] for k in e.t))
        return E({idx: 1})

    def relu(self, e, lo, hi):
        if hi <= 0:
            return ZERO
        if lo >= 0:
            return e  # ReLU is the identity here
        return self.neuron(e)

    def mat(self, e):
        """Force a non-negative expr onto a single wire."""
        if not e.t or (len(e.t) == 1 and e.c == 0 and next(iter(e.t.values())) == 1):
            return e
        return self.neuron(e)

    def fit(self, e):
        return self.mat(e) if len(e.t) > self.fan else e


# ---------- bit-level gadgets ----------
def bitsum(net, bits):
    """<=3 bits -> (sum bit, carry bit).  carry = [s>=2] = ReLU(s-1) - ReLU(s-2)."""
    e, lo, hi = bitlin([(1, b) for b in bits])
    h1 = net.relu(lin([(1, e)], -1), lo - 1, hi - 1)
    h2 = net.relu(lin([(1, e)], -2), lo - 2, hi - 2)
    carry = lin([(1, h1), (-1, h2)])
    return lin([(1, e), (-2, carry)]), carry


def ch_bit(net, e, f, g):
    x, lo, hi = bitlin([(1, e), (1, f)], -1)  # e AND f
    y, lo2, hi2 = bitlin([(1, g), (-1, e)])  # (NOT e) AND g
    return lin([(1, net.relu(x, lo, hi)), (1, net.relu(y, lo2, hi2))])


def rotr(w, n):
    return [w[(i + n) % 32] for i in range(32)]


def shr(w, n):
    return [w[i + n] if i + n < 32 else ZERO for i in range(32)]


def xor3w(net, x, y, z):
    return [bitsum(net, [x[i], y[i], z[i]])[0] for i in range(32)]


def cword(v):
    return [E(c=(v >> i) & 1) for i in range(32)]


def is_const(w):
    return all(not b.t for b in w)


def cval(w):
    return sum(b.c << i for i, b in enumerate(w))


# ---------- 32-bit addition mod 2^32 ----------
def add_bool(net, words, final=True):
    consts = [w for w in words if is_const(w)]
    rest = [w for w in words if not is_const(w)]
    k = sum(cval(w) for w in consts) & M32
    if not rest:
        return cword(k)
    if consts and k:
        rest.append(cword(k))
    acc = rest[0]
    for w in rest[1:]:
        out, c = [], ZERO
        for i in range(32):
            s, c = bitsum(net, [acc[i], w[i], c])
            out.append(net.fit(s))
        acc = out
    return [net.mat(b) for b in acc] if final else acc


def add_int(net, words):
    V = lin([(1 << i, w[i]) for w in words for i in range(32)])
    if not V.t:
        return cword(V.c & M32)
    maxval = sum(cval(w) if is_const(w) else M32 for w in words)
    r = net.mat(V)
    bits = [ZERO] * 32
    for k in range(maxval.bit_length() - 1, 0, -1):
        t1 = net.neuron(lin([(1, r)], 1 - (1 << k)))
        t2 = net.neuron(lin([(1, r)], -(1 << k)))
        bit = net.neuron(lin([(1, t1), (-1, t2)]))  # bit k = [r >= 2^k]
        if k < 32:
            bits[k] = bit
        # subtract via the 0/1 bit (2^k*t1 - 2^k*t2 would overflow float64's 2^53)
        r = net.neuron(lin([(1, r), (-(1 << k), bit)]))  # r mod 2^k
    bits[0] = r
    return bits


# ---------- SHA-256 ----------
def schedule(net, W, mode):
    W = list(W)
    for t in range(16, 64):
        s0 = xor3w(net, rotr(W[t - 15], 7), rotr(W[t - 15], 18), shr(W[t - 15], 3))
        s1 = xor3w(net, rotr(W[t - 2], 17), rotr(W[t - 2], 19), shr(W[t - 2], 10))
        ops = [s1, W[t - 7], s0, W[t - 16]]
        W.append(add_int(net, ops) if mode == "int" else add_bool(net, ops))
    return W


def compress(net, state, W, mode):
    a, b, c, d, e, f, g, h = state
    for t in range(64):
        S1 = xor3w(net, rotr(e, 6), rotr(e, 11), rotr(e, 25))
        ch = [ch_bit(net, e[i], f[i], g[i]) for i in range(32)]
        S0 = xor3w(net, rotr(a, 2), rotr(a, 13), rotr(a, 22))
        mj = [bitsum(net, [a[i], b[i], c[i]])[1] for i in range(32)]
        t1 = [h, S1, ch, cword(KC[t]), W[t]]
        if mode == "int":
            ne = add_int(net, [d] + t1)
            na = add_int(net, t1 + [S0, mj])
        else:
            T1 = add_bool(net, t1, final=False)
            ne = add_bool(net, [d, T1])
            na = add_bool(net, [T1, S0, mj])
        a, b, c, d, e, f, g, h = na, a, b, c, ne, e, f, g
    add = add_int if mode == "int" else add_bool
    return [add(net, [s, v]) for s, v in zip(state, [a, b, c, d, e, f, g, h])]


def build(mode, fan=6):
    net = Net(512, fan)
    inp = [E({i: 1}) for i in range(512)]
    # block 1: the message itself (big-endian words; our words are LSB-first)
    W1 = [[inp[32 * t + 31 - i] for i in range(32)] for t in range(16)]
    # block 2: padding for a 512-bit message -> all constants
    pad = b"\x80" + b"\x00" * 55 + (512).to_bytes(8, "big")
    W2 = [cword(int.from_bytes(pad[4 * t:4 * t + 4], "big")) for t in range(16)]
    st = [cword(v) for v in IV]
    st = compress(net, st, schedule(net, W1, mode), mode)
    st = compress(net, st, schedule(net, W2, mode), mode)
    out = [st[j][31 - i] for j in range(8) for i in range(32)]  # MSB-first
    return net, out


class ShaNet(torch.nn.Module):
    def __init__(self, net, out, dtype):
        super().__init__()
        self.N, self.n_in, self.dtype = net.n_in + len(net.rows), net.n_in, dtype
        by = defaultdict(list)
        for j in range(len(net.rows)):
            by[net.depth[net.n_in + j]].append(j)
        self.layers = []
        for d in sorted(by):
            r, c, v, bias = [], [], [], []
            for ri, j in enumerate(by[d]):
                t, b = net.rows[j]
                bias.append(float(b))
                for k, w in t.items():
                    r.append(ri); c.append(k); v.append(float(w))
            Wm = torch.sparse_coo_tensor([r, c], torch.tensor(v, dtype=dtype),
                                         (len(by[d]), self.N)).coalesce()
            self.layers.append((torch.tensor([net.n_in + j for j in by[d]]), Wm,
                                torch.tensor(bias, dtype=dtype).unsqueeze(1)))
        self.out = out

    @torch.no_grad()
    def forward(self, bits):  # bits: (B, 512) of 0/1
        X = torch.zeros(self.N, bits.shape[0], dtype=self.dtype)
        X[: self.n_in] = bits.T.to(self.dtype)
        for idx, Wm, b in self.layers:  # each layer: ReLU(W x + b)
            X[idx] = torch.relu(torch.sparse.mm(Wm, X) + b)
        cols = []
        for e in self.out:
            y = torch.full((bits.shape[0],), float(e.c), dtype=self.dtype)
            for k, w in e.t.items():
                y = y + w * X[k]
            cols.append(y)
        return torch.stack(cols, 1)


def to_bits(msgs):
    return torch.tensor([[(m[i // 8] >> (7 - i % 8)) & 1 for i in range(512)] for m in msgs])


def to_hex(row):
    v = 0
    for x in row.tolist():
        if x != x or abs(x) > 1e6:  # NaN / overflow -> precision broke
            return "(broken: overflow/NaN)"
        v = (v << 1) | int(round(x))
    return f"{v:064x}"


def run(mode, dtype, fan=6, B=64, show=False):
    t0 = time.time()
    net, out = build(mode, fan)
    model = ShaNet(net, out, dtype)
    weights = sum(len(t) for t, _ in net.rows)
    biases = sum(1 for _, c in net.rows if c != 0)
    msgs = [os.urandom(64) for _ in range(B - 2)] + [b"\x00" * 64, b"\xff" * 64]
    y = model(to_bits(msgs))
    exact01 = bool(((y == 0) | (y == 1)).all())
    got = [to_hex(r) for r in y]
    ok = sum(g == hashlib.sha256(m).hexdigest() for g, m in zip(got, msgs))
    print(f"[{mode:4s} {str(dtype).split('.')[-1]:7s} fan={fan}] neurons={len(net.rows):,} "
          f"weights={weights:,} biases={biases:,} layers={len(model.layers):,} "
          f"match={ok}/{B} exact01={exact01} ({time.time() - t0:.1f}s)")
    if show:
        print("   msg    :", msgs[0].hex()[:32], "...")
        print("   NN     :", got[0])
        print("   hashlib:", hashlib.sha256(msgs[0]).hexdigest())
    return ok == B


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run("bool", torch.float64, show=True)
    run("bool", torch.float32)
    run("int", torch.float64, show=True)
    run("int", torch.float32)
