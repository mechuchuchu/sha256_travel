"""Smooth neural surrogates for the fixed SHA-256 gates."""

from __future__ import annotations

from itertools import product

import torch
from torch import nn


GATE_NAMES = ("xor_and", "xor3", "choose", "majority", "prefix")
GATE_DIMS = {
    "xor_and": (2, 2),
    "xor3": (3, 1),
    "choose": (3, 1),
    "majority": (3, 1),
    "prefix": (4, 2),
}


class SmoothMLP(nn.Module):
    """A wider smooth MLP used to distill one fixed ReLU gate."""

    def __init__(self, input_dim: int, output_dim: int, width: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        size = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(size, width), nn.Tanh()))
            size = width
        layers.append(nn.Linear(size, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AnchoredSmoothGate(nn.Module):
    """Smooth MLP with a polynomial correction fixing every binary vertex.

    For each Boolean vertex v, the multilinear basis phi_v is 1 at v and 0
    at every other Boolean vertex. Adding its weighted residual makes the
    model reproduce the source gate's truth table exactly while remaining
    smooth throughout [0, 1]^d.
    """

    def __init__(self, vertices, truth_table, width=64, depth=3):
        super().__init__()
        self.input_dim = vertices.shape[1]
        self.output_dim = truth_table.shape[1]
        self.net = SmoothMLP(self.input_dim, self.output_dim, width, depth).to(dtype=vertices.dtype)
        self.register_buffer("vertices", vertices.clone())
        self.register_buffer("truth_table", truth_table.clone())
        self.register_buffer("anchor_correction", torch.zeros_like(truth_table))
        self._anchors_cached = False

    def cache_anchor_correction(self):
        """Cache fixed anchor residuals after the MLP has finished training."""
        with torch.no_grad():
            self.anchor_correction.copy_(self.truth_table - self.net(self.vertices))
        self._anchors_cached = True

    def forward(self, x):
        single = x.ndim == 1
        x = x.reshape(-1, self.input_dim)
        corrections = self.anchor_correction if self._anchors_cached else \
                      self.truth_table - self.net(self.vertices)
        factors = torch.where(self.vertices.unsqueeze(0) > 0,
                              x.unsqueeze(1), 1 - x.unsqueeze(1))
        basis = factors.prod(dim=-1)
        y = self.net(x) + basis @ corrections
        binary_rows = ((x == 0) | (x == 1)).all(dim=-1)
        weights = 2 ** torch.arange(self.input_dim - 1, -1, -1,
                                    device=x.device, dtype=torch.int64)
        safe_binary = x.clamp(0, 1).round().to(torch.int64)
        indices = (safe_binary * weights).sum(dim=-1)
        exact = self.truth_table[indices]
        # Match the Boolean table exactly in the forward value while
        # retaining the smooth anchored model's derivative.
        y = y + (exact - y).detach() * binary_rows.unsqueeze(-1)
        return y[0] if single else y


def binary_vertices(input_dim, *, dtype=torch.float64):
    return torch.tensor(list(product((0.0, 1.0), repeat=input_dim)), dtype=dtype)


def source_gate_map(source_model):
    return {name: getattr(source_model, name) for name in GATE_NAMES}


def build_smooth_gate(name, source_gate, *, width=64, depth=3, dtype=torch.float64):
    input_dim, _ = GATE_DIMS[name]
    vertices = binary_vertices(input_dim, dtype=dtype)
    with torch.no_grad():
        truth = source_gate(*vertices.unbind(-1)).reshape(len(vertices), -1)
    return AnchoredSmoothGate(vertices, truth, width=width, depth=depth)


def gate_paths(name):
    if name == "xor_and":
        return [((0, 0), (1, 1), "(0,0) → (1,1)")]
    if name in ("xor3", "choose", "majority"):
        dim, _ = GATE_DIMS[name]
        return [(tuple(0 for _ in range(dim)), tuple(1 for _ in range(dim)),
                 "all-zero → all-one")]
    if name == "prefix":
        return [((0, 0, 0, 0), (0, 1, 1, 0), "generate"),
                ((0, 0, 0, 0), (0, 1, 0, 1), "propagate")]
    raise KeyError(name)
