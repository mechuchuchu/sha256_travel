"""Full SHA-256 network using the trained smooth gate surrogates."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from sha256_nn import SHA256NN
from smooth_nn import GATE_NAMES, AnchoredSmoothGate, build_smooth_gate


DEFAULT_CHECKPOINT = Path(__file__).with_name("smooth_gates.pt")


class SmoothGateOp(nn.Module):
    """Adapt an AnchoredSmoothGate to ReLUGate's variadic input interface."""

    def __init__(self, gate: AnchoredSmoothGate):
        super().__init__()
        self.gate = gate
        self.input_dim = gate.input_dim
        self.output_dim = gate.output_dim

    def forward(self, *inputs: Tensor) -> Tensor:
        stacked = torch.stack(torch.broadcast_tensors(*inputs), dim=-1)
        leading_shape = stacked.shape[:-1]
        flat = stacked.reshape(-1, self.input_dim)
        output = self.gate(flat)
        return output.reshape(*leading_shape, self.output_dim)


class SmoothSHA256NN(SHA256NN):
    """SHA-256 network with smooth, truth-table-preserving gate replacements.

    Binary inputs follow the same gate truth tables as SHA256NN. Fractional
    inputs flow through the smooth tanh surrogates, producing a smooth
    piecewise-wired extension of the same SHA-256 graph.
    """

    def __init__(self, mode: str = "message", checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64):
        super().__init__(mode)
        super().double()
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported smooth gate checkpoint")
        source_gates = {name: getattr(self, name) for name in GATE_NAMES}
        replacements = {}
        for name in GATE_NAMES:
            gate = build_smooth_gate(
                name,
                source_gates[name],
                width=checkpoint["width"],
                depth=checkpoint["depth"],
                dtype=torch.float64,
            )
            gate.load_state_dict(checkpoint["state_dicts"][name], strict=True)
            gate.cache_anchor_correction()
            gate.eval()
            for parameter in gate.parameters():
                parameter.requires_grad_(False)
            replacements[name] = SmoothGateOp(gate)
        for name, gate in replacements.items():
            setattr(self, name, gate)
        self.to(device=device, dtype=dtype)
        self.eval()

    def _validate_input(self, bits: Tensor) -> None:
        if bits.ndim < 1 or bits.shape[-1] != 512:
            raise ValueError("expected shape (..., 512)")
        if bits.is_complex():
            raise ValueError("input must contain finite values in [0, 1]")
        valid = torch.isfinite(bits) & (bits >= 0) & (bits <= 1)
        if not torch.all(valid).item():
            raise ValueError("input must contain finite values in [0, 1]")
