"""Continuous-input variant of the fixed SHA-256 gate network.

Unlike :class:`sha256_nn.SHA256NN`, this class accepts finite values in [0, 1].
Fractional inputs are passed through the same ReLU circuit as-is; outputs are
not guaranteed to be binary and are not SHA-256 digests.
"""

from __future__ import annotations

import torch
from torch import Tensor

from sha256_nn import SHA256NN


class FractionalSHA256NN(SHA256NN):
    """SHA circuit that allows finite fractional values between 0 and 1."""

    def _validate_input(self, bits: Tensor) -> None:
        if bits.ndim < 1 or bits.shape[-1] != 512:
            raise ValueError("expected shape (..., 512)")
        if bits.is_complex():
            raise ValueError("input must contain finite values in [0, 1]")
        valid = torch.isfinite(bits) & (bits >= 0) & (bits <= 1)
        if not torch.all(valid).item():
            raise ValueError("input must contain finite values in [0, 1]")
