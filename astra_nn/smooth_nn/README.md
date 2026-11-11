# Smooth SHA gate surrogates

This directory contains a separate set of smooth neural models for the five
fixed gates in `sha256_nn.py`: XOR/AND, XOR3, Choose, Majority, and carry-prefix.
Each model is a 64-wide, three-hidden-layer `tanh` MLP trained in float64 to
distill the source gate on fractional inputs.

The training loss combines source-output MSE with an input-Hessian penalty.
`tanh` keeps the model differentiable throughout the input cube; the Hessian
penalty discourages sharp bends. A multilinear residual correction anchors the
model at every Boolean input vertex. The forward result is patched by less than
rounding error there so the saved model returns the source truth table exactly.

Run training from the repository root:

```bash
uv run python -m smooth_nn.train --steps 1000 --width 64 --depth 3 \
  --batch-size 64 --smoothness-weight 0.001
uv run python -m smooth_nn.plot
```

The output files are `smooth_gates.pt`, `smooth_gates.metrics.json`, and
`smooth_gate_comparison.png` in this directory. The plot compares each source
ReLU curve with its smooth surrogate along selected binary-to-binary paths.
Fractional outputs approximate the original ReLU extension; they are not
promised to match exactly. Only the Boolean truth tables are exact.

`SmoothSHA256NN` replaces every gate in the full 512-to-256 SHA graph:

```python
import torch
from sha256_nn import bytes_to_bits, bits_to_bytes
from smooth_nn.network import SmoothSHA256NN

message = bytes(range(64))
model = SmoothSHA256NN(mode="message")
with torch.inference_mode():
    digest = bits_to_bytes(model(bytes_to_bits(message)))
```

The full model still uses the original SHA wiring, constants, padding, and
rounds. Its gate inputs and outputs are exact Boolean values when the original
input is binary, so the resulting digest matches the standard SHA-256 output.
For fractional inputs, each gate uses its smooth surrogate.
