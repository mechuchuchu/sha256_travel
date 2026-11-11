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

## Optimize a message toward a target digest

`optimize.py` uses Adam on 512 continuous input values in `[0, 1]` and
minimizes output-bit MSE against a target digest. A small binarization penalty
encourages values toward 0 or 1. By default, the target is the SHA-256 digest
of `bytes(range(64))`; pass another 64-byte message or a 32-byte digest to
change it:

```bash
uv run python -m smooth_nn.optimize --steps 50 --output smooth_nn/optimization_result.json
uv run python -m smooth_nn.optimize --target-hex YOUR_64_HEX_DIGIT_DIGEST --steps 50
```

The script also rounds the final input at 0.5 and checks its ordinary SHA-256
digest. A low loss for the relaxed smooth model does not imply that this
rounded message has the target digest.

For an L-BFGS experiment on the `0.009` checkpoint, run:

```bash
uv run python -m smooth_nn.optimize_lbfgs --steps 5
```

It uses the same default target and random seed as the Adam example. A sigmoid
keeps the input in `[0, 1]`, and a fixed loss scale makes the small gradients
usable by the strong Wolfe line search. The result is written to
`optimization_0p009_lbfgs.json`.

## Compare a larger, smoother gate model

`stability_experiment.py` is a separate experiment. It trains 128-wide gates
for 1000 steps with smoothness weight `0.1`, saves them to
`smooth_gates_stability.pt`, then measures target-loss input gradients for
both that checkpoint and the default checkpoint at the same fractional
inputs:

```bash
uv run python -m smooth_nn.stability_experiment
uv run python -m smooth_nn.stability_experiment --smoothness-weight 0.01 \
  --output smooth_nn/smooth_gates_stability_0p01.pt \
  --report smooth_nn/smooth_gates_stability_0p01.gradients.json
uv run python -m smooth_nn.stability_experiment --smoothness-weight 0.003 \
  --output smooth_nn/smooth_gates_stability_0p003.pt \
  --report smooth_nn/smooth_gates_stability_0p003.gradients.json
```

Training steps, width, depth, smoothness weight, and gradient sample count can
be changed with command-line options. The alternate checkpoint, gate metrics,
and gradient comparison are saved separately from the default model files.
The report includes gradient norms and their variation across the fixed input
samples, so vanishing gradients are visible alongside exploding gradients.

To train and compare all weights from `0.004` through `0.009` in steps of
`0.001`, run:

```bash
uv run python -m smooth_nn.sweep_smoothness
```

This saves one checkpoint and gate-metrics file per weight, plus the combined
gradient report `smooth_gates_sweep.gradients.json`.
