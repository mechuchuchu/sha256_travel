# SHA-256 scaled-ELU adaptive sign-SGD

This directory contains a self-contained experiment script, its report, and
the logs used for the current analysis. The XOR model's 128-bit weights are
both stored as a binary file and hard-coded into the standalone SHA script.

## Run

Requirements: Python, PyTorch, and NumPy. The script does not import any of the
other project Python files.

```bash
python sha256_xor_elu_scaled_sgd_adaptive.py \
  --steps 1000 --device auto --log results/my_run.log
```

Use `--device cpu` or `--device cuda` to select a device explicitly. The seed
defaults to `20261110`; change it with `--seed`. The script prints every tenth
step and flushes each row to the log as it runs.

To retrain and export the XOR model with mpmath 128-bit backpropagation:

```bash
python train_xor_net_float128.py --steps 20000 --precision 128
```

The trainer writes `xor_net_f128.bin` and its JSON metadata. Its 528 parameter
bytes are also embedded in the standalone SHA script. PyTorch evaluates that
script in float64, so the embedded binary128 weights are rounded to float64
when loaded into tensors. The high-precision trainer itself only requires
mpmath.

## Files

- `sha256_xor_elu_scaled_sgd_adaptive.py`: standalone implementation.
- `train_xor_net_float128.py`: manual 128-bit mpmath XOR training/export.
- `xor_net_f128.bin` and `xor_net_f128.json`: 33 big-endian IEEE-754
  binary128 parameters (528 bytes) and their layout metadata.
- `REPORT.md`: experiment design and current findings.
- `results/`: current and earlier logs for this optimizer.

The forward pass is a differentiable neural relaxation of SHA-256 that uses a
hard-coded scaled-ELU XOR network. It is an optimization experiment, not a
replacement for the standard SHA-256 implementation in `hashlib`.
