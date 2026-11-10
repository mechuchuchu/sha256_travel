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

The trainer writes the best observed checkpoint (including the final Adam
update) to `xor_net_f128.bin` and its JSON metadata. `best_step` counts completed
Adam updates, and `best_mse` is measured at training precision before export.
`final_mse` and `xor_outputs_00_01_10_11` describe the saved binary128 weights,
reloaded and evaluated at `evaluation_precision_bits` (at least 180 bits).
Binary128 rounding can make this MSE differ from `best_mse`.

The current 528 parameter bytes are also embedded in the standalone SHA script.
After retraining, that script's `_XOR_NET_F128_HEX` must be synchronized with the
new binary file; the export tests check this. The mpmath scripts load the binary
file directly on startup. PyTorch evaluates the standalone
script in float64, so the embedded binary128 weights are rounded to float64
when loaded into tensors. The high-precision trainer itself only requires
mpmath.

## mpmath 128-bit forward/backward

The CPU implementation below requires only `mpmath`. It runs all 64 rounds of
the same learned-XOR relaxation and computes MSE derivatives for all 512 input
values, including padding. Model values, local derivatives, gradient accumulation,
and updates use 128 binary significand bits throughout; no float64 conversion is
used. The binary128 weight file contains 113 significand bits per parameter.

```bash
python sha256_mpmath_backward.py
```

This performs one forward/backward evaluation and writes the input, 256 output
values, 512 gradients, loss, and timings to `results/sha256_mpmath_backward.json`.
Numeric model data is saved as decimal strings to preserve precision. A measured
64-round run took approximately 13 seconds forward and 0.5 seconds backward on
the development CPU; timings depend on the machine and input.

For repeated evaluations with a projected sign-SGD update between them:

```bash
python sha256_mpmath_backward.py --steps 2 --lr 1e-20
```

This uses a fixed learning rate, without the PyTorch script's candidate search.
`--precision` changes the arithmetic precision (minimum 113); `--message-hex`
accepts up to 55 message bytes and `--target-hex` accepts exactly 32 target bytes.
`--weights` selects a binary128 file. The default seed is `20261110`, but this
script uses Python's `random.Random`, so its default samples differ from the
NumPy-based script. `--rounds` can reduce the rounds for debugging.

The Python API returns mpmath values directly:

```python
from sha256_mpmath_backward import SHA256MP, bytes_to_bits, pad_message

model = SHA256MP(precision=128)
x = bytes_to_bits(pad_message(b"abc"))
target = bytes_to_bits(bytes(32))
result = model.evaluate(x, target)
print(result.loss, result.gradient[0])
```

Supply decimal strings or mpmath values for fractional inputs. Bit arrays store
big-endian 32-bit words with the least-significant bit first within each word.
Weights remain fixed: backward computes input gradients, not weight gradients.
The saved file `results/sha256_mpmath_backward_128.json` records a historical run
for `abc` with its standard SHA-256 digest as the target, using the weights from
before the best-checkpoint export fix.
The corresponding run with the corrected best weights is saved in
`results/sha256_mpmath_backward_best_128.json` (MSE approximately `1.60e-76`).

Run numerical derivative and SHA structure checks with:

```bash
python -m unittest -v test_sha256_mpmath_backward.py
```

## mpmath 128-bit adaptive sign-SGD

`sha256_xor_elu_scaled_sgd_adaptive_128.py` ports the original adaptive optimizer
to the mpmath engine. It uses the same 39 learning rates (`1e-20` through `1e-1`),
gradient signs, `[0, 1]` projection, first-candidate tie breaking, and exact-state
cycle replay. The smallest candidate loss is accepted even if it exceeds the
base loss, matching the original algorithm. Precision defaults to 128 binary
significand bits for the full computation, including learning rates and updates.

```bash
python sha256_xor_elu_scaled_sgd_adaptive_128.py \
  --steps 1 --device cpu --log results/adaptive_128.log
```

This requires `mpmath` and NumPy, plus `sha256_mpmath_backward.py` and the weight
file in this directory. NumPy is used only to generate exactly the same message
and target bytes as the original adaptive script for a given seed. The new log
prints target bytes in big-endian order; the old script prints native-endian
target words. The underlying target bits agree.

`--steps` defaults to 1000 as in the original, but CPU evaluation of all 39
candidates takes several minutes per step. Candidate progress is printed during
the search; each completed step is flushed to the log using the original CSV
columns, with enough digits to preserve mpmath precision. `--device auto` also
selects CPU. `--precision`, `--weights`, and diagnostic `--rounds` are supported.
To extend the learning-rate grid below `1e-20`, use e.g.
`--min-lr-exponent -40` (79 candidates). This optimizer remains a continuous
SHA-256 surrogate, and padding bits are optimized along with the message bits.

The historical full 64-round, 39-candidate smoke test completed in 412.4 seconds on the
development CPU. It selected learning rate `0.1`, reducing MSE from approximately
`0.4765625` to `0.46875`; the log is `results/adaptive_128_1_step.log`.
It used the weights from before the best-checkpoint export fix.

```bash
python -m unittest -v
```

## Multiprocess candidate evaluation

`sha256_xor_elu_scaled_sgd_adaptive_128_multiprocess.py` is a separate copy of
the 128-bit adaptive optimizer with parallel candidate evaluation:

```bash
python sha256_xor_elu_scaled_sgd_adaptive_128_multiprocess.py \
  --steps 1 --workers 8 --log results/adaptive_128_multiprocess.log
```

The parent computes one forward/backward pass per step. Persistent worker
processes each load the current binary128 weight file once and evaluate the
candidate forward passes. The process pool is reused across steps. Inputs,
targets, and losses travel as exact mpmath integer tuples, preserving the
configured precision without conversion through float64. Results are consumed
in candidate order, preserving first-candidate ties and exact-state cycle replay.

Omitting `--workers` uses the CPUs available to the process, capped at the
candidate count. `--workers 1` runs sequentially without creating a process
pool. Other arguments, including `--precision`, `--weights`, `--rounds`, and
`--min-lr-exponent`, work as in the original. Memory usage grows with the number
of workers; the gradient pass remains sequential, so speedup depends on the
workload and available CPU resources. This version uses the `spawn` start method.

Run the process and precision checks with:

```bash
python -m unittest -v test_sha256_adaptive_128_multiprocess.py
```

A 64-round, one-step comparison using three candidates (`--min-lr-exponent -2`)
produced identical CSV rows: 29.77 seconds sequentially versus 19.05 seconds
with three workers, including startup (1.56x overall speedup on this machine).
This is a reduced-grid measurement, not a benchmark of the default 39
candidates. Commands, timings, and the weight-file hash are recorded in
`results/adaptive_128_multiprocess_benchmark.json`.

## Files

- `sha256_xor_elu_scaled_sgd_adaptive.py`: standalone implementation.
- `sha256_xor_elu_scaled_sgd_adaptive_128.py`: mpmath adaptive sign-SGD counterpart.
- `sha256_xor_elu_scaled_sgd_adaptive_128_multiprocess.py`: parallel candidate
  evaluation with a persistent process pool.
- `test_sha256_adaptive_128.py`: high-precision selection and cycle replay checks.
- `test_sha256_adaptive_128_multiprocess.py`: spawned-process precision, candidate
  ordering, worker errors, CLI parity, and sequential cycle replay checks.
- `train_xor_net_float128.py`: manual 128-bit mpmath XOR training/export.
- `test_train_xor_net_float128.py`: best-checkpoint export, saved metrics, and
  embedded-weight synchronization checks.
- `sha256_mpmath_backward.py`: mpmath SHA surrogate and reverse-mode input gradients.
- `test_sha256_mpmath_backward.py`: finite-difference and SHA structure checks.
- `xor_net_f128.bin` and `xor_net_f128.json`: 33 big-endian IEEE-754
  binary128 parameters (528 bytes) and their layout metadata.
- `REPORT.md`: experiment design and current findings.
- `results/`: current and earlier logs for this optimizer.

The forward pass is a differentiable neural relaxation of SHA-256 that uses a
hard-coded scaled-ELU XOR network. It is an optimization experiment, not a
replacement for the standard SHA-256 implementation in `hashlib`.
