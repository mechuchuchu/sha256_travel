# Adaptive sign-SGD on the scaled-ELU SHA-256 surrogate

## Goal

Optimize a 512-bit padded message block so the relaxed model's 256 output
values approach a random 256-bit target under mean squared error (MSE). The
optimizer keeps only the sign of the input gradient, then tests a learning
rate grid at every step and selects the candidate with the lowest next-step
MSE.

## Model and update rule

The script contains the SHA-256 constants, message schedule, and 64-round
compression structure. XOR is replaced with a trained `Linear(2, 8) -> ELU ->
Linear(8, 1) -> scaled sigmoid` network. Addition uses a relaxed ripple-carry
formula. This creates a differentiable surrogate; it is not the bit-exact
standard SHA-256 function for fractional inputs.

For current input `x`, target `y`, and model output `f(x)`, the objective is

`MSE(x) = mean((f(x) - y) ** 2)`.

The gradient direction is `sign(∂MSE/∂x)`. Each step evaluates 39 candidates
from `1e-20` through `1e-1` using

`x_next = clamp(x - learning_rate * sign(gradient), 0, 1)`.

The lowest-MSE candidate becomes the next input. Input and loss calculations
use float64. The default seed is `20261110`; the run uses a random 32-byte
message and a separately generated random 256-bit target.

## Findings so far

- The earlier one-step grid chose `3e-2`, with loss changing from
  `0.476007392446950006` to `0.476007392446949062`.
- In the first per-step run, the learning-rate floor was `1e-16`. Near step 35,
  the optimizer alternated between two nearby input states and the loss
  alternated by about `6e-12`.
- The current rerun extends the grid down to `1e-20`. It first selected
  `1e-20` at step 35. The maximum input change was about `1.36e-20`; the
  measured loss change was `-5.55111512312578270e-17`, approximately one
  float64 spacing at this loss magnitude.
- The loss fell from `0.476007392446950006` to
  `0.424877264097074658` by step 34. Through step 136, the displayed loss had
  not improved further, although the input continued to move by tiny amounts.

## 128-bit XOR network

The 2-8-1 scaled-ELU XOR network was retrained for 20,000 Adam updates using
hand-coded reverse-mode derivatives and `mpmath` at 128 binary bits after
fixing best-checkpoint export. The best checkpoint occurred after 10,860
updates, with training-precision MSE `4.3182081165225427e-78`. The trainer now
includes the final update in its checkpoint comparison and serializes
`best_values`, rather than the final iterate.

Reloading the saved binary128 weights and evaluating them at 180-bit precision
gives XOR MSE `1.3511927935800097e-76`, versus `2.4918012890942453e-24` for the
previous file. The saved model's outputs for `00` and `11` are approximately
`9.81e-39` and `1.83e-38`; the other two outputs are close to one. Export rounding
and evaluation precision account for the difference from the training MSE.
Metadata records `best_mse` separately from the saved-model `final_mse` and
outputs, along with the precision used to evaluate the exported model.

Its 33 parameters are stored in
`xor_net_f128.bin` as 528 bytes of big-endian IEEE-754 binary128 data, with the
layout documented in `xor_net_f128.json`. The same bytes are embedded as a hex
literal in `sha256_xor_elu_scaled_sgd_adaptive.py`. PyTorch has no native
float128 tensor dtype, so the SHA experiment decodes those bytes to float64
when it creates model tensors. The XOR weights are preserved in binary128;
the SHA forward pass and input optimization still run in float64.

## Historical float64 run status

At the earlier check, the 1000-step run with the `1e-20` minimum had reached
step 136. Its log was `results/adaptive_1000_min1e20.log`, which is not present
in this checkout. This run began before the binary128 XOR weights
were trained, so it uses the earlier rounded weights from the original
project module. It does not measure the newly embedded binary128 checkpoint.

## Interpretation and limits

The negative `loss_change` means the candidate loss was lower than the loss at
the start of that step. Near `0.425`, a change around `5.55e-17` is one
representable float64 step, so the apparent improvement is at the precision
limit of this calculation. Lowering the learning-rate floor broke the earlier
two-state oscillation and produced a one-step representable decrease, but the
loss then plateaued at the printed precision. Higher-precision input state,
forward values, gradients, and updates would be needed to investigate smaller
changes reliably.

This experiment evaluates MSE only. It does not measure hash correctness or
cryptographic security, and the learned XOR relaxation is not a substitute
for standard SHA-256.

## mpmath forward/backward implementation

`sha256_mpmath_backward.py` now evaluates the learned-XOR surrogate with an
isolated mpmath context at 128 significand bits. It reads the actual binary128
weight bytes directly and keeps forward values, local Jacobians, accumulated
input gradients, and optional sign-SGD updates in that context. It does not
depend on PyTorch or NumPy. XOR network derivatives and carry/choose/majority
derivatives are recorded on a scalar reverse-mode tape; rotations and shifts
reuse node references.

A full 64-round evaluation for the message `abc`, targeting
`ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`, completed with
75,946 tape nodes and 512 finite, nonzero input gradients. MSE was approximately
`2.5265156864e-24`, and the maximum absolute input gradient was approximately
`3.9973822229e-32`. Forward took 12.93 seconds and backward took 0.52 seconds on
the development CPU. The complete values are saved in
`results/sha256_mpmath_backward_128.json`. This is a forward/backward smoke test,
not a preimage optimization result; its target is the known hash of its input.

The above smoke-test artifact used the old, final-iterate weights and is kept
as a historical result. The corrected trainer now exports the best checkpoint,
and the mpmath implementation loads that new binary file on startup. Its
precision setting does not add precision to the stored 113-bit significands.

## 128-bit adaptive sign-SGD

`sha256_xor_elu_scaled_sgd_adaptive_128.py` applies the original adaptive update
rule through the mpmath engine. It preserves the NumPy-seeded input and target,
the 39-candidate grid from `1e-20` to `1e-1`, projection to `[0, 1]`, and cycle
replay. Learning rates, candidate comparisons, state comparisons, and logged
model values retain the configured mpmath precision. Target bytes in the new
log use big-endian order; the old log serialized native-endian words.

A complete 128-bit step with seed `20261110` and all 64 rounds took 412.4 seconds
on the development CPU. It selected `0.1`, reducing loss from approximately
`0.4765624999995988337324511136148584169672614` to
`0.4687499999994853675690903454916395072318703`. The maximum input gradient was
approximately `1.2055136769e-20`. See `results/adaptive_128_1_step.log`.
This historical run used the binary128 file from before the best-checkpoint
export fix and is separate from the older float64 results that used the
previous rounded weights. Eleven tests passed at the time,
covering the engine's derivatives and SHA wiring, seeded input compatibility,
sub-float64 updates, candidate ties, clipping, and two-state cycle replay.

## Verification after best-checkpoint export

All 14 tests passed after retraining and synchronizing the embedded PyTorch
weights. The three export regressions cover an earlier checkpoint beating the
final iterate, an improving final update, and equality of the embedded bytes
with the binary file. Exported metadata is also checked through the consuming
mpmath engine's independent decoder and XOR evaluator.

A new 64-round mpmath forward/backward run for `abc`, using its known SHA-256
digest as the target, produced MSE `1.6032554778113813e-76` and 512 finite,
nonzero input gradients at 128-bit precision. The maximum absolute gradient
was `8.24274748701376e-59`. Forward took 9.97 seconds and backward 0.34 seconds;
see `results/sha256_mpmath_backward_best_128.json`. A PyTorch float64 run with
the same message and target also completed with finite outputs and gradients.
The earlier adaptive optimization logs have not been rerun with the new weights.
