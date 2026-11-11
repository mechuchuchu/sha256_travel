# Full-SHA midpoint fine-tuning experiment

`original_0p009.pt` is a copy of `../smooth_gates_sweep_0p009.pt`; training
writes a separate checkpoint and refuses to overwrite either the source or an
existing output/report. The final `trained_0p009.pt` is tracked by Git; other
checkpoints remain ignored.

```bash
uv run python -m smooth_nn.meta_experiment.train --steps 6 --validation-pairs 4 --lr 0.0001
```

At each step, independent random 64-byte messages `a`, `b` produce the
midpoint `(bits(a) + bits(b))/2`, with the SHA-256 of `a` as the target. The
training objective contains `(midpoint_mse - 0.23)^2`, a gate curvature
penalty, and an MSE between each underlying MLP's raw output and the original
gate truth table at binary vertices. The anchored gate itself preserves the
binary truth table exactly; its binary loss alone would always be zero.

The input-gradient term uses two extra full-hash evaluations at the midpoint
plus and minus a random sign direction on the bits where the messages differ.
A central finite difference estimates a directional input derivative, and
backpropagation through those evaluations approximates its mixed parameter
and input gradient. It is not exact second-order autograd, and one direction
per step is a noisy estimate of the input-gradient norm in that subspace.

The report records input messages, losses, and held-out midpoint losses before
and after training. It also checks both endpoints of each validation pair
against `hashlib`. A small run cannot establish performance on all inputs or
that the target MSE is achievable; continuous midpoint outputs are not hashes.
The saved checkpoint's original gate metrics are labeled as source metrics,
not as measurements of the fine-tuned gates.

To repeat the original message interpolation with the trained checkpoint and
plot both results together:

```bash
uv run python -m smooth_nn.plot_input_interpolation \
  --checkpoint smooth_nn/meta_experiment/trained_0p009.pt \
  --output smooth_nn/meta_experiment/input_interpolation_trained.png
uv run python -m smooth_nn.meta_experiment.plot_interpolation_comparison
```
