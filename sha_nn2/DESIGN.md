# SHA-256 Layer-Matching MLP with Learned Skip Connections

**Status:** Design draft; implementation has not started.

## 1. Purpose

Build a trainable student network for the fixed-input SHA-256 network in [`sha_nn.py`](sha_nn.py). The student receives a 512-bit message and is trained to reproduce both selected internal activations of the existing network and its final 256-bit hash output.

The student is intentionally very deep. Its intermediate widths are based on 2,000 randomly selected teacher layers. Each selected teacher layer is represented by a three-affine-layer MLP block with two 2,048-unit hidden layers. Learned skip coefficients connect earlier student activations to later student layers when their widths match.

This document records the intended architecture and explicit assumptions for a later implementation. Items that still need a decision are listed in [Open decisions](#12-open-decisions).

## 2. Existing teacher network

`sha_nn.py` constructs a fixed ReLU network that computes SHA-256 for exactly one 512-bit (64-byte) input message. SHA-256 padding is built into the network as a second block. Inputs and hash outputs are represented as bits, ordered MSB first at the Python interface.

The source has two construction modes, selected by `build(mode, fan=6)`:

| Teacher mode | Hidden layers at `fan=6` | Notes |
|---|---:|---|
| `bool` | 4,412 | Bit-level arithmetic. |
| `int` | 13,502 | Integer arithmetic encoded through ReLU thresholds; requires float64 for reliable exact arithmetic. |

These counts are the number of non-input depth groups (`len(ShaNet.layers)`). The teacher mode must be fixed before sampling because it determines the available layers and widths.

## 3. Scope and terminology

- **Teacher:** the frozen network produced by `sha_nn.build(mode, fan=6)` and wrapped by `ShaNet`.
- **Teacher layer:** one depth group in `ShaNet.layers`, not an individual neuron or sparse weight row.
- **Width:** the number of neurons in a teacher depth group, or the size of a student activation vector.
- **Anchor:** a sampled teacher layer whose activation is a direct training target for a corresponding student block output.
- **Student layer:** one affine transformation in the new MLP. A three-affine-layer block therefore contributes three student layers.
- **Activation:** the vector emitted by a student layer after its activation function.

The term “copy” means copying the selected teacher layer widths and using the teacher activations as targets. It does not mean copying teacher weights into the student.

## 4. Sampling 2,000 teacher layers

1. Construct the teacher with a fixed mode and `fan=6`.
2. Sample exactly 2,000 distinct teacher layers uniformly without replacement, using a recorded random seed.
3. Sort the sampled layers by increasing teacher depth. Random selection determines which depths are included; sorting preserves the teacher's forward order.
4. Record each selected layer's depth, width, and stable identifier. Let the ordered widths be `w[1] ... w[2000]`.
5. For every training input, collect the teacher activation at each sampled layer. These 2,000 vectors are the anchor targets for that input.

Sampling is over layers, not over neurons. A layer's entire activation vector is the target.

## 5. Student architecture

### 5.1 Main path

Build one three-affine-layer block for each sampled anchor. Define `w[0] = 512`, the binary input width. For block `k` in `1 ... 2000`:

```text
w[k-1] -> 2048 -> 2048 -> w[k]
```

The output of block `k` is the student activation corresponding to sampled teacher layer `k`. Consequently:

- The first block maps the 512-bit input to the first anchor width.
- Each later block maps the previous anchor representation to the next anchor width.
- The 2,000 blocks contain exactly 6,000 affine student layers.

Add a final hash head after the last anchor:

```text
w[2000] -> 256 logits
```

The student therefore has 6,001 affine layers including the output head, which is approximately 6,000 layers. The input vector is not counted as an affine layer.

### 5.2 Activations

Proposed baseline:

- Apply ReLU after the first and second affine transformations in each anchor block.
- Apply ReLU after the block's third transformation so its anchor output is non-negative, like teacher ReLU activations.
- Keep the final 256-bit hash head linear and treat its outputs as logits for the output loss.

This gives each block the following form, before skip additions are applied:

```text
h1 = ReLU(W1 x + b1)
h2 = ReLU(W2 h1 + b2)
anchor = ReLU(W3 h2 + b3)
```

The output head produces 256 logits. Applying sigmoid and thresholding each logit at 0.5 yields the predicted hash bits.

### 5.3 Widths and parameter shapes

For block `k`, the three weight matrices have shapes:

```text
W1: (2048, w[k-1])
W2: (2048, 2048)
W3: (w[k], 2048)
```

The final head has shape `(256, w[2000])`. Bias vectors have one entry per output unit.

## 6. Learned skip connections

### 6.1 Candidate connections

Create a directed skip candidate from an earlier student activation to a later student layer only when their vector widths match. Do not create self-connections or backward connections. Each candidate has one trainable scalar coefficient shared across all units in that vector. Coefficients are unconstrained real values and should initially be zero.

Let `h[i]` be an earlier activation and let student affine layer `j` emit a vector of width `d[j]`. The proposed shape-compatible residual rule adds matching earlier activations to the destination pre-activation:

```text
z[j] = W[j] h[j-1] + b[j]
       + sum(alpha[i,j] * h[i]
             for i < j if width(h[i]) == d[j])
h[j] = activation[j](z[j])
```

Here `alpha[i,j]` is one scalar per eligible ordered pair `(i, j)`. The ordinary adjacent-layer path remains present. Skip terms are summed before the destination layer's activation function.

This is the design's explicit interpretation of “add an earlier layer into a later layer.” It uses the width of the earlier activation and the destination layer's output/pre-activation width for the shape check. If “layer input” is intended literally as adding to the vector before multiplication by `W[j]`, the equation and width check need to be changed before implementation.

### 6.2 Expected size and implementation concerns

Each block contains two 2,048-wide hidden activations, so the student has about 4,000 activations of width 2,048. If all eligible same-width pairs are connected, those activations alone produce roughly eight million scalar skip coefficients. Each coefficient applies to a 2,048-element vector. The skip aggregation therefore needs an implementation designed for dense long-range connectivity; a Python loop over individual pairs is unlikely to be practical.

There is one 2,048-by-2,048 matrix per block, between its two 2,048-unit hidden layers. Across 2,000 blocks this is about 8.39 billion weights (about 33.6 GB in float32 for those matrices alone), before biases, other matrices, activations, gradients, optimizer state, or skip parameters. Standard Adam training would require several times the weight memory. Memory and compute feasibility must be measured before a full run.

## 7. Training data and targets

Inputs are 512-element binary vectors corresponding to 64-byte messages. Use the same bit order as `sha_nn.to_bits`: message bytes are expanded MSB first. The teacher's 256 output bits use the same MSB-first ordering as `build()`'s `out` list.

For each message `x`, the frozen teacher provides:

- `t[k](x)`: the full activation vector at sampled anchor `k`, of width `w[k]`.
- `y(x)`: the final 256-bit SHA-256 digest.

Generate separate training, validation, and test message sets. Sample random messages uniformly over 512-bit inputs. Include all-zero and all-one messages in evaluation for continuity with the existing script's boundary examples. Do not use test messages for optimizer updates or early stopping decisions.

Teacher activations can be cached if storage is practical. Otherwise, compute them in batches with a teacher forward path that exposes selected layer activations. Any added capture method should leave the current teacher computation and output ordering unchanged.

## 8. Losses and optimization objective

Use an anchor loss at all 2,000 sampled points and a final hash loss:

```text
L_anchor = mean over k=1..2000 of mean((student_anchor[k] - t[k])^2)
L_hash   = binary_cross_entropy_with_logits(student_logits, target_hash_bits)
L_total  = lambda_anchor * L_anchor + lambda_hash * L_hash
```

The per-anchor mean prevents wider teacher layers from dominating solely because they contain more units. `lambda_anchor` and `lambda_hash` are configuration parameters and must be logged. Their values are not fixed by this design.

Track both continuous losses and discrete hash accuracy. Convert hash logits to bits with a 0.5 sigmoid threshold (equivalently, logit threshold zero). A message counts as an exact match only if all 256 predicted bits match the teacher digest.

## 9. Evaluation and success criteria

Evaluate the frozen teacher and student on identical held-out binary messages. Report:

- Per-bit hash accuracy.
- Exact 256-bit digest match count and rate.
- Anchor mean-squared error, reported overall and by sampled layer.
- Hash and anchor losses.
- Number of teacher layers sampled, sampled widths, total student affine layers, parameter count, trainable skip coefficient count, peak memory, and throughput.
- Random seed, teacher mode, data seed, optimizer settings, and loss weights.

Exact matching on all possible messages cannot be established by finite sampling: the input space contains `2^512` messages. Any reported exact-match rate applies only to the stated evaluation distribution.

## 10. Reproducibility and checkpoints

Save enough metadata to rebuild the same experiment:

- Teacher mode and `fan`.
- Ordered sampled teacher layer identifiers, depths, and widths.
- Sampling and dataset random seeds.
- Student block dimensions and activation choices.
- Skip-edge index pairs and coefficient initialization.
- Loss weights, optimizer configuration, and training progress.
- Checkpointed model parameters and measured validation metrics.

Store the teacher sampling manifest with the checkpoint. Re-sampling after training starts would change the meaning of anchor outputs and invalidate the checkpoint.

## 11. Suggested implementation components

1. **Teacher layer manifest:** expose stable depth-group IDs, widths, and selected activations from `ShaNet`.
2. **Sampler:** choose 2,000 unique layers with a seeded RNG and sort them by teacher depth.
3. **Student builder:** create 2,000 anchor blocks and the 256-logit output head from the manifest widths.
4. **Skip builder:** enumerate eligible earlier-to-later activation pairs with matching widths; create one scalar parameter for each pair.
5. **Teacher capture and dataset pipeline:** generate message batches and corresponding anchor/hash targets.
6. **Training loop:** compute the weighted anchor and hash losses, optimize student weights and skip coefficients, and record validation metrics.
7. **Evaluation/checkpointing:** calculate exact-hash matches and save the sampling manifest and configuration alongside model weights.

## 12. Open decisions

Resolve these before implementation:

1. **Teacher mode:** sample from `bool` (4,412 layers) or `int` (13,502 layers), or make it a command-line option. This design does not select one.
2. **Skip injection point:** this document adds skips to the destination pre-activation and matches the source activation width to the destination output width. Confirm that this is the intended meaning of adding an earlier layer into a later layer.
3. **Skip scope:** confirm that every eligible earlier-to-later pair is wanted, including pairs involving the input and output head. The size estimate assumes all eligible hidden-layer pairs and excludes raw input and final logits as skip sources/targets.
4. **Loss weights:** choose initial `lambda_anchor` and `lambda_hash`, then tune on validation data.
5. **Resource target:** decide the available accelerator/host memory and whether the full approximately 8.4-billion-weight design is feasible. If not, the architecture needs a structural reduction before implementation.
6. **Sampling policy:** uniform by layer is assumed. Sampling by depth range, width, or a mixture would produce a different student.

