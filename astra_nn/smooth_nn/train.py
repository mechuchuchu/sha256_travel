"""Train smooth, truth-table-preserving surrogates for the SHA gate modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sha256_nn import SHA256NN
from smooth_nn import GATE_DIMS, GATE_NAMES, build_smooth_gate
from smooth_nn import gate_paths, source_gate_map


HERE = Path(__file__).resolve().parent


def path_samples(name, device="cpu", dtype=torch.float64, count=513):
    points = []
    for start, end, _ in gate_paths(name):
        a = torch.tensor(start, device=device, dtype=dtype)
        b = torch.tensor(end, device=device, dtype=dtype)
        ts = torch.linspace(0, 1, count, device=device, dtype=dtype).unsqueeze(-1)
        points.append(a + ts * (b - a))
    return torch.cat(points)


def curvature_penalty(model, x):
    x = x.detach().requires_grad_(True)
    outputs = model(x)
    penalty = x.new_zeros(())
    terms = 0
    for output_index in range(outputs.shape[-1]):
        first = torch.autograd.grad(outputs[:, output_index].sum(), x,
                                    create_graph=True, retain_graph=True)[0]
        for input_index in range(x.shape[-1]):
            second_row = torch.autograd.grad(first[:, input_index].sum(), x,
                                             create_graph=True, retain_graph=True)[0]
            penalty = penalty + second_row.square().mean()
            terms += 1
    return penalty / max(terms, 1)


def train_gate(name, source, *, steps, width, depth, lr, smoothness_weight,
               batch_size, seed, dtype=torch.float64):
    torch.manual_seed(seed)
    input_dim, output_dim = GATE_DIMS[name]
    model = build_smooth_gate(name, source, width=width, depth=depth, dtype=dtype)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    paths = path_samples(name, dtype=dtype)
    path_target = source(*paths.unbind(-1)).reshape(len(paths), output_dim).detach()
    random_count = batch_size // 2
    path_count = batch_size - random_count

    for step in range(1, steps + 1):
        random_x = torch.rand(random_count, input_dim, dtype=dtype)
        path_indices = torch.randint(len(paths), (path_count,))
        x = torch.cat((random_x, paths[path_indices]), dim=0)
        with torch.no_grad():
            target = source(*x.unbind(-1)).reshape(batch_size, output_dim)
        prediction = model(x)
        data_loss = (prediction - target).square().mean()

        curvature_x = torch.rand(min(16, batch_size), input_dim, dtype=dtype)
        smoothness = curvature_penalty(model, curvature_x)
        loss = data_loss + smoothness_weight * smoothness

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
            print(f"{name} step {step}/{steps}: data={data_loss.item():.6e} "
                  f"curvature={smoothness.item():.6e}", flush=True)

    model.eval()
    model.cache_anchor_correction()
    vertices = model.vertices
    with torch.no_grad():
        vertex_output = model(vertices)
        vertex_error = (vertex_output - model.truth_table).abs().max().item()
        random_x = torch.rand(4096, input_dim, dtype=dtype)
        random_target = source(*random_x.unbind(-1)).reshape(4096, output_dim)
        random_mse = (model(random_x) - random_target).square().mean().item()
        p_out = model(paths)
        path_mse = (p_out - path_target).square().mean().item()

    return model, {
        "binary_vertex_max_error": vertex_error,
        "fractional_random_mse": random_mse,
        "selected_path_mse": path_mse,
        "steps": steps,
        "width": width,
        "depth": depth,
        "smoothness_weight": smoothness_weight,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--smoothness-weight", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=72631)
    parser.add_argument("--output", type=Path, default=HERE / "smooth_gates.pt")
    args = parser.parse_args()
    if args.steps < 1 or args.width < 1 or args.depth < 1 or args.batch_size < 2:
        parser.error("steps, width, depth must be positive and batch-size at least 2")

    torch.set_num_threads(1)
    source_model = SHA256NN().double().eval()
    sources = source_gate_map(source_model)
    models = {}
    metrics = {}
    for index, name in enumerate(GATE_NAMES):
        model, metrics[name] = train_gate(
            name, sources[name], steps=args.steps, width=args.width, depth=args.depth,
            lr=args.lr, smoothness_weight=args.smoothness_weight,
            batch_size=args.batch_size, seed=args.seed + index,
        )
        models[name] = model.state_dict()
        print(f"{name} metrics: {json.dumps(metrics[name], sort_keys=True)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "dtype": "float64",
        "activation": "tanh",
        "width": args.width,
        "depth": args.depth,
        "metrics": metrics,
        "state_dicts": models,
    }, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved models to {args.output.resolve()}")
    print(f"Saved metrics to {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
