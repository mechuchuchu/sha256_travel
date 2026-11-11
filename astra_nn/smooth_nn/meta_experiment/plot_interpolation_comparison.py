"""Compare saved interpolation measurements from the original and trained models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=HERE.parent / 'input_interpolation_0p009.json')
    parser.add_argument('--trained', type=Path, default=HERE / 'input_interpolation_trained.json')
    parser.add_argument('--output', type=Path, default=HERE / 'input_interpolation_comparison.png')
    args = parser.parse_args()
    original = json.loads(args.original.read_text())
    trained = json.loads(args.trained.read_text())
    for key in ('start_message', 'end_message', 'target_digest'):
        if original[key] != trained[key]:
            parser.error(f'{key} differs between measurements')
    ts = [row['t'] for row in original['samples']]
    if ts != [row['t'] for row in trained['samples']]:
        parser.error('interpolation samples differ between measurements')
    old_losses = [row['mse'] for row in original['samples']]
    new_losses = [row['mse'] for row in trained['samples']]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
    for ax, inner in zip(axes, (False, True)):
        sl = slice(1, -1) if inner else slice(None)
        ax.plot(ts[sl], old_losses[sl], label='Original 0.009', color='#2563eb', marker='.', markersize=3)
        ax.plot(ts[sl], new_losses[sl], label='Fine-tuned', color='#dc2626', marker='.', markersize=3)
        ax.set(xlim=(0, 1), ylabel='MSE', title='Interior samples, enlarged' if inner else 'SHA-256 interpolation: original and fine-tuned')
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[1].set_xlabel('Interpolation t: (1 - t) × start + t × end')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f'Saved {args.output.resolve()}')
    closest = min(range(len(ts)), key=lambda index: abs(ts[index] - 0.5))
    for label, values in (('original', old_losses), ('fine-tuned', new_losses)):
        print(f'{label}: t={ts[closest]:.6g} mse={values[closest]:.9g}; interior range={min(values[1:-1]):.9g} to {max(values[1:-1]):.9g}; endpoints={values[0]:.9g}, {values[-1]:.9g}')


if __name__ == '__main__':
    main()
