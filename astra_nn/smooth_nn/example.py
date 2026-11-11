"""Run the smooth-gate SHA model on 64 raw message bytes."""

from __future__ import annotations

import argparse
import hashlib

import torch

from sha256_nn import bits_to_bytes, bytes_to_bits
from smooth_nn.network import DEFAULT_CHECKPOINT, SmoothSHA256NN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message-hex", default=bytes(range(64)).hex(),
                        help="exactly 64 raw bytes")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    message = bytes.fromhex(args.message_hex)
    if len(message) != 64:
        parser.error("message must encode exactly 64 bytes")
    model = SmoothSHA256NN(checkpoint=args.checkpoint or DEFAULT_CHECKPOINT)
    with torch.inference_mode():
        digest = bits_to_bytes(model(bytes_to_bits(message)))
    print(f"smooth network: {digest.hex()}")
    print(f"hashlib:        {hashlib.sha256(message).hexdigest()}")
    print(f"matches:        {digest == hashlib.sha256(message).digest()}")


if __name__ == "__main__":
    main()
