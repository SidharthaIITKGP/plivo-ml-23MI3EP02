"""Print legal width candidates for a vocab/layer/head configuration."""
import argparse

from model import Config, GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_params", type=int, default=2_000_000)
    args = parser.parse_args()

    for layers in args.layers:
        best = None
        for width in range(args.heads, 320, args.heads):
            cfg = Config()
            cfg.vocab_size = args.vocab_size
            cfg.n_layer = layers
            cfg.n_head = args.heads
            cfg.n_embd = width
            cfg.block_size = args.block_size
            params = GPT(cfg).n_params()
            if params <= args.max_params:
                best = (width, params)
        print(f"layers={layers} best_width={best[0]} params={best[1]}")


if __name__ == "__main__":
    main()
