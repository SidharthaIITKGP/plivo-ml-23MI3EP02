"""Average compatible checkpoints without changing the evaluator interface."""
import argparse
import json

import torch


def average_model_states(checkpoints):
    avg_state = {}
    for index, ckpt in enumerate(checkpoints):
        state = ckpt["model"]
        if index == 0:
            for key, value in state.items():
                if torch.is_floating_point(value):
                    avg_state[key] = value.detach().cpu().clone() / len(checkpoints)
                else:
                    avg_state[key] = value.detach().cpu().clone()
        else:
            for key, value in state.items():
                if torch.is_floating_point(value):
                    avg_state[key].add_(value.detach().cpu(), alpha=1.0 / len(checkpoints))
    return avg_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in args.checkpoints
    ]
    base = checkpoints[-1]
    averaged = dict(base)
    averaged["model"] = average_model_states(checkpoints)
    averaged["steps"] = base.get("steps")
    averaged["averaged_from"] = list(args.checkpoints)
    torch.save(averaged, args.out)
    print(json.dumps({"saved": args.out, "averaged_from": args.checkpoints}))


if __name__ == "__main__":
    main()
