"""Preflight checks for the final submission folder."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from evaluate import load_model
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def check_tokenizer(tok):
    samples = [
        "Plain ASCII text.\nSecond line.",
        "हिन्दी और English एक साथ।",
        "नमस्ते दुनिया!",
        "emoji: 😀🚀✨",
        "symbols: © € ₹ ™ — ‘quotes’",
        "combining: e\u0301 अ\u093c",
        "tabs\tspaces  and\nnewlines\n",
        "rare Unicode: 𐍈 𝛑 漢字 العربية",
        "",
    ]
    for text in samples:
        ids = tok.encode(text)
        assert all(isinstance(i, int) for i in ids)
        assert all(0 <= i < tok.vocab_size for i in ids)
        assert tok.decode(ids) == text


def run_evaluator(checkpoint, text_path):
    proc = subprocess.run(
        [
            sys.executable,
            "evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--text_file",
            str(text_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, proc.stdout
    return json.loads(lines[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--dev_file", default="../data/dev_eval.txt")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    model, cfg, ckpt = load_model(checkpoint)
    assert ckpt.get("steps") <= MAX_STEPS
    assert model.n_params() <= MAX_PARAMS
    tok = tokenizer_mod.load()
    check_tokenizer(tok)

    dev_result = run_evaluator(checkpoint, args.dev_file)
    assert dev_result["steps"] == ckpt.get("steps")
    assert dev_result["n_params"] == model.n_params()

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write("English हिन्दी 😀\nSecond line।\n")
        smoke_path = handle.name
    try:
        smoke_result = run_evaluator(checkpoint, smoke_path)
    finally:
        Path(smoke_path).unlink(missing_ok=True)

    print(json.dumps({"ok": True, "dev": dev_result, "smoke": smoke_result}))


if __name__ == "__main__":
    main()
