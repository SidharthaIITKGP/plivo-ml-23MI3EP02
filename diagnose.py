"""Extra diagnostics around the official bpb metric."""
import argparse
import json

import torch

from evaluate import bits_per_byte, load_model
import tokenizer as tokenizer_mod


def score_text(model, cfg, tok, text):
    bpb, n_scored, n_tokens = bits_per_byte(model, cfg, tok, text)
    return {"bpb": round(bpb, 4), "tokens": n_tokens, "tokens_scored": n_scored}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--text_file", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    model, cfg, ckpt = load_model(args.checkpoint)
    tok = tokenizer_mod.load()
    text = open(args.text_file, encoding="utf-8").read()
    midpoint = len(text) // 2
    paragraphs = [p for p in text.splitlines() if p]
    hindi = "\n".join(p for p in paragraphs if any(0x0900 <= ord(ch) <= 0x097F for ch in p))
    non_hindi = "\n".join(p for p in paragraphs if not any(0x0900 <= ord(ch) <= 0x097F for ch in p))

    result = {
        "full": score_text(model, cfg, tok, text),
        "first_half": score_text(model, cfg, tok, text[:midpoint]),
        "second_half": score_text(model, cfg, tok, text[midpoint:]),
        "hindi_paragraphs": score_text(model, cfg, tok, hindi) if hindi else None,
        "non_hindi_paragraphs": score_text(model, cfg, tok, non_hindi) if non_hindi else None,
        "n_params": model.n_params(),
        "steps": ckpt.get("steps"),
        "vocab_size": tok.vocab_size,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
