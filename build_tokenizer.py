"""Build assignment-legal tokenizers from train_corpus.txt only."""
import argparse
import re
from collections import Counter

import tokenizer


def build_hybrid(text):
    chars = sorted({ch for ch in text if 0x0900 <= ord(ch) <= 0x097F})
    return tokenizer.HybridCharByteTokenizer(chars)


def replace_pair(seq, pair, new_id):
    out = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


def build_bpe(text, vocab_size, max_piece_bytes):
    assert vocab_size >= 256
    segment_counts = Counter(re.findall(tokenizer.SPLIT_PATTERN, text))
    sequences = {segment: tuple(segment.encode("utf-8")) for segment in segment_counts}
    token_bytes = [bytes([i]) for i in range(256)]
    merges = []

    while len(token_bytes) < vocab_size:
        pair_counts = Counter()
        for segment, seq in sequences.items():
            weight = segment_counts[segment]
            for pair in zip(seq, seq[1:]):
                merged_len = len(token_bytes[pair[0]]) + len(token_bytes[pair[1]])
                if merged_len <= max_piece_bytes:
                    pair_counts[pair] += weight
        if not pair_counts:
            break
        pair, count = max(pair_counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
        new_id = len(token_bytes)
        token_bytes.append(token_bytes[pair[0]] + token_bytes[pair[1]])
        merges.append(pair)
        sequences = {
            segment: replace_pair(seq, pair, new_id)
            for segment, seq in sequences.items()
        }
        if new_id % 32 == 0 or new_id + 1 == vocab_size:
            print(f"merge {new_id}: pair={pair} count={count}")

    return tokenizer.ByteBPETokenizer([b.hex() for b in token_bytes], merges)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--type", choices=["byte", "hybrid", "bpe"], required=True)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--max_piece_bytes", type=int, default=24)
    parser.add_argument("--out", default="tokenizer.json")
    args = parser.parse_args()

    text = open(args.data, encoding="utf-8").read()
    if args.type == "byte":
        tok = tokenizer.ByteTokenizer()
    elif args.type == "hybrid":
        tok = build_hybrid(text)
    else:
        tok = build_bpe(text, args.vocab_size, args.max_piece_bytes)

    ids = tok.encode(text)
    assert tok.decode(ids) == text
    tok.save(args.out)
    print(
        f"saved {args.out}: type={args.type} vocab={tok.vocab_size} "
        f"bytes={len(text.encode('utf-8'))} tokens={len(ids)}"
    )


if __name__ == "__main__":
    main()
