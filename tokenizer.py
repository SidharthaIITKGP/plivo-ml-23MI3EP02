"""Lossless tokenizer implementations with byte fallback."""
import json
import re
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("tokenizer.json")
SPLIT_PATTERN = r"\s+|[^\s]+"


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "byte", "vocab_size": 256}, f)


class HybridCharByteTokenizer:
    def __init__(self, characters):
        self.characters = list(characters)
        self.char_to_id = {ch: 256 + i for i, ch in enumerate(self.characters)}
        self.id_to_bytes = [bytes([i]) for i in range(256)]
        self.id_to_bytes.extend(ch.encode("utf-8") for ch in self.characters)
        self.vocab_size = len(self.id_to_bytes)

    def encode(self, text):
        ids = []
        for ch in text:
            token_id = self.char_to_id.get(ch)
            if token_id is None:
                ids.extend(ch.encode("utf-8"))
            else:
                ids.append(token_id)
        return ids

    def decode(self, ids):
        return b"".join(self.id_to_bytes[i] for i in ids).decode("utf-8")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "hybrid_char_byte",
                    "characters": self.characters,
                    "vocab_size": self.vocab_size,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


class ByteBPETokenizer:
    def __init__(self, token_bytes, merges):
        self.id_to_bytes = [bytes.fromhex(item) for item in token_bytes]
        self.merges = [tuple(pair) for pair in merges]
        self.vocab_size = len(self.id_to_bytes)
        self.cache = {}

    def _encode_segment(self, segment):
        cached = self.cache.get(segment)
        if cached is not None:
            return cached
        ids = tuple(segment.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            new_id = 256 + rank
            out = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = tuple(out)
        self.cache[segment] = list(ids)
        return self.cache[segment]

    def encode(self, text):
        ids = []
        for segment in re.findall(SPLIT_PATTERN, text):
            ids.extend(self._encode_segment(segment))
        return ids

    def decode(self, ids):
        return b"".join(self.id_to_bytes[i] for i in ids).decode("utf-8")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "byte_bpe",
                    "vocab_size": self.vocab_size,
                    "token_bytes_hex": [b.hex() for b in self.id_to_bytes],
                    "merges": [list(pair) for pair in self.merges],
                    "split_pattern": SPLIT_PATTERN,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


def load(path=None):
    path = DEFAULT_PATH if path is None else Path(path)
    if not path.exists():
        return ByteTokenizer()
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    tok_type = spec.get("type")
    if tok_type == "byte":
        return ByteTokenizer()
    if tok_type == "hybrid_char_byte":
        return HybridCharByteTokenizer(spec["characters"])
    if tok_type == "byte_bpe":
        return ByteBPETokenizer(spec["token_bytes_hex"], spec["merges"])
    raise ValueError(f"unknown tokenizer type: {tok_type}")
