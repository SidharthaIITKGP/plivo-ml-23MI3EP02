1. The final model is a 4-layer, 188-dimensional causal Transformer with 4 attention heads and a context length of 128.
2. It contains 1,923,240 parameters and was trained for exactly 2,000 optimizer steps on only the provided training corpus.
3. The tokenizer is a lossless 512-token byte-level BPE tokenizer trained only on train_corpus.txt with complete byte fallback for arbitrary UTF-8 input.
4. Compared with raw bytes, it reduced the development sequence from 159,225 to 79,358 tokens while preserving exact round trips.
5. Training used AdamW, peak learning rate 1.5e-3, 100 warmup steps, cosine decay to 1.5e-4, batch size 32, and gradient clipping at 1.0.
6. Controlled runs showed that larger batch size, AdamW, byte-level BPE, and no-tie width 188 improved bpb, while weight tying, GPT-style initialization, and LR 2e-3 did not help.
7. The final development score was 1.7543 bpb, improving by 0.6175 bpb over the untouched 2,000-step baseline.
8. Late-checkpoint averaging improved the raw 2,000-step score, with EMA decay 0.99 selected over raw weights, simple checkpoint averages, and slower deeper models.
9. The submitted evaluator, checkpoint, tokenizer, and code pass the step, parameter, arbitrary-UTF-8, and lossless-round-trip checks.
