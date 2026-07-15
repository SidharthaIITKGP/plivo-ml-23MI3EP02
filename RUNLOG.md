# RUNLOG

| Run | Steps | Main change | Params | Dev bpb | Delta vs target | Runtime | Decision |
|---|---:|---|---:|---:|---:|---:|---|
| 00 | 2000 | Untouched starter baseline | 1,339,840 | 2.3718 | - | 89s | Reference |
| S00 | 600 | Untouched 600-step baseline | 1,339,840 | 2.9308 | - | 35s | Screening reference |
| S01 | 600 | Warmup + cosine, LR 3e-4 | 1,339,840 | 2.9698 | -0.0390 vs S00 | 31s | Reject |
| S02 | 600 | Warmup + cosine, LR 6e-4 | 1,339,840 | 2.9220 | +0.0088 vs S00 | 27s | Keep direction |
| S03 | 600 | Warmup + cosine, LR 1e-3 | 1,339,840 | 2.9056 | +0.0164 vs S02 | 27s | Keep |
| S04 | 600 | Batch 16 | 1,339,840 | 2.8182 | +0.0874 vs S03 | 48s | Keep |
| S05 | 600 | AdamW, beta2 0.95, wd 0.1, clip 1.0 | 1,339,840 | 2.7299 | +0.0883 vs S04 | 43s | Keep |
| S06 | 600 | Weight tying at 160 width | 1,298,880 | 2.7770 | -0.0471 vs S05 | 40s | Reject for 160-wide path |
| S07 | 600 | GPT-style scaled init | 1,339,840 | 2.7613 | -0.0314 vs S05 | 40s | Reject |
| S08 | 600 | Hybrid Devanagari-byte tokenizer | 1,367,680 | 2.5987 | +0.1312 vs S05 | 41s | Strong fallback |
| S10 | 600 | Hybrid tokenizer + 4x192 no-tie | 1,936,128 | 2.5804 | +0.0183 vs S08 | 52s | Hybrid champion |
| S09 | 600 | 512-token byte BPE, 160-wide no-tie | 1,421,760 | 2.3751 | +0.2236 vs S08 | 42s | Keep BPE |
| S11 | 600 | BPE + 4x192 tied | 1,902,720 | 2.3901 | -0.0150 vs S09 | 50s | Reject tied path |
| S12 | 600 | BPE + 4x188 no-tie | 1,923,240 | 2.3706 | +0.0045 vs S09 | 50s | Keep no-tie width |
| S13 | 600 | BPE 4x188 no-tie, batch 32 | 1,923,240 | 2.2824 | +0.0882 vs S12 | 102s | Keep |
| S14 | 600 | BPE 4x188 batch 32, LR 1.5e-3 | 1,923,240 | 2.1593 | +0.1231 vs S13 | 111s | Keep |
| S15 | 600 | BPE 4x188 batch 32, LR 2e-3 | 1,923,240 | 2.1691 | -0.0098 vs S14 | 105s | Reject |
| F01 | 2000 | Final BPE 4x188 batch 32, LR 1.5e-3 | 1,923,240 | 1.7621 | +0.6097 vs full baseline | 351s | Final |
| F02 | 2000 | F01 recipe with EMA 0.99 | 1,923,240 | 1.7543 | +0.6175 vs full baseline | 372s | Final selected |
| S16 | 600 | BPE 5 layers x width 172 | 1,984,708 | 2.2377 | -0.0784 vs S14 | 120s | Reject |
| S17 | 600 | BPE 6 layers x width 156 | 1,944,384 | 2.2700 | -0.1107 vs S14 | 134s | Reject |
| S18 | 600 | BPE 4x188, batch 48 | 1,923,240 | 2.0407 | +0.1186 vs S14 | 170s | Promising, not promoted due deadline |
| S19 | 600 | BPE 4x188, batch 64 | 1,923,240 | 1.9869 | +0.1724 vs S14 | 243s | Promising, not promoted due deadline |

## Run 00 - Untouched starter baseline

### Hypothesis
The provided starter is intentionally mediocre but provides the required full-run reference.

### Exact command
```bash
python3 train.py --data ../data/train_corpus.txt --steps 2000 --out ../runs/run_00_baseline/ckpt.pt --log_every 200
python3 evaluate.py --checkpoint ../runs/run_00_baseline/ckpt.pt --text_file ../data/dev_eval.txt
```

### Results
The baseline used raw byte tokenization, 1,339,840 parameters, and reached 2.3718 dev bpb after exactly 2,000 steps.

### Conclusion
This established the score to beat and confirmed the starter interface worked.

## Run S00 - Untouched 600-step screening control

### Hypothesis
A 600-step baseline provides a fair comparison point for short experiments.

### Results
The 600-step starter reached 2.9308 dev bpb.

### Conclusion
Short screens were compared against S00 or the current 600-step champion, not against the full baseline.

## Runs S01-S03 - Learning-rate schedule search

### Hypothesis
Warmup plus cosine decay may improve optimization under the fixed step budget, but the peak LR must be high enough for a 600-step screen.

### Results
LR 3e-4 worsened bpb to 2.9698, LR 6e-4 improved to 2.9220, and LR 1e-3 improved to 2.9056.

### Conclusion
The schedule family was useful only after raising peak LR; 1e-3 became the next champion.

## Run S04 - Larger batch

### Hypothesis
Under a step cap, a larger batch increases token exposure per optimizer step and should improve bpb if CPU time remains acceptable.

### Results
Batch 16 improved dev bpb from 2.9056 to 2.8182.

### Conclusion
Batch 16 was kept because the improvement was large and runtime remained practical.

## Run S05 - AdamW recipe

### Hypothesis
AdamW, beta2 0.95, weight decay, and gradient clipping should train more stably than plain Adam.

### Results
The combined optimizer recipe improved dev bpb from 2.8182 to 2.7299.

### Conclusion
The recipe was retained. Because time was limited, this was treated as a coupled optimizer experiment rather than four isolated runs.

## Run S06 - Weight tying at 160 width

### Hypothesis
Tying input and output embeddings may regularize the model and save parameters.

### Results
At 160 width, weight tying reduced parameters but worsened dev bpb from 2.7299 to 2.7770.

### Conclusion
Weight tying was rejected for the main no-tie path, but later tested where vocabulary and parameter limits forced the tradeoff.

## Run S07 - GPT-style initialization

### Hypothesis
Smaller GPT-style initialization with scaled residual projections may improve stability.

### Results
The run scored 2.7613 bpb, worse than S05.

### Conclusion
Baseline initialization was kept; in this short CPU training regime, the smaller initialization appeared to slow useful learning.

## Run S08 - Hybrid Devanagari-byte tokenizer

### Hypothesis
Representing train-observed Devanagari code points as single tokens should improve bilingual modeling while preserving byte fallback.

### Results
The tokenizer reduced train tokens from 7,318,592 to 5,714,900 and dev tokens from 159,225 to 113,073. Dev bpb improved to 2.5987.

### Conclusion
This became the safe tokenizer fallback because it was simple, lossless, and clearly beneficial.

## Run S10 - Hybrid tokenizer with 4x192 model

### Hypothesis
The hybrid tokenizer leaves enough parameter budget for a wider no-tie model.

### Results
A 4-layer, 192-wide model with 1,936,128 parameters improved hybrid dev bpb to 2.5804.

### Conclusion
This became the safe fallback system if BPE failed.

## Run S09 - 512-token byte BPE

### Hypothesis
A deterministic byte-level BPE trained only on train_corpus.txt should reduce sequence length more than character hybrid tokenization while remaining lossless.

### Results
The BPE tokenizer reduced train tokens to 3,873,809 and dev tokens to 79,358. With the 160-wide model it reached 2.3751 bpb.

### Conclusion
BPE was a major win and replaced the hybrid tokenizer as the primary path.

## Runs S11-S12 - BPE capacity and tying comparison

### Hypothesis
Using more of the parameter budget should improve BPE modeling, but tying may hurt because it previously lost at 160 width.

### Results
BPE 4x192 tied scored 2.3901, worse than the smaller no-tie BPE model. BPE 4x188 no-tie stayed under the cap at 1,923,240 params and improved slightly to 2.3706.

### Conclusion
The final architecture kept untied embeddings and used width 188, the largest tested no-tie width safely below the cap.

## Runs S13-S15 - Batch and LR refinement

### Hypothesis
BPE plus a wider model may benefit from larger batch and a higher LR than the byte-token setup.

### Results
Batch 32 improved bpb to 2.2824. Raising LR to 1.5e-3 improved bpb further to 2.1593. Raising LR again to 2e-3 worsened bpb to 2.1691.

### Conclusion
Batch 32 and LR 1.5e-3 were selected for the final full run.

## Run F01 - Final 2,000-step checkpoint

### Hypothesis
The best 600-step recipe should improve substantially when trained for the full allowed 2,000 optimizer steps with a 100-step warmup.

### Exact command
```bash
../../env/bin/python train.py --data ../data/train_corpus.txt --steps 2000 --batch 32 --out ../runs/run_f01_bpe512_4x188_batch32_lr15e4/ckpt.pt --log_every 200 --lr 1.5e-3 --min_lr 1.5e-4 --warmup_steps 100 --optimizer adamw --beta2 0.95 --weight_decay 0.1 --grad_clip 1.0 --n_embd 188 --n_head 4 --n_layer 4
../../env/bin/python evaluate.py --checkpoint ../runs/run_f01_bpe512_4x188_batch32_lr15e4/ckpt.pt --text_file ../data/dev_eval.txt
```

### Results
The final model used 1,923,240 parameters, exactly 2,000 optimizer steps, and scored 1.7621 dev bpb. Diagnostics were 1.5175 bpb on the first dev half, 2.0425 on the second half, 1.1123 on Hindi-containing paragraphs, and 2.3746 on non-Hindi paragraphs.

### Conclusion
This checkpoint was selected as final because it had the best measured dev bpb, remained under every hard cap, preserved arbitrary UTF-8 byte fallback, and passed the exact evaluator interface.

## Run F02 - Late checkpoint averaging and EMA

### Hypothesis
Late weight averaging or EMA could improve generalization without increasing parameters, using extra optimizer steps, or changing the evaluator interface.

### Exact controlled change
The F01 recipe was rerun with checkpoints saved at steps 1600, 1700, 1800, 1900, and 2000, plus EMA checkpoints with decays 0.99, 0.995, and 0.999 starting after step 400. The model architecture, tokenizer, seed, batch size, optimizer, learning-rate schedule, and 2,000-step budget were unchanged.

### Command
```bash
../../env/bin/python train.py --data ../data/train_corpus.txt --steps 2000 --batch 32 --out ../runs/run_f02_avg_ema/ckpt.pt --log_every 200 --lr 1.5e-3 --min_lr 1.5e-4 --warmup_steps 100 --optimizer adamw --beta2 0.95 --weight_decay 0.1 --grad_clip 1.0 --n_embd 188 --n_head 4 --n_layer 4 --save_dir ../runs/run_f02_avg_ema/checkpoints --save_steps 1600,1700,1800,1900,2000 --ema_decays 0.99,0.995,0.999 --ema_start_step 400
```

### Results
Raw step 2000 scored 1.7621 bpb. Averaging steps 1800, 1900, and 2000 scored 1.7566 bpb. EMA 0.99 scored 1.7543 bpb, EMA 0.995 scored 1.7557 bpb, and EMA 0.999 scored 1.8954 bpb.

### Conclusion
EMA 0.99 was selected as the final checkpoint because it improved dev bpb from 1.7621 to 1.7543 while preserving the same parameter count, step count, tokenizer, and official interface.

## Runs S16-S17 - Depth-versus-width allocation

### Hypothesis
Deeper models might use the parameter budget better than the 4-layer, width-188 model.

### Results
The largest valid 5-layer candidate was width 172 with 1,984,708 parameters and scored 2.2377 bpb at 600 steps. The largest valid 6-layer candidate was width 156 with 1,944,384 parameters and scored 2.2700 bpb at 600 steps.

### Conclusion
Both deeper models were slower and worse than the 4x188 control, so depth was rejected.

## Runs S18-S19 - Larger effective batch

### Hypothesis
Because optimizer steps are capped, increasing batch size could improve bpb by exposing the model to more tokens per update.

### Results
Batch 48 scored 2.0407 bpb at 600 steps, and batch 64 scored 1.9869 bpb at 600 steps, both better than the batch-32 600-step control of 2.1593 bpb.

### Conclusion
Batch 64 was the most promising unfinished direction, but it was not promoted to a final 2,000-step checkpoint because the GitHub deadline made the verified EMA 0.99 checkpoint the safer submission.
