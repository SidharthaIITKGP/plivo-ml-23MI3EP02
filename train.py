"""Configurable CPU trainer for the 2,000-step LLM speedrun."""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    starts = torch.randint(0, len(ids) - block - 1, (batch,), device=ids.device)
    offsets = torch.arange(block + 1, device=ids.device)
    seq = ids[starts[:, None] + offsets[None, :]]
    return seq[:, :-1].to(device), seq[:, 1:].to(device)


def get_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def build_config(args, vocab_size):
    cfg = Config()
    cfg.vocab_size = vocab_size
    cfg.block_size = args.block_size
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.n_embd = args.n_embd
    cfg.dropout = args.dropout
    cfg.tie_weights = args.tie_weights
    cfg.mlp_type = args.mlp_type
    cfg.init_type = args.init_type
    return cfg


def public_config(cfg):
    return {
        k: getattr(cfg, k)
        for k in dir(cfg)
        if not k.startswith("_") and not callable(getattr(cfg, k))
    }


def build_optimizer(model, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    decay_params = []
    no_decay_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
    )


def parse_int_list(value):
    if not value:
        return []
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    if not value:
        return []
    return [float(item) for item in value.split(",") if item.strip()]


def clone_model_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def update_ema_state(ema_state, model, decay):
    model_state = model.state_dict()
    for key, value in model_state.items():
        cpu_value = value.detach().cpu()
        if torch.is_floating_point(cpu_value):
            ema_state[key].mul_(decay).add_(cpu_value, alpha=1.0 - decay)
        else:
            ema_state[key] = cpu_value.clone()


def save_checkpoint(path, model_state, cfg, args, losses, lrs, grad_norms, n_params, step):
    ckpt = {
        "model": model_state,
        "config": public_config(cfg),
        "steps": step,
        "train_loss_curve": list(losses),
        "lr_curve": list(lrs),
        "grad_norm_curve": list(grad_norms),
        "train_args": vars(args),
        "n_params": n_params,
        "tokens_seen": step * args.batch * cfg.block_size * args.grad_accum,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tie_weights", action="store_true")
    parser.add_argument("--mlp_type", choices=["gelu", "swiglu"], default="gelu")
    parser.add_argument("--init_type", choices=["baseline", "gpt"], default="baseline")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default="ckpt.pt")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_steps", default="")
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--ema_decays", default="")
    parser.add_argument("--ema_start_step", type=int, default=0)
    args = parser.parse_args()

    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    assert args.grad_accum >= 1
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(
        f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
        f"(vocab {tok.vocab_size})"
    )
    assert len(ids) > args.block_size + 1

    cfg = build_config(args, tok.vocab_size)
    model = GPT(cfg).to(device)
    n_params = model.n_params()
    print(f"model: {n_params:,} params")
    assert n_params <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    opt = build_optimizer(model, args)
    min_lr = args.lr if args.min_lr is None else args.min_lr

    model.train()
    save_steps = set(parse_int_list(args.save_steps))
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    ema_decays = parse_float_list(args.ema_decays)
    ema_states = {
        decay: clone_model_state(model)
        for decay in ema_decays
    }
    t0 = time.time()
    losses = []
    lrs = []
    grad_norms = []
    for step in range(1, args.steps + 1):
        lr = get_lr(step - 1, args.steps, args.warmup_steps, args.lr, min_lr)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(ids, cfg.block_size, args.batch, device)
            _, loss = model(x, y)
            step_loss += loss.item()
            (loss / args.grad_accum).backward()

        grad_norm = None
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm = float(grad_norm)
        opt.step()
        if ema_states and step >= args.ema_start_step:
            for decay, ema_state in ema_states.items():
                update_ema_state(ema_state, model, decay)

        step_loss /= args.grad_accum
        losses.append(step_loss)
        lrs.append(lr)
        if grad_norm is not None:
            grad_norms.append(grad_norm)
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            msg = (
                f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  "
                f"({(time.time() - t0) / step * 1000:.0f} ms/step)"
            )
            if grad_norm is not None:
                msg += f"  grad {grad_norm:.2f}"
            print(msg)
        if save_dir is not None and step in save_steps:
            path = save_dir / f"step_{step}.pt"
            save_checkpoint(
                path,
                clone_model_state(model),
                cfg,
                args,
                losses,
                lrs,
                grad_norms,
                n_params,
                step,
            )
            print(json.dumps({"saved_step_checkpoint": str(path), "step": step}))

    save_checkpoint(
        args.out,
        model.state_dict(),
        cfg,
        args,
        losses,
        lrs,
        grad_norms,
        n_params,
        args.steps,
    )
    if save_dir is not None:
        for decay, ema_state in ema_states.items():
            path = save_dir / f"ema_{decay:g}.pt"
            save_checkpoint(
                path,
                ema_state,
                cfg,
                args,
                losses,
                lrs,
                grad_norms,
                n_params,
                args.steps,
            )
            print(json.dumps({"saved_ema_checkpoint": str(path), "decay": decay}))
    print(json.dumps({"saved": args.out, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
