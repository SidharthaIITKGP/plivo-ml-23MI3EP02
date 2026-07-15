"""Small decoder-only language model for the speedrun assignment."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self):
        self.vocab_size = 256
        self.block_size = 128
        self.n_layer = 4
        self.n_head = 4
        self.n_embd = 160
        self.dropout = 0.0
        self.tie_weights = False
        self.mlp_type = "gelu"
        self.init_type = "baseline"


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj.SCALE_INIT = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class GeluMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.proj.SCALE_INIT = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = max(1, ((hidden + 31) // 32) * 32)
        self.in_proj = nn.Linear(cfg.n_embd, 2 * hidden)
        self.out_proj = nn.Linear(hidden, cfg.n_embd)
        self.out_proj.SCALE_INIT = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.drop(self.out_proj(F.silu(gate) * value))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        if cfg.mlp_type == "gelu":
            self.mlp = GeluMLP(cfg)
        elif cfg.mlp_type == "swiglu":
            self.mlp = SwiGLU(cfg)
        else:
            raise ValueError(f"unknown mlp_type: {cfg.mlp_type}")

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

    def _init(self, module):
        if isinstance(module, nn.Linear):
            if self.cfg.init_type == "gpt":
                std = 0.02
                if getattr(module, "SCALE_INIT", False):
                    std *= (2 * self.cfg.n_layer) ** -0.5
            elif self.cfg.init_type == "baseline":
                std = 0.05
            else:
                raise ValueError(f"unknown init_type: {self.cfg.init_type}")
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02 if self.cfg.init_type == "gpt" else 0.05
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
