import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChameleonConfig:
    vocab_size: int = 101045  # img tokens + text tokens, check enc.n_vocab in prepare_mini_coco.py
    n_layers: int = 12
    n_heads: int = 12
    dim: int = 768
    max_seq_len: int = 512
    eps: float = 1e-5
    dropout: float = 0.0  # 7b chameleon has 0.1, 34b has 0.0
    swin_norm: bool = False
    qk_norm: bool = True
    qk_norm_eps: float = 1e-5


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, config: ChameleonConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.dim // config.n_heads

        self.wq = nn.Linear(
            config.dim,
            config.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            config.dim,
            config.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            config.dim,
            config.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            config.n_heads * self.head_dim,
            config.dim,
            bias=False,
        )

        if self.config.qk_norm:
            self.q_ln = nn.LayerNorm(self.head_dim, config.qk_norm_eps)
            self.k_ln = nn.LayerNorm(self.head_dim, config.qk_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.config.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.config.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.config.n_heads, self.head_dim)

        if self.config.qk_norm:
            xq, xk = self.q_ln(xq), self.k_ln(xk)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        scores = torch.matmul(xq.transpose(1, 2), xk.transpose(1, 2).transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv.transpose(1, 2))
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FFN(nn.Module):
    def __init__(
        self,
        config: ChameleonConfig,
    ):
        super().__init__()
        hidden_dim = int(2 * config.dim / 3)
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)

        self.w1 = nn.Linear(
            config.dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, config.dim, bias=False
        )
        self.w3 = nn.Linear(
            config.dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config: ChameleonConfig):
        super().__init__()
        self.config = config

        self.attn = Attention(config)
        if config.dropout > 0.0:
            self.dropout_1 = nn.Dropout(config.dropout)
            self.dropout_2 = nn.Dropout(config.dropout)

        self.ln_1 = RMSNorm(config.dim, eps=config.eps)
        self.ln_2 = RMSNorm(config.dim, eps=config.eps)
        self.ffn = FFN(config)

    def forward(self, x, freqs_cis, mask=None):
        if self.config.swin_norm:
            x = x + self.ln_1(self.dropout_1(self.attn(x, freqs_cis=freqs_cis, mask=mask)))
            x = x + self.ln_2(self.dropout_2(self.ffn(x)))
        else:
            x = x + self.dropout_1(self.attn(self.ln_1(x), freqs_cis=freqs_cis, mask=mask))
            x = x + self.dropout_2(self.ffn(self.ln_2(x)))
        return x


class ChameleonModel(torch.nn.Module):
    def __init__(self, config: ChameleonConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.dim)
        self.decoder = nn.ModuleList(
            [Block(config) for _ in range(config.n_layers)]
        )
        self.ln_f = RMSNorm(config.dim, eps=config.eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(self.config.dim // self.config.n_heads, self.config.max_seq_len * 2, theta=10000)

    def forward(self, input_ids, labels=None):
        batch_size, sequence_length = input_ids.size()

        x = self.wte(input_ids)

        freqs_cis = self.freqs_cis[:sequence_length]

        causal_attention_mask = torch.triu(
            torch.ones((self.config.max_seq_len, self.config.max_seq_len), dtype=x.dtype, device=x.device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        causal_attention_mask *= torch.finfo(causal_attention_mask.dtype).min

        for l in self.decoder:
            x = l(x, freqs_cis=freqs_cis, mask=causal_attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits

        # z-loss regularization
        z_loss = 1e-5 * torch.logsumexp(logits, dim=-1).pow(2).mean()

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(
            logits.view(-1, self.config.vocab_size), labels.view(-1).long()
        )
        loss = loss.sum() / (batch_size * sequence_length)
        loss = loss.to(logits.dtype)

        # add z_loss to loss
        loss += z_loss

        return loss