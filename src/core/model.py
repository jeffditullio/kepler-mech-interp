"""
Small transformer for Kepler regression: discrete-digit (M, e) inputs -> scalar E.

Hand-rolled (not nn.Transformer) because the interp tools hook into the
residual stream, attention patterns, and MLP activations. Standard pre-LN block: each
sublayer is `x = x + sublayer(LN(x))`.

Architecture:
  input: token IDs (B, L=2d+1) over vocab {0..9, ANS=10}  -- [M(d), e(d), ANS]
  body:  embed -> N x (attn + MLP) -> final LN
  head:  linear(d_model, 1) read off at the last position, sigmoid -> [0, 1)

The "algorithm" lives in the body. The output head is a single direction in
residual space along which E is read.
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.core.config import VOCAB_SIZE, Config


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, d_head)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, L, d_head)
        # Causal mask is folded into scaled_dot_product_attention via is_causal.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, L, d_head)
        return self.out(out.transpose(1, 2).reshape(B, L, D))


_ACTIVATIONS = {"gelu": F.gelu, "relu": F.relu}


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.fc2 = nn.Linear(cfg.d_mlp, cfg.d_model)
        # Resolved at construction so forward() stays a simple call. Old
        # checkpoints without an activation field default to gelu via
        # Config's dataclass default.
        self.act = _ACTIVATIONS[cfg.activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class KeplerTransformer(nn.Module):
    """
    Discrete-digit inputs -> scalar regression output in [0, 1).

    Sequence layout: [M_digits, e_digits, ANS] -- length 2*n_digits + 1.
    Output is read off the residual stream at the final position (the ANS token).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.seq_len = 2 * cfg.n_digits + 1  # [M_d, e_d, ANS]
        self.tok_emb = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.pos_emb = nn.Embedding(self.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # final_ln=False -> Identity: exactly-linear readout (see Config).
        # getattr: configs saved before the field existed default to True.
        self.ln_f = nn.LayerNorm(cfg.d_model) if getattr(cfg, "final_ln", True) else nn.Identity()
        self.head = nn.Linear(cfg.d_model, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, 2d+1) int64
        returns: (B,) float in (0, 1) -- predicted normalized E
        """
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # (B, L, d_model)
        # Read off the regression scalar at the final position (the ANS token).
        # This is the "E direction" in residual space; the interp tools ask
        # how that direction is constructed layer by layer.
        z = self.head(x[:, -1, :]).squeeze(-1)  # (B,) the logit
        # Output map to normalized E in [0,1). Sweep-able: sigmoid (default,
        # bounded), tanh (bounded, rescaled to [0,1)), linear (unbounded --
        # tests whether the bounded readout nonlinearity is load-bearing).
        act = getattr(self.cfg, "out_activation", "sigmoid")
        if act == "sigmoid":
            return torch.sigmoid(z)
        if act == "tanh":
            return 0.5 * (torch.tanh(z) + 1.0)
        if act == "clamp":
            # Bounded like sigmoid/tanh but PIECEWISE-LINEAR: identity in the
            # interior (no bulk amplification), hard-clipped at the [0,1] edges
            # (cusp control). Separates "bounding helps" from "smooth warp hurts".
            # CENTERED (z+0.5) so a zero-init logit lands at 0.5 (mid-range, live
            # gradient) like sigmoid(0)=0.5 / tanh-map(0)=0.5; the raw z.clamp had
            # a DEAD GRADIENT at init (z<0 -> clipped to 0 -> stuck, train_loss~1/3).
            return (z + 0.5).clamp(0.0, 1.0)
        if act in ("linear", "none"):
            return z
        raise ValueError(f"unknown out_activation {act!r}")
