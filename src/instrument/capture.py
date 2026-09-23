"""
Model instrumentation: the shared machinery for running the model with hooks.

This is the ONLY layer where forward passes meet hooks and surgery. The interp
tools (src/analysis) define WHAT to capture; the pure math they feed lives in
src/kernels. Primitives here:

  - w_eff            : the effective readout direction (the readout projection's direction)
  - token_batches    : the batched-forward loop every hooked capture shares
  - ans_embedding    : embedding direct path at the readout (ANS) token
  - write_projection_hook : project a component's readout-position write
  - raw_write_hook / ln_exact_terms : a component's raw readout-position
                         write, and the exact per-component logit split
                         through the final LayerNorm (centered w_eff over the
                         per-input scale), which sums to the logit
  - recompute_qkv    : per-head q/k/v from a module input (SDPA hides them)
  - head_ov          : one head's OV weight slices
  - patched_attn_forward / patched_mlp_forward : ablation surgery
  - mean_head_outputs / mean_mlp_outputs       : grid-means for mean-ablation
  - head_writes        : per-head residual writes at the readout position
  - run_write_patched  : forward with a per-row delta subtracted from one
                         layer's attention write at the readout position
  - run_write_interchanged : differentiable forward with the write's coordinate
                         along a direction replaced by a source input's
                         coordinate (the DAS interchange intervention)
  - mean_neuron_acts / run_neurons_ablated : mean-ablation of a SET of one
                         layer's MLP neurons at the readout position
  - neuron_acts / neuron_readout_coefs : per-input readout-position MLP
                         activations + the weight-side readout coefficients
                         (the |c|*sigma contribution score's two factors)
"""

from contextlib import ExitStack, contextmanager

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def _with_hook(module, fn, *, pre=False):
    """Register a forward hook (pre=True: pre-hook) on module, yield, remove it."""
    handle = module.register_forward_pre_hook(fn) if pre else module.register_forward_hook(fn)
    try:
        yield
    finally:
        handle.remove()


def w_eff(model):
    """Effective readout direction w_eff = head.weight ∘ ln_f.gain. Projecting
    a write on it is the READOUT PROJECTION: the write's content along the
    readout direction before the final LayerNorm's mean-subtraction and shared
    per-input 1/sigma. Direct logit attribution keeps both (ln_exact_terms).
    With the final LN off (ln_f = Identity, Config.final_ln=False) the two
    coincide and w_eff is exact."""
    if isinstance(model.ln_f, torch.nn.Identity):
        return model.head.weight[0].detach()
    return (model.head.weight[0] * model.ln_f.weight).detach()


def token_batches(inputs, device, batch=4096):
    """Yield int64 token batches (numpy (N, L) -> torch on device); clears the
    MPS cache between batches to bound peak memory. The hooked-capture
    counterpart of core.runs.run_model (which owns the no-hook loop)."""
    for i in range(0, inputs.shape[0], batch):
        yield torch.from_numpy(inputs[i : i + batch]).to(device)
        if device == "mps":
            torch.mps.empty_cache()


def ans_embedding(model, tok):
    """Embedding direct path at the readout (ANS) token: tok_emb + pos_emb.
    Constant over the grid by construction (the readout position always holds
    ANS) -- a built-in sanity check for attribution."""
    return model.tok_emb(tok)[:, -1, :] + model.pos_emb.weight[-1]


def write_projection_hook(store, name, w):
    """Forward hook: project the component's residual write at the readout
    position onto direction w; append numpy chunks to store[name]."""

    def hook(_m, _inp, out):
        store.setdefault(name, []).append((out[:, -1, :] @ w).cpu().numpy())

    return hook


def raw_write_hook(store, name):
    """Forward hook: append the component's raw residual write at the readout
    position, (B, d_model) numpy, to store[name]."""

    def hook(_m, _inp, out):
        store.setdefault(name, []).append(out[:, -1, :].cpu().numpy())

    return hook


def ln_exact_terms(model, comps):
    """The exact per-component split of the logit through the final LayerNorm.
    With x the sum of the raw writes at the readout position,

        logit = sum_c (w_c . c) / sigma(x) + const,

    w_c = w_eff - mean(w_eff) (LN's mean subtraction folded into the readout),
    sigma(x) the per-input LN std, const = head.bias + head.weight . ln_f.bias.
    This is direct logit attribution with the cached scale kept in; the readout
    projection w_eff . c alone drops both the centering and the 1/sigma. With the
    final LN off the identity is w_eff . c + head.bias. Returns
    (terms {name: (N,)}, const, sigma (N,)); sum(terms) + const == logit."""
    w = w_eff(model).cpu().numpy()
    x = sum(comps.values())
    n = x.shape[0]
    if isinstance(model.ln_f, torch.nn.Identity):
        w_c, sigma, const = w, np.ones(n), float(model.head.bias.item())
    else:
        w_c = w - w.mean()
        mu = x.mean(axis=1, keepdims=True)
        sigma = np.sqrt(((x - mu) ** 2).mean(axis=1) + model.ln_f.eps)
        const = float(model.head.bias.item() + (model.head.weight[0] @ model.ln_f.bias).item())
    return {k: (c @ w_c) / sigma for k, c in comps.items()}, const, sigma


def recompute_qkv(attn, x, nh, dh):
    """Per-head q/k/v stack recomputed from an attention module's INPUT -- the
    fused scaled_dot_product_attention never materializes them. Returns
    (B, L, 3, nh, dh); q, k, v = result.unbind(dim=2)."""
    B, L, _ = x.shape
    return (x @ attn.qkv.weight.t()).reshape(B, L, 3, nh, dh)


def head_ov(attn, h, d_model, dh):
    """Head h's OV-circuit weights: (W_V_h rows (dh, D), W_O_h columns (D, dh))."""
    Wqkv = attn.qkv.weight.detach()  # (3D, D)
    Wo = attn.out.weight.detach()  # (D, D)
    return Wqkv[2 * d_model + h * dh : 2 * d_model + (h + 1) * dh, :], Wo[:, h * dh : (h + 1) * dh]


def transfer_curve(model, layer, h):
    """Head h's digit->logit transfer curve: w_eff . OV_h(tok_emb(d)) for
    d = 0..9. Weights-only (no attention weights, no LayerNorm); linear, so a
    position embedding would only add a constant -- one curve per head serves
    every source field. Returns a (10,) numpy array."""
    d_model = model.tok_emb.weight.shape[1]
    attn = model.blocks[layer].attn
    dh = d_model // attn.n_heads
    Wv_h, Wo_h = head_ov(attn, h, d_model, dh)
    w = w_eff(model)
    tok = model.tok_emb.weight.detach()
    return np.array([float(w @ ((tok[d] @ Wv_h.t()) @ Wo_h.t())) for d in range(10)])


# ----------------------------------------------------------------------
# Ablation surgery (patched forwards + on-distribution grid means)
# ----------------------------------------------------------------------


def patched_attn_forward(kill_heads, mean_out=None):
    """Ablate head(s): zero their output (mean_out=None) or replace with the
    grid-mean (mean_out: (nh, L, dh)). kill_heads is an int or iterable."""
    kill = [kill_heads] if isinstance(kill_heads, int) else list(kill_heads)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, nh, L, dh)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True).clone()
        for h in kill:
            out[:, h] = torch.zeros(self.d_head, device=x.device) if mean_out is None else mean_out[h]
        return self.out(out.transpose(1, 2).reshape(B, L, D))

    return forward


def patched_mlp_forward(mean_out=None):
    """Ablate a layer's MLP: zero its residual write (mean_out=None) or replace
    with the grid-mean (mean_out: (L, d_model))."""

    def forward(self, x):
        B, L, D = x.shape
        if mean_out is None:
            return torch.zeros_like(x)
        return mean_out[None].expand(B, L, D)

    return forward


@torch.no_grad()
def mean_mlp_outputs(model, cfg, inputs, device):
    """Grid-mean of each layer's MLP residual write, {layer: (L, d_model)}."""
    L = model.seq_len
    means = {b: torch.zeros(L, cfg.d_model, device=device) for b in range(cfg.n_layers)}
    cap, n = {}, 0
    with ExitStack() as stack:
        for b, blk in enumerate(model.blocks):
            stack.enter_context(_with_hook(blk.mlp, lambda _m, _i, out, b=b: cap.__setitem__(b, out)))
        for tok in token_batches(inputs, device):
            model(tok)
            for b in range(cfg.n_layers):
                means[b] += cap[b].sum(dim=0)
            n += cap[0].shape[0]
    return {b: means[b] / n for b in means}


@torch.no_grad()
def head_writes(model, layer, inputs, device):
    """Per-head residual writes at the readout position for one layer's
    attention: [(N, d_model)] per head. Their sum equals the layer's attn
    write at that position (the module output the residual stream receives)."""
    attn = model.blocks[layer].attn
    nh, dh = attn.n_heads, attn.d_head
    chunks = [[] for _ in range(nh)]

    def hook(_m, inp, _out):
        qkv = recompute_qkv(attn, inp[0], nh, dh)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        for h in range(nh):
            chunks[h].append((o[:, h, -1, :] @ attn.out.weight[:, h * dh : (h + 1) * dh].T).cpu().numpy())

    with _with_hook(attn, hook):
        for tok in token_batches(inputs, device):
            model(tok)
    return [np.concatenate(c) for c in chunks]


@torch.no_grad()
def run_write_patched(model, layer, inputs, device, delta=None):
    """Forward pass with a per-row vector delta (N, d_model) SUBTRACTED from
    one layer's attention write at the readout position. delta=None runs the
    identical hooked path unpatched (the fair baseline). Returns raw model
    outputs (N,), normalized E-space."""
    attn = model.blocks[layer].attn
    state = {"offset": 0}

    def hook(_m, _inp, output):
        i = state["offset"]
        B = output.shape[0]
        if delta is not None:
            output[:, -1, :] -= torch.from_numpy(delta[i : i + B]).to(output.device)
        state["offset"] = i + B
        return output

    with _with_hook(attn, hook):
        out = [model(tok).cpu().numpy() for tok in token_batches(inputs, device)]
    return np.concatenate(out)


def run_write_interchanged(model, layer, tok, source_write, direction):
    """Forward pass with the readout-position attention write's coordinate
    along unit `direction` REPLACED by `source_write`'s coordinate -- the
    interchange intervention of distributed alignment search (Geiger et al.
    2023). tok: (B, L) int64 torch; source_write: (B, d_model) torch, the
    attention write captured on the source inputs; direction: (d_model,)
    torch unit vector. NOT no_grad: gradients flow to `direction`, so a DAS
    optimizer can train it. Returns raw model outputs (B,), normalized
    E-space, with graph."""
    attn = model.blocks[layer].attn

    def hook(_m, _inp, output):
        write = output[:, -1, :]
        coordinate_swap = (source_write @ direction - write @ direction)[:, None] * direction[None, :]
        patched = output.clone()
        patched[:, -1, :] = write + coordinate_swap
        return patched

    with _with_hook(attn, hook):
        return model(tok)


@torch.no_grad()
def neuron_acts(model, layer, inputs, device):
    """One layer's post-GELU MLP activations at the readout position over
    `inputs`, (N, d_mlp) numpy -- the per-input counterpart of
    mean_neuron_acts."""
    mlp = model.blocks[layer].mlp
    chunks = []
    with _with_hook(mlp.fc2, lambda _m, inp: chunks.append(inp[0][:, -1, :].cpu().numpy()), pre=True):
        for tok in token_batches(inputs, device):
            model(tok)
    return np.concatenate(chunks)


def neuron_readout_coefs(model, layer):
    """c_i = w_eff . W_out[:, i] for one layer's MLP: how hard neuron i's
    activation pushes the logit (pure weights, LN gain folded)."""
    return w_eff(model).cpu().numpy() @ model.blocks[layer].mlp.fc2.weight.detach().cpu().numpy()


@torch.no_grad()
def mean_neuron_acts(model, layer, inputs, device):
    """Grid-mean of one layer's post-GELU MLP activations at the readout
    position, (d_mlp,) -- the replacement values for neuron mean-ablation."""
    mlp = model.blocks[layer].mlp
    acc, n = None, 0

    def hook(_m, inp):
        nonlocal acc, n
        h = inp[0][:, -1, :]
        acc = h.sum(dim=0) if acc is None else acc + h.sum(dim=0)
        n += h.shape[0]

    with _with_hook(mlp.fc2, hook, pre=True):
        for tok in token_batches(inputs, device):
            model(tok)
    return acc / n


@torch.no_grad()
def run_neurons_ablated(model, layer, neuron_idx, inputs, device, means=None):
    """Forward pass with a SET of one layer's MLP neurons ablated at the
    readout position: replaced by their grid-mean activations (means: (d_mlp,)
    from mean_neuron_acts) or zeroed (means=None). neuron_idx=() runs the
    identical hooked path unpatched (the fair baseline). Returns raw model
    outputs (N,), normalized E-space."""
    mlp = model.blocks[layer].mlp
    idx = torch.as_tensor(list(neuron_idx), dtype=torch.long, device=device)

    def hook(_m, inp):
        if idx.numel() == 0:
            return None
        h = inp[0]
        h[:, -1, idx] = 0.0 if means is None else means[idx]
        return (h,)

    with _with_hook(mlp.fc2, hook, pre=True):
        out = [model(tok).cpu().numpy() for tok in token_batches(inputs, device)]
    return np.concatenate(out)


@torch.no_grad()
def mean_head_outputs(model, attn, cfg, inputs, device):
    """Grid-mean of per-head SDPA output, (nh, L, dh), for mean-ablation."""
    nh, dh = cfg.n_heads, cfg.d_model // cfg.n_heads
    acc = torch.zeros(nh, model.seq_len, dh, device=device)
    n = 0
    cap = {}

    def hook(_m, inp):
        qkv = recompute_qkv(attn, inp[0], nh, dh)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        cap["o"] = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B,nh,L,dh)

    with _with_hook(attn, hook, pre=True):
        for tok in token_batches(inputs, device):
            model(tok)
            acc += cap["o"].sum(dim=0)
            n += cap["o"].shape[0]
    return acc / n
