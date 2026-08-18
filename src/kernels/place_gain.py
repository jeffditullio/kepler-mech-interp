"""
Per-place digit-gain kernels: the weights-only first-order account of digit
sensitivity (the read-depth mechanism).

At layer 0 the attention input at place p is exactly LN(tok(d) + pos_p): no
mixing has happened yet, so keys, values, and their per-digit variation are
closed-form in the weights. LayerNorm couples the position vector to the
digit direction; without it every place shares one digit gain by linearity
(`use_ln=False` is that control). Projecting each place's write through the
downstream Jacobian J and summing the OV and QK responses SIGNED across heads
gives a first-order prediction of the behavioral digit-sensitivity ladder.

Pure numpy (the kernels-layer rule). The analysis shell
(src/analysis/place_gain) supplies model weights, the grid-mean attention row,
and J.
"""

import numpy as np

N_DIGIT_VALUES = 10  # digit tokens 0-9; higher token ids (ANS) are not digits


def layernorm(x, gain, bias, eps=1e-5):
    """LayerNorm over the last axis (biased variance, matching torch)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def digit_writes(tok_emb, pos_emb, ln1_gain, ln1_bias, W_qkv, W_O, n_heads, ans_id, use_ln=True):
    """Layer-0 read tensors for every (head, place, digit), from weights alone.

    tok_emb (V, D), pos_emb (L, D), W_qkv (3D, D), W_O (D, D). Sequence layout:
    positions 0..L-2 are digit places, L-1 is the readout (ANS). Returns
      writes     (nh, P, 10, D)  W_O-projected value write of digit d at place p
      scores     (nh, P, 10)     q_ANS . k_p(d) / sqrt(d_head)
      self_write (nh, D)         the ANS position's own value write
      emb_ans    (D,)            ANS residual before attention
    """
    L, D = pos_emb.shape
    P = L - 1
    dh = D // n_heads
    digits = np.arange(N_DIGIT_VALUES)

    def qkv_of(x):  # (..., D) -> q, k, v each (..., nh, dh)
        y = layernorm(x, ln1_gain, ln1_bias) if use_ln else x
        qkv = (y @ W_qkv.T).reshape(*y.shape[:-1], 3, n_heads, dh)
        return qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]

    x = tok_emb[digits][None, :, :] + pos_emb[:P][:, None, :]  # (P, 10, D)
    _, k, v = qkv_of(x)  # (P, 10, nh, dh)
    emb_ans = tok_emb[ans_id] + pos_emb[P]
    q_ans, _, v_ans = qkv_of(emb_ans)  # (nh, dh)

    scores = np.einsum("pdnh,nh->npd", k, q_ans) / np.sqrt(dh)  # (nh, P, 10)
    writes = np.empty((n_heads, P, N_DIGIT_VALUES, D))
    self_write = np.empty((n_heads, D))
    for h in range(n_heads):
        O_h = W_O[:, h * dh : (h + 1) * dh]  # (D, dh)
        writes[h] = v[:, :, h] @ O_h.T
        self_write[h] = O_h @ v_ans[h]
    return writes, scores, self_write, emb_ans


def attention_mix(writes, self_write, attn_row):
    """Grid-mean attention output at the readout: sum of attn-weighted mean
    writes over all positions including the ANS self-write. attn_row (nh, L)."""
    nh, P = writes.shape[:2]
    mix = (attn_row[:, :P, None] * writes.mean(axis=2)).sum(axis=(0, 1))
    return mix + (attn_row[:, P, None] * self_write).sum(axis=0)


def first_order_sensitivity(writes, scores, self_write, attn_row, J):
    """First-order predicted digit sensitivity per place, plus gain ladders.

    attn_row (nh, L): grid-mean attention from the readout query (last entry =
    the ANS self-weight). J (D,): d(output in rad)/d(residual at the readout,
    after the attention write). Per digit d at place p, summed over heads:
      OV response:  attn_p * J.write_p(d)
      QK response:  the exact softmax derivative d(mix)/d(score_p) is
                    attn_p * (write_p - mix), so
                    attn_p * (score_p(d) - mean_d score_p) * J.(mean write_p - mix)
    Returns dict:
      g_OV (nh, P)  digit gain of the value write, ||std_d write_p||
      g_QK (nh, P)  digit gain of the routing score, std_d score_p
      pred (P,)     std_d of the signed OV+QK response, all heads
    """
    nh, P = writes.shape[:2]
    mix = attention_mix(writes, self_write, attn_row)
    g_OV = np.linalg.norm(writes.std(axis=2), axis=-1)
    g_QK = scores.std(axis=2)

    response = np.zeros((P, N_DIGIT_VALUES))
    for h in range(nh):
        attn = attn_row[h, :P, None]  # (P, 1)
        ov = attn * (writes[h] @ J)  # (P, 10)
        detach = (writes[h].mean(axis=1) - mix) @ J  # (P,)
        qk = attn * (scores[h] - scores[h].mean(axis=1, keepdims=True)) * detach[:, None]
        response += ov + qk
    return {"g_OV": g_OV, "g_QK": g_QK, "pred": response.std(axis=1), "mix": mix}
