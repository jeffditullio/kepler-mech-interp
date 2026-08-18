# Metrics reference

Every metric quoted in the paper: skim the table, details below it. Scope
tags used in the paper: **(primary)** = `d8_l1_h2_gelu_lin_mse_800k_s0`;
**(family)** = the 210-model designed family; **(all)** = the 240 curated
checkpoints. Family distributions are quoted min/median/max with no accuracy
cuts; the degenerate d4 corner is named when it drives a tail.

Notation: eval grid = 400×200 jittered points over M ∈ [−π, π),
e ∈ [0, 0.999); E_pred = model output (radians); E_true = float64 Newton
solution; `w_eff = w_head ∘ γ_ln` = readout direction (output head weights
scaled elementwise by the final LayerNorm gain); "library" = the 11-term trig
set {1, M, sinM, cosM, sin2M, cos2M, e, e·sinM, e·cosM, e·sin2M, M·e};
"tail" = the deepest 6 of a field's 12 places.

| Group | Metric | Variable | What it tells you | Math |
|---|---|---|---|---|
| accuracy | median / bulk-max / max error | `median`, `max_bulk`, `max` | overall accuracy and worst cases | median / max over e<0.9 / max of \|E_pred − E_true\| on the eval grid |
| surface | M/e dependence | `M_dep`, `e_dep` | how much of the output is M- vs e-structure | M_dep = std(mean over e of E_pred grid); e_dep = std(residual); exact split: var = M_dep² + e_dep² |
| surface | shape survival | `e_corr`, `M_corr` | does surviving structure keep the true shape | Pearson(residual surface, true e-correction field); Pearson(M-profile, true M-profile) |
| line | linear structure | `L`, `p_L` | a value-ordered line exists (basis-free) | variance explained by the best digit-ordered line; p vs isotropic-Gaussian null |
| line | openness | `openness` | line vs circle | ‖emb(0) − emb(9)‖ / mean adjacent step (circle ≈ 1) |
| line | ramp deviation | `ramp_dev` | even spacing (basis-free) | max \|normalized DFT spectrum − ideal ramp 1/sin(πk/N)\| |
| line | curvature excess | `excess` | bend beyond a line (PC-anchored) | R²(value\|PC1+PC2) − R²(value\|PC1); circle calibration 0.42 |
| line | PC1 correlation | `pc1_pearson`, `pc1_spearman` | spacing/order along PC1 (PC-anchored) | \|Pearson\| / \|Spearman\| of PC1 coordinate vs digit value |
| depth | geometry depth | `geometry_M/e/gap`, `geometry_range_M/e` | how many places have their own position vector | leading run of places with dist-to-tail-centroid > 5·tail_noise; valid only if range = max dist/tail_noise ≥ 5 |
| depth | behavior depth | `behavior_M/e/gap` | how many places the output uses | leading run of digit-sensitivity places > 5·median(tail) |
| depth | attention depth | `attention_M/e/gap` | how many places attention routes | leading run of places with head-summed mean mass − median(tail) > 5·std(tail) |
| read | QK profile | (audit only) | where the readout query looks; which read moves with e | mean attention per key position over the grid; std over e of M-averaged weights |
| read | OV transfer curve | `ov_top_*`, `ov_sum_*` (`rho`, `r`, `span`) | is a digit read as a quantity | f_h(d) = w_eff·W_OV_h·tok(d), d = 0…9; "sum" = Σ_h f_h; stats \|Spearman\|, \|Pearson\|, max−min |
| read | head organization | `dissoc` | which heads carry the read | single / double / distributed from per-head mean-ablation |
| jobs | output literal fit | `dec_R2`, `dec_M`, `dec_esinM`, `library_resid_med` | what formula the output computes | least squares of E_pred (radians) on the library; median residual |
| jobs | component fits | (audit only) | which component writes which term | each component's write, projected on w_eff, fit on the library (logit space, additive) |
| jobs | e-purity | `purity_attn_share`, `parity_leak_attn/mlp`, `raw_e_cancellation` | does attention write e-corrections along the readout | rms e·sin(nM) content attn/(attn+mlp), content = \|coef\|·std(feature); forbidden-term rms / own fit-residual rms; \|c_attn[e]+c_mlp[e]\| / max\|each\| |
| jobs | MLP load | `mlp_factor` | how load-bearing the MLP is | median error with MLP mean-ablated ÷ baseline |
| register | rank-1 share | `register_share` | is the write's e-content one direction | top principal-direction variance share of the write's e-content |
| register | hiddenness | `register_cos_weff` | is the register invisible to the readout | \|cos(u_e, w_eff)\| |
| register | kill | `register_surv_esinM` | necessity of the register | surviving e·sinM coef after subtracting raw-e content along u_e, ÷ baseline |
| register | steering | `register_steer_half_corr/med` | the register is a settable dial | half-dose patch vs the model run at e/2: surface corr; median \|diff\| |
| register | transplant | `register_vs_e0_corr/med` | full patch reproduces e = 0 behavior | full-dose patch vs the model run at e = 0: corr; median \|diff\| |
| register | read/write ownership | `register_read_write_cos` | the writer head follows the reader | cos(per-head e-read share vector, per-head e-write share vector) |
| register | decode sufficiency | `decode_cubic/isotonic/full_write_rms` | does rank-1 carry the fine e-content | decode e from write·u_e per M-column (cubic / isotonic fit); reference = decode from full write |
| anatomy | comb fraction | `comb_frac` | staircase vs smooth neuron tuning | power fraction of the tuning curve at digit-grid frequencies |
| scaling | scaling fits | `scaling_fits.csv` | error vs params / steps | power-law fits over the family grid |

## Details

- **median / max_bulk / max** — 3 values/model. `tables/model_metrics.csv`;
  all 240.
- **M_dep / e_dep** — 2 values per condition (baseline + each ablation).
  `_analysis/ablation_mean.txt`; all 240; kernel `src/kernels/metrics.py`.
  Baseline pair also in `tables/circuit_metrics.csv` (`M_dep`, `e_dep`
  columns; 220 rows, 20 wrap Skips).
  e_dep holds the e main effect plus any M×e interaction. Family fact:
  M_dep/e_dep = 9.9–10.0 in 210/210, but the ratio is task-determined for any
  accurate solver (it certifies the demand, not the circuit).
- **e_corr / M_corr** — 2 values/condition, same audit. e_corr near 0 =
  incoherent wobble; near 1 = spared structure.
- **number-line battery (L, p_L, openness, ramp_dev, excess, pc1_*)** —
  computed on the 10 digit-token embeddings (one shared set per model);
  kernel `src/kernels/geometry.py`; `tables/model_metrics.csv`; all 240.
  Family: L 0.31/0.52/0.93, worst p_L 1.6e-4, openness 1.84/3.04/8.13,
  ramp_dev 0.001/0.034/0.11, excess 0/0.019/0.54 (four models past the 0.42
  circle calibration, all still open), |pc1_pearson| 0.62/0.97/1.00. The
  basis-free three (L, ramp_dev, openness) certify; the PC-anchored two
  (excess, pc1_*) measure deformation and are defeated by rotation.
- **read depth (all three instruments)** — kernel `src/kernels/depth.py`;
  audits `_analysis/read_depth.txt`; table `tables/circuit_metrics.csv`; all
  240. Shared conventions: depth = leading run of resolved places, k = 5,
  tail = deepest 6 places; gap distributions stable over k ∈ {3, 5, 8}
  (per-model sweep in the audit's gap-sweep line and the table's
  `gap_k3`/`gap_k8` columns; behavior modal +1 at every k: 200/205/206).
  Absolute depths track accuracy (behavior +0.82 with −log error) — the gap
  is the claim, not the level.
  - *geometry*: family 16 models n/a (no ladder-plus-floor; cluster range
    < 5; 9 d4, 4 d8, 2 d32, 1 d16 — 13 at ≤400k plus d4-s2 and clamp s3/s4
    at 800k); valid gap (n=194): +1 in 148, 0 in 28, +2 in 14, outside
    {0,1,2} in 4 (short-horizon d32). Ladder-plus-floor is at-convergence:
    800k valid in 57/60; range distribution 2.3/67/1175.
  - *behavior*: family gap +1 in 205/210, 0 in 5, no negatives — the
    cleanest family regularity in the paper. Tail floor is tight (std/median
    of tail: median 0.0045), so the floor estimator choice is immaterial.
  - *attention*: family gap +1 in 121, 0 in 38, +2 in 37, −1 in 3, ≥+3 in
    11. Excess rule because softmax keeps a high common floor. Kernel depth
    on the primary is 4/3 (the old by-eye count was 3/2 — quote the kernel).
- **QK profile** — per-head, per-position; `_analysis/readout_attention.txt`;
  all 240; not aggregated into a table.
- **OV transfer curve** — weights-only; no attention weights, no LayerNorm.
  The read is linear, so a position embedding adds a constant: one curve per
  head, shared by the M and e fields (the pre-2026-07-31 per-source stats
  were duplicates). Spans are in w_eff units, uncalibrated by LayerNorm's
  per-input scale — within-model ratios are the claim. 3 stats ×
  (n_heads + 1)/model; `_analysis/ov.txt` + `tables/circuit_metrics.csv`
  (top-span head + sum); all 240. Family (sum): |Spearman| 0.09/0.99/1.00
  (softening at width: below 0.95 in 57 models, 24 of them d32–d128 in the
  0.84–0.93 band); |Pearson| 0.001/0.95/0.999, not tracking accuracy
  (Spearman −0.03, Pearson +0.17 against −log error) — linearity is not
  task-pinned; the output's dec_M is.
- **dissoc** — 1 label/model; `tables/model_metrics.csv`; family counts
  single 28 / double 141 / distributed 41.
- **output literal fit** — 4 values/model; kernel `src/kernels/fits.py`;
  `_analysis/decompose.txt` (220 of 240; the 20 wrap models Skip the
  standard-task guard); `dec_*` CSV columns filled for linear-output models
  (n=195). Family dec_M 0.986±0.004.
- **component fits** — 12 values/component/model; `_analysis/decompose.txt`;
  220 of 240 (wrap models Skip). Signal terms: M (attn0); e·sinM, e·sin2M (mlp0). Parity forbids
  1, e, and the cosines in the output (E odd in M); sinM, sin2M, M·e are
  absorbers (returned even for the exact solution).
- **e-purity** — 4 values/model; kernel `src/kernels/fits.py::e_purity`;
  `_analysis/decompose.txt` + `tables/circuit_metrics.csv`; 220 of 240
  (wrap models Skip). The
  coefficient read is defined for LINEAR-output models only (n=195:
  0.001/0.053/0.827; 800k linear 0.002/0.036/0.217); through sigmoid, tanh,
  and clamp the logit-space parity and coefficient semantics do not apply, and
  those models are certified by the ablation instead. Organization-linked
  tail (distributed median 0.126, double 0.032, single 0.079). Parity
  leakage ≈ 1 means noise-level. Share is a ratio — quote beside
  `mlp_factor` when corrections are weak.
- **mlp_factor** — 1 value/model; `tables/model_metrics.csv`; family
  29/376/9,420.
- **register metrics** — u_e = the raw-e term's direction from the vector
  regression of the layer-0 attention write on the library
  (`src/kernels/fits.py::vector_fit`). Audits `_analysis/e_register.txt`;
  columns in `tables/model_metrics.csv`; all 240. Family: share
  0.72/0.94/1.00; |cos(u_e, w_eff)| 0.00/0.16/0.86 (> 0.5 only d4/short
  horizons); steer corr ≥ 0.999; vs-e0 corr is near-free — cite the median
  difference; read/write cos 0.08/0.998/1.00 (reversals at the d4
  short-horizon corner).
- **decode sufficiency** — 3 values/model; `_analysis/register_decode.txt` +
  `tables/circuit_metrics.csv`; 220 of 240 (wrap models Skip). rms in e-units. Family
  isotonic/full-write ratio 0.00/0.95/93.5: typical models at the floor;
  > 5× in 56, concentrated at d32–d64 and persistent across horizons within
  a seed.
- **comb_frac** — 1 value/neuron; neuron audits in `_analysis/`; quoted for
  the primary's corrector neuron n17 (0.751). Denominator-confounded across
  runs — compare within a model only.
- **scaling fits** — `tables/scaling_fits.csv`; computed by
  `repro/scaling_fit.py` over the family grid.
