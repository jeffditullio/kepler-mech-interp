# Claims ledger — Kepler interp paper

The claim summary, nothing else. One block per claim: **Claim · Holds · Fails ·
Metrics · Source · Guard** (guard = the misread the row exists to prevent);
paper home in each header. Tiers: `bedrock` (ablation-verified causal) · `solid`
(clean, reproduced) · `soft` (tendency/exemplar; paper says "we observe").
Numbers-truth = `tables/model_metrics.csv` + `tables/scaling_fits.csv` +
`models/<model>/_analysis/`; `repro/checks.py` enforces the conjunctions.
History = git. Row numbers are historic IDs, not an order.

**Family: 210 = 42 configs × 5 seeds** (grid 36 · mae-d8 · sig · tanh · clamp ·
relu · h4), all 1-layer + final LN. **Off-family: 30** = wrap cells 20 ·
lnoff 5 · 2-layer 5. Retired: the 5 non-d8 mae rungs.

## Framing & methods (intro; no evidence rows)

- First **weights-read circuit** account of a continuous **transcendental**
  equation; both qualifiers load-bearing (closest prior: SHO via probing).
- Representation matches task: number LINE vs mod-add circle / addition helix.
- Legibility, not accuracy (Boyd owns 4e-10).
- Testbed: 240 models + code released at
  https://github.com/jeffditullio/kepler-mech-interp.
- Scalar E-regression beats digit-AR (digit-CE mis-weights error by place).
- Jittered grid for every error statistic; clean grid ONLY for rFFT tools.

## Rows

### 1 · Number line — bedrock — §4.1, Figs 2–3, Fig A1, Table A1, App C
Claim   The ten digit embeddings form a line-dominated, open 1-D value code. Never a circle. The claim is the CONJUNCTION p_L ∧ L ∧ ramp_dev ∧ openness (basis-free certifiers, the checks.py floors); Pearson and excess are PC-anchored deformation meters (rotation / curvature census), not gates.
Holds   210/210. Primary pristine: PC1 85%, r² 0.99, excess 0.00, ramp_dev 0.02.
Fails   M50r_Ewrap cell 5/5 (Row 98). Aligned-wrap weakened 2/5, still passes.
Metrics Family worsts: p_L 1.6e-4 · L 0.31 · openness 1.84 · excess 0.54 (relu s0; three d8-s1 short-horizon 0.44–0.46, all open) · ramp_dev 0.111. Curved 29/210 (excess ≥ 0.15); |Pearson| < 0.95 in 81/210 (pc1_pearson column).
Source  battery embedding columns (incl. pc1_pearson/pc1_spearman); Table A1 rows straight from model_metrics.csv.
Guard   L can't separate line from circle (openness's job); openness's collapsed-ellipse hole is flagged by excess; where PC1 rotates, ramp_dev + openness carry the claim.

### 2 · Output shapes the circuit (head census) — soft — §4.2, App F, Fig A4
Claim   Head organization is contingent: curvature, MAE loss, and width tip it toward a split reader; behavior does not pin the circuit (accuracy moves ≤1.33× across output swaps).
Holds   Census 141 double / 28 single / 41 distributed. d32 30/30 double; d64 24/30; d128 27/30; d8 nonlinear outputs 12/15 vs linear 1/5; mae 3/5.
Fails   Every single-factor framing. Clamp s0 single but 4/5 seeds double (discriminator retired); "all linear seeds single" is 2/5.
Metrics Exemplars (s0): sig −head1 e_dep 0.21→0.023; tanh →0.004, roles permuted; linear/clamp-s0 single.
Source  dissoc column (ablation.py classifier); figure_output_dissociation.py.
Guard   Exemplars are seed 0. Head count is the least stable circuit property; the line and the register are the invariants.

### 4 · High-order Fourier–Bessel output — solid — §4.6, Fig 6, App I
Claim   The output is genuinely HIGH-order FB, not a low-order closed form: bₙ ≈ (2/n)Jₙ(ne) to n ≈ 30 (tail energy 1.3e-4), which wins worst case at matched median.
Holds   5-seed d8: median at FB order 5, bulk-max at 23; matched-median win ~5× (5.2e-2 vs FB5 2.5e-1). Ladder slopes (classical_ladder.csv rung=slope): FB 0.15, model 0.84, Boyd 0.93.
Fails   "Beats every FB rung" retired: FB23's bulk-max dips below d8's, at a 6-orders-finer median.
Metrics Rule-outs: Lagrange diverges past e 0.6627; sigmoid capstone 32× worse despite logit R² 0.993.
Source  spectrum.txt, classical_comparison.txt, tables/classical_ladder.csv.
Guard   Spectrum match = accuracy, not mechanism. "Order 6–8" is median-ladder placement only. s0's 3.1e-2 bulk-max was a lucky seed, never the headline.

### 5 · Distributed compute: attn structure, MLP precision — solid — §4.3, App G/H
Claim   Attention carries the M skeleton, the MLP the precision; no neuron encodes a Bessel term.
Holds   Primary MLP mean-ablation: 400× (M_corr 0.99 spared, e_corr −0.41 dies). Onset negative replicated on primary, d128, 2-layer.
Metrics Family: min 29×; factor ∝ params^0.46 R² 0.90.
Source  ablation.py --mean --mlp (mlp_factor column); scaling_fits.csv.
Guard   MEAN-ablation is the declared method (zero overstated: 707×). High harmonics live in the embeddings, not MLP neurons.

### 6 · Scaling: graded power-law spectrum — solid — App H, Fig A6, Table A4
Claim   Median/bulk-max/max are separable power laws, no floor, graded exponents, steps > params.
Holds   Drop-d4: median −0.45/−0.39 · bulk-max −0.41/−0.33 · max −0.34/−0.28 (SE ±0.03/±0.02; grading 2.5–4 SE). Median steepest in all 5 seed refits.
Fails   d4 degenerate. Middle rung soft: bulk-max/max swap in 2/5 refits.
Source  scaling_fit.py → scaling_fits.csv. Per-seed refit rows included.
Guard   Old "bulk-max floor" and "FLOP collapse" were d4 artifacts, retracted.

### 8 · Loss sets the operating point — soft — App I
Claim   MAE sacrifices the cusp for the bulk: at d8 the max/median gap under MAE is a multiple of the MSE gap in every seed.
Holds   d8, 5 seeds per loss: MAE gaps 93–145× vs MSE 32–56× (max/median per seed, model_metrics.csv).
Guard   "≈Boyd" is operating-point language only, never shape. No degree-matching.

### 9 · The e-register — solid — §4.4, Fig 1b, Fig 5, App G
Claim   Attention hands e to the MLP through ONE rank-1 direction ⊥ the readout; subtracting it yields the model's own e=0 behavior; SCALING it rescales e — the register is the model's e variable.
Holds   210/210: share 0.72–1.00, half-dose corr ≥ 0.9988, zero-dose med ≤ 0.197 (worsts all short-horizon corners). Off-family: lnoff 5/5 · extM unwrapped 10/10 · M50r_Ewrap de-ranks (share 0.49–0.68) yet the dial steers (≥0.995, seam-only breaks).
Fails   Kill not uniform: surviving e·sinM to 55%, >20% in 15/210 (six d16-MSE, six d64-MSE, relu s0, one d128, one tanh seed). 2-layer SEED-SPLIT: s3/s4 clean layer-0 kill, s0/s1/s2 fail (surv 78–99%; s0's share 0.96 is fit geometry, 0.90-collinear with u_M). Scoped 1-layer. Decode fine-content NOT family-universal (2026-07-30, register_decode frozen for all 240; 220 computed, 20 wrap Skips): isotonic/floor ratio median 0.95 but >5× in 56/210, width-linked (d64 median 11.4, d32 1.87) and seed-persistent (d32-s1, d64-s2 across horizons) — at width the fine e rides off the top direction (consistent with the rank-5 note). Family cubic rms min/med/max 9.0e-4/8.0e-3/5.7e-2. OPEN: interpret the wide-model tail.
Metrics Primary: kill 0.767→−0.05 / 0.441→−0.00, M intact; cos(û_e,w_eff) 0.04. Controls: e·sinM-direction patch null; random direction not null. Fit-sufficiency: structure kept, precision-only cost 5.3e-2. Decode: cubic calibration rms 3.8e-3 ≈ full-write floor (register_decode.txt). Fig 1b gap is one-sided and rank-1's price (rank-5 patch collapses it).
DAS     Optimization cross-check, 7 models (primary + 4 re-seeds, lnoff s0, d128): e-interchange DAS lands on the SAME channel — same top MLP readers, cos(v, û_e) 0.93–0.99, dial functionally ≈ û_e — with a slightly better 1-D aim. Full random-M interchange is a stricter READ task û_e was never fit for (û_e at/below the no-op floor 7/7; the optimum still reads through the same neurons, cos 0.59–0.89; d128 0.843 → width does not close it). Cosines are LOWER bounds (swap-cancelled terms are free directions). The exact-dual read direction is a dead end (cos ≤ 0.39 to everything).
Source  e_register.py + steering_asymmetry.py (reproduces every Fig 1b number) + das_register.py (DAS cross-check, DEEP tier); register_* columns; audits models/*/_analysis/e_register.txt + das_register.txt + register_decode.txt (since 2026-07-30 on 220 of 240; the 20 wrap models Skip the standard-task guard).
Guard   vs-e0 CORR is near-free — cite med diff. Patch is linear-in-e (overshoot; deep reads survive). Ownership = gauge, follows the read (corr 0.92 over 205 two-head models; 4-head cos 0.82–0.98 over 5 seeds; the three reversals = the degenerate d4 short-horizon s3 corner; register_read_write_cos column). Visibility is at-convergence: >0.5 only at d4/short horizons (max 0.86); all 800k low.

### 10 · MLP anatomy: spline family + comb corrector; lnoff twin — solid — §4.4, Figs A5/A7, App G
Claim   11/32 neurons live: a soft-spline family plus ONE digit-comb corrector (n17); the Fig A7 block residual is the digit floor. The algorithm relearns without final LN.
Holds   n17 comb_frac 0.751; ablating it → 12× AND blockier (0.105→0.141). e needs no comb (linear register path). lnoff twin 5 seeds: everything reappears at 2.1–3.1× (five seeds); register visible (visw 0.28–0.72) → DLA-hiddenness is LN-contingent.
Source  comb_ablation.py; frozen_ln.py; comb_depth.py (audits models/*/_analysis/).
Guard   |c|·σ ≠ damage under live LN (10th neuron = 100×); comb FRACTIONS are denominator-confounded across runs. OPEN: what cancels the register's w_eff component on the twin.

### 11 · Reader head: QK routes, OV converts; necessity — bedrock — §4.2, Fig 4, Table A2, App E
Claim   QK concentrates readout attention on leading digits by significance; OV converts digit value monotonically to logit; the reader head is NECESSARY.
Holds   Primary: attn M .319/.090/.028, e .114/.035; OV spans 0.005 vs 0.001. −head1 ≈ flat (1% variance); −head0 spares structure, pays 180× precision. Double dissociation on sig/tanh.
Holds   FAMILY OV (2026-07-31, ov.py reworked; numbers re-pinned to circuit_metrics.csv 2026-08-05): the weights-only OV read is LINEAR, so source position adds a constant — ONE transfer curve per head (the old per-source stats were duplicates; Fig 4c redrawn). Head-summed curve (the block's read, well-defined for every organization): |Spearman| 0.09/0.99/1.00 — monotone quantity-read is family-TYPICAL, softening at width (sum_rho < 0.95 in 57/210; 24 of them are d32–d128 in the 0.84–0.93 band; family min is a sigmoid seed at 0.21 excl. d4). |Pearson| median 0.95; linearity does not track accuracy (pinned def: Spearman(sum |r|, −log10 median err) = −0.03; the Pearson variant is +0.17) — per-head/per-curve linearity is NOT task-pinned (the output's dec_M is; the parts are gauge). Composite read does NOT straighten deformed embedding lines (summed-curve |r| below the embedding |Pearson| in 128/210). Width softening survives attention-mass weighting (2026-08-05, repro/weighted_ov.py → tables/weighted_ov.csv: weighted summed curve <0.95 in 67 vs 57 unweighted) — real, not a summation artifact. Softening is not an accuracy defect (2026-08-05, trivial arithmetic over circuit_metrics.csv sum_rho × model_metrics.csv median, 210 family rows): Spearman(sum_rho, −log10 median err) = −0.30; the 57 sub-0.95 models skew wide with median err 4.9e-4 vs 1.4e-3 for the 153 at ≥0.95 — the softened readers are, if anything, the MORE accurate models (width confound: both track d).
Fails   Sufficiency dropped (vacuous at 2 heads).
Source  readout_attention.txt, ov.txt (per-head + sum lines, all 240) + tables/circuit_metrics.csv, ablation_mean.txt.
Guard   Mean attention mass does NOT carry the 10×/place weighting; place and position are confounded by the fixed-width layout. Kernel attention depth (excess rule) resolves FOUR M places, not the by-eye three — quote the kernel number.

### 12 · Read depth: LayerNorm sets per-place gain — solid (primary) — App D
Claim   LN couples position to digit gain; a first-order weights account reproduces digit sensitivity at every resolved place within ~2× (no-LN control exactly flat).
Fails   The deep floor is not first-order (~9× overshoot; OV-vs-QK cancellation).
Source  place_gain.txt; kernels/place_gain.py.
Guard   Claim only the resolved places.

### 13 · Digit sensitivity, position geometry, the M/e calculus — solid — §4.2, App D, Fig A2
Claim   Sensitivity falls ~10×/place to a 2.3e-4 floor; position geometry separates only leading places (collapse ≈500×); both put M one place deeper than e; the ~10× M/e ratio is the equation's calculus (sinE ~0.6 × range 6.3).
Holds   FAMILY (2026-07-31, read_depth battery, kernels/depth.py, k=5; gap distributions stable over k ∈ {3,5,8} — promoted 2026-08-05 into the audits' gap-sweep line and circuit_metrics gap_k3/gap_k8 columns; behavior modal +1 at every k: 200/205/206): behavior gap M−e = +1 in 205/210, 0 in 5, no negatives — the cleanest family regularity we have. M_dep/e_dep 9.9–10.0 in 210/210 (exact 9.86–10.02; M_dep/e_dep columns promoted into circuit_metrics.csv 2026-08-05; task-forced for any accurate solver). Attention-mass gap +1 in 121/210 (0:38, +2:37, −1:3, ≥+3:11). Geometry (with the cluster-validity guard): n/a in 16 models, valid gap dist over 194 = {+1:148, 0:28, +2:14, −2:2, −1:1, +3:1} — 190/194 inside {0,1,2}; the 4 stragglers are short-horizon d32 marginal geometries. Ladder-plus-floor is AT-CONVERGENCE: cluster range (max dist/tail noise) 2.3/67/1175, no floor (guard: range < 5) in 16 (9 d4, 4 d8, 2 d32, 1 d16; 13 at ≤400k + d4-s2 and clamp s3/s4 at 800k), so 800k valid 57/60.
Source  digit_sensitivity.txt; read_depth.txt (all 240) + tables/circuit_metrics.csv; figure_position_geometry.py.
Guard   The 10× slope is accuracy-forced; the information is in the FLOOR. ABSOLUTE depths track accuracy (behavior +0.82 with −log err) — the GAP is the claim, not the level. §4.1's at-convergence scoping LANDED 2026-08-05 (57/60; "typically one place deeper"). An earlier "56/60 (incl. MAE s3)" used an exploratory <20× cut, not the committed <5× guard — corrected.

### 14 · Decompose: attention writes pure M; the MLP writes the products — solid — §4.3, Tables 2/A3
Claim   Along w_eff: attn0 pure-M (e-terms ≤ 7e-4), mlp0 carries the products; output fit ≈ truth fit to ≤0.001 in all 11 terms (M·e absorber included).
Holds   attn0 R² 0.989, mlp0 0.975, output 0.998 (M .988, e·sinM .767, e·sin2M .441). Median library residual 0.027 rad, 18× the model's error (family range 0.027–0.037; library_resid_med column). 2-layer: attn1 writes the product straight onto the readout — no hidden e variable there.
Holds   FAMILY e-purity (2026-07-31, kernels/fits.py::e_purity in decompose audits; 2026-08-05 tail autopsy): the coefficient read is LINEAR-OUTPUT-only — through sig/tanh/clamp the logit-space parity/coefficient semantics break (the all-outs 800k "worst" 0.49 is tanh-800k-s3, a ratio of near-noise numbers); nonlinear outs are ablation-certified only. Linear n=195: 0.001/0.053/0.827 (max is d16-50k-s2, a LINEAR short-horizon model — the tail is led by short-horizon linear models); 800k linear (n=45): 0.002/0.036/0.217; trend (linear, median by horizon): 0.095 at 25k → 0.036 at 800k. The long-horizon linear tail is REAL joint carriage in distributed organizations (d128-400k-s1: attn e·sinM +0.019 vs mlp +0.054; ≥400k tops 0.22–0.26). Organization-linked (linear models): distributed median 0.126 vs double 0.032 vs single 0.079. Attn parity leakage at noise (median 0.9× own fit residual); raw-e cancellation < 0.2 in 21/210 (component-level parity violations that cancel jointly — real but minority).
Source  decompose.txt (full 11-coef matrix, output + truth rows; e-purity line) + tables/circuit_metrics.csv.
Guard   The output-truth match is an ACCURACY tautology; ablations are the proof. Component rows are relative w_eff units. The test-pinned truth triple is a different sample; don't cross-quote. SCOPE: "attention writes pure M" is primary + family-TYPICAL, not universal — abstract/intro/§4.3 scoping LANDED 2026-08-05 ("writes the M skeleton"; §4.3 family sentence linear-scoped). An earlier prose pass quoted a PRE-KERNEL scratch metric (0.087→0.020, max 0.78) — corrected to the kernel/CSV numbers. Share is a ratio: quote beside mlp_factor when corrections are weak.

### 15 · The cusp is the equation's wall — solid — §4.6, App I
E ≈ (6M)^(1/3) at the corner; every method concentrates error there; grid-max still scales. Source: error_map.txt.

### 16 · Width-invariant algorithm, width-eroded legibility — solid — App H
Claim   Line + register persist at every width/horizon/seed; L falls, MLP reliance and effective order rise with width. The register does NOT dilute (|corr| ≤ 0.3).
Fails   d4 degenerate.
Guard   This answers Clock-and-Pizza to workshop standard; keep the caution in related work.

### 17 · Attribution blind-spot trio — soft — §5, App J
Claim   Energy share (52/48 vs flatten/400×), DLA (register ⊥ w_eff), and |c|·σ (10th neuron 100×) all fail, each failure measured; on the lnoff twin damage matches the score and the register is visible — the blind spots belong to the normalized model.
Source  App J numbers; frozen_ln.txt; e_register.txt visw.

### 98 · Wrap: no circle; the hardest cell costs the line — solid (d8/800k) — §4.5, Fig A8, Table A5, App K
Claim   Wrapping does not buy a circle. Aligned cells keep the line and primary accuracy; incommensurate unwrapped keeps the line; incommensurate WRAPPED pays 82× AND dissolves — fails every line test, no circle in its place. Graded: intact → weakened (2/5 aligned-wrapped) → gone.
Holds   20 models, 5 seeds/cell; predictions written before launch.
Fails   Old "line survives every cell" RETRACTED.
Metrics M50r_Ewrap: median 0.14–0.17 (82×), L 0.07–0.23, wrap 1.04–1.48, flat spectrum, scrambled PCA, worst at seams.
Source  battery line columns on Ewrap rows; wrap tools.
Guard   The dissolved cell is a THIRD STATE: neither line nor circle, precursors stalled, geometry unresolved. Accuracy-forced reads can't distinguish arbitrary keys there. Caveats: LR freeze, width unresolved.

### 99 · The wrap is computed behaviorally, by different means — solid — App K
Claim   Every cell computes the wrap (per-rotation flat 1.0–2.4; phase corr 0.72–0.99): digit ROUTING when aligned (phase R² 0.97–0.98, 5 seeds; accuracy 7.7e-4–2.3e-3) vs a spline along the M line when incommensurate (R² 0.01–0.07; wrapped-incommensurate final 0.10–0.26).
Source  wrap_marginals, neuron_periodicity.

### 100 · Circle precursors at the predicted frequencies — solid (d8/800k) — §4.5, Fig A8
Claim   Precursors form at f_j = frac((R/π)·10^(−j−1)), zero free parameters; above the 0.23 line-code null the entire run in 5/5 seeds (gains 0.5–1.0); convert in none.
Source  circle_probe / circle_progress.

### Scope — §5
fp32 ceiling · cusp wall · one-specimen depth (QK/OV, place_gain, anatomy = primary-only; exemplars s0) · 2-layer seed split (joint patch = named next experiment).
