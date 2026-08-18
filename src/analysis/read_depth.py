"""
Read depth per instrument: how many leading places of M and of e the model
resolves, measured three independent ways on one checkpoint (kernels.depth):

  geometry   position-embedding distance to the collapsed tail cluster
  behavior   digit sensitivity (vary one digit, std of E_pred)
  attention  mean attention mass from the readout query, summed over heads

Each instrument reports an M depth, an e depth, and their gap M - e, at the
canonical k = 5 and at k = 3 and 8 (the k-stability of the gap is part of
the claim). The §4.2 claim is the GAP (~1 place at every stage, a law in
behavior); the instruments are thresholded proxies of the routing chain and
do NOT obey behavior <= attention <= geometry model by model.

Usage:
    uv run python -m src.analysis.read_depth d8_l1_h2_gelu_lin_mse_800k_s0
"""

from src.analysis._cli import Result, run_tool
from src.analysis.digit_sensitivity import sensitivity
from src.analysis.readout_attention import patterns
from src.core.data import positions
from src.core.runs import Bundle
from src.kernels.depth import MIN_CLUSTER_RANGE, K, attention_depth, geometry_depth, sensitivity_depth

K_SWEEP = (3.0, K, 8.0)


def analyze(bundle: Bundle) -> Result:
    """Result metrics:
    geometry        {"M": depth | None, "e": ...} from position-embedding
                    clustering; None when the field has no ladder-plus-floor
                    structure (cluster_range < MIN_CLUSTER_RANGE)
    geometry_range  {"M": cluster_range, "e": ...} the ladder's dynamic range
    behavior        {"M": depth, "e": depth} from digit sensitivity
    attention       {"M": depth, "e": depth} from summed-head attention mass (layer 0)
    gap             instrument -> M depth - e depth (None if either side is None)
    gap_sweep       k -> {instrument -> gap} at every k in K_SWEEP
    """
    cfg, ck, device = bundle.cfg, bundle.ck, bundle.device
    layout = positions(cfg)
    fields = {"M": layout["M"], "e": layout["e"]}

    pos_emb = ck["model"]["pos_emb.weight"].float().numpy()
    sens = sensitivity(cfg, ck, device)
    attn_by_layer, _MM, _EE = patterns(cfg, ck, device, layout["readout"])
    mass = attn_by_layer[0].mean(axis=0).sum(axis=0)  # (L,): grid-mean, head-summed

    def depths_at(k):
        geometry, geometry_range = {}, {}
        for tag, sl in fields.items():
            depth, _dist, _noise, cluster_range = geometry_depth(pos_emb[sl], k=k)
            geometry[tag] = depth if cluster_range >= MIN_CLUSTER_RANGE else None
            geometry_range[tag] = round(cluster_range, 1)
        behavior = {tag: sensitivity_depth(sens[tag], k=k)[0] for tag in fields}
        attention = {tag: attention_depth(mass[sl], k=k)[0] for tag, sl in fields.items()}
        geometry_ok = geometry["M"] is not None and geometry["e"] is not None
        gap = {
            "geometry": geometry["M"] - geometry["e"] if geometry_ok else None,
            "behavior": behavior["M"] - behavior["e"],
            "attention": attention["M"] - attention["e"],
        }
        return geometry, geometry_range, behavior, attention, gap

    geometry, geometry_range, behavior, attention, gap = depths_at(K)
    gap_sweep = {k: depths_at(k)[4] for k in K_SWEEP}

    out = [f"{bundle.run_name} (step {ck.get('step', '?')})  read depth (places resolved, k={K:.0f})"]
    out.append("  instrument   M   e   gap")
    for name, d in (("geometry", geometry), ("behavior", behavior), ("attention", attention)):
        cells = [("n/a" if d[t] is None else f"{d[t]:3d}") for t in ("M", "e")]
        g = gap[name]
        out.append(f"  {name:10s} {cells[0]:>3s} {cells[1]:>3s} {'  n/a' if g is None else f'{g:5d}'}")
    ranges = f"M {geometry_range['M']:.1f}  e {geometry_range['e']:.1f}"
    out.append(f"  geometry cluster range  {ranges}  (valid >= {MIN_CLUSTER_RANGE:.0f})")
    sweep_cells = []
    for k in K_SWEEP:
        g = gap_sweep[k]
        cells = "/".join("n" if g[i] is None else f"{g[i]:+d}" for i in ("geometry", "behavior", "attention"))
        sweep_cells.append(f"k={k:.0f} {cells}")
    out.append("  gap sweep (geom/behav/attn)  " + "   ".join(sweep_cells))
    return Result(
        "\n".join(out),
        geometry=geometry,
        geometry_range=geometry_range,
        behavior=behavior,
        attention=attention,
        gap=gap,
        gap_sweep=gap_sweep,
    )


def main() -> None:
    run_tool(analyze)


if __name__ == "__main__":
    main()
