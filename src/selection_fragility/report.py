"""selection_fragility.report — the one-screen report.

Prose lives HERE, never in the returned data structure (prior art: `sensemakr` puts interpretation
only in `print()`; `deepchecks` separates a `Check` that measures from a `Condition` the user writes
at their own call site). No "fragile"/"screen" verdict language anywhere -- that whole design was
retired (§1 of the design doc), not reworded."""
import numpy as np
from .fragility import pooled_winner, decision_breakdown
from .identify import _run_mcs
from .resolution import resolution_report, mcb_bound
from .prop22 import prop22_certifies
from .pivot import concentration_share, pivot_agreement
def _fmt_label(lbl):
    if hasattr(lbl, "strftime"):
        try:
            return lbl.strftime("%Y-%m")
        except Exception:
            return str(lbl)
    return str(lbl)
def _disambiguate_labels(labels_subset):
    """Format a list of period labels for display, the same way _fmt_label does, EXCEPT any labels
    that collide after its "%Y-%m" truncation get a finer "%Y-%m-%d" label instead (or, if that still
    collides, an index suffix). FIXED 2026-08-26 (round-2 review, "does it deliver on its promises"
    lens): PIVOT's "periods responsible" list used bare _fmt_label on every removed period, so two
    genuinely distinct periods landing in the same month printed as an indistinguishable duplicate,
    e.g. `k*=2 of 5 periods (2020-04, 2020-04)` -- reproduced directly. Only labels that actually
    collide in THIS call's own set are expanded; a report with no collision looks exactly as before."""
    base = [_fmt_label(l) for l in labels_subset]
    if len(set(base)) == len(base):
        return base
    finer = []
    for lbl, txt in zip(labels_subset, base):
        if hasattr(lbl, "strftime"):
            try:
                finer.append(lbl.strftime("%Y-%m-%d"))
                continue
            except Exception:
                pass
        finer.append(txt)
    if len(set(finer)) == len(finer):
        return finer
    seen = {}
    out = []
    for txt in finer:
        seen[txt] = seen.get(txt, 0) + 1
        out.append(txt if finer.count(txt) == 1 else f"{txt} (#{seen[txt]})")
    return out
def _fmt_pct(x):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:.1f}%"
def _strict_win_counts(L):
    models = sorted(L)
    M = np.vstack([np.asarray(L[m], float) for m in models])
    pmin = M.min(axis=0)
    is_min = (M == pmin[None, :])
    unique_min = is_min & (is_min.sum(axis=0) == 1)
    wins = unique_min.sum(axis=1)
    return {models[i]: int(wins[i]) for i in range(len(models))}
def report(panel, alpha=0.10, full=False):
    """One-screen summary of a LossPanel: VERDICT (identification), LEADERBOARD (mean loss / in-MCS /
    periods won), RESOLUTION (observed edge vs MDE, MCB bound), PIVOT (k*, named periods or largest
    contributor, concentration_share, pivot_agreement). `full=True` is accepted for forward
    compatibility with a future expanded view; the default view is intentionally the compact one.

    COST SCALES STEEPLY WITH MODEL COUNT (K), NOT PERIOD COUNT (T) -- documented 2026-09-07, round-4
    8-lens PyPI-preflight audit. The underlying `arch.bootstrap.MCS` elimination is the dominant
    cost; measured directly on this package's own report(): ~1s at K=100, ~3s at K=200, potentially
    minutes at K=500+ (T scales fine even to T=5000). `LossPanel.from_losses`/`from_forecasts`
    already warn once, at construction time, when K exceeds ~100 -- see that warning for the
    up-to-date measured numbers rather than trusting this docstring to stay current."""
    # FIXED 2026-09-02 (10-agent code-review pass, CONFIRMED GAP): report() used to access
    # panel.losses/.labels directly with no type check, so a raw dict (the Core API's fragility()/
    # model_confidence_set() input shape, and what examples/quickstart.py itself teaches) raised a
    # bare AttributeError instead of a clear message. Every sibling raw-array-tier entry point
    # (pooled_winner, decision_breakdown, winner_stability, fragility()) already accepts a
    # LossPanel via _unwrap_panel() -- this is the OPPOSITE direction (report() REQUIRES a
    # LossPanel, since it uses panel-only fields like .labels), so the fix is a clear rejection,
    # not an unwrap. report() is the README's own "recommended entry point" -- the single most
    # plausible mistake for a user coming from the Core API or migrating between entry points.
    if not (hasattr(panel, "losses") and hasattr(panel, "labels") and hasattr(panel, "weights")):
        raise TypeError(
            f"report() expects a LossPanel, got {type(panel).__name__}. Build one first with "
            f"LossPanel.from_losses(...) or LossPanel.from_forecasts(...)."
        )
    L = panel.losses
    w = panel.weights
    labels = panel.labels
    models = list(L)
    K = len(models)
    T = len(labels)
    # FIXED 2026-08-26 (independent ML-engineer review): report() used to call identified()/
    # mcs_size()/_run_mcs() with no `w=`, silently computing MCS membership on the UNWEIGHTED panel
    # while the LEADERBOARD means just below use the panel's real weights -- exactly the
    # "champion marked both in-MCS and excluded" inconsistency resolution_report() already refuses
    # to reproduce internally (see identify._validate_mcs_weights). Passing `w=w` here makes VERDICT
    # consistent with LEADERBOARD, and surfaces the same clean refusal `resolution_report` gives
    # for genuinely non-uniform weights -- caught below and reported honestly rather than letting
    # report() crash uncaught partway through (previously: VERDICT/LEADERBOARD rendered on stale
    # unweighted MCS stats, then RESOLUTION crashed with an uncaught ValueError).
    # DEDUPED 2026-09-07 (round-4 8-lens PyPI-preflight audit, CONFIRMED HIGH): this used to call
    # identified(), mcs_size(), AND _run_mcs() separately -- three independent calls with identical
    # arguments, each re-running the same arch bootstrap MCS computation from scratch (on top of a
    # FOURTH, inside resolution_report() below, which is left as-is -- see its own docstring for why
    # that one is harder to dedupe safely without changing a public function's signature). Measured:
    # report() at K=200 models took ~9.3s and the MCS computation alone accounted for the overwhelming
    # majority of it via cProfile. identified()/mcs_size() are both trivial wrappers around
    # _run_mcs()'s own result (identified = len(survivors)==1, msize = len(survivors)), so calling
    # _run_mcs() once and deriving both from its result is a pure speedup with no behavior change --
    # verified byte-identical output on report()'s own existing golden-value test before/after.
    mcs_error = None
    try:
        m = _run_mcs(L, alpha=alpha, w=w)
        survivors = set(m.included)
        msize = len(survivors)
        ident = bool(msize == 1)
    except ValueError as e:
        ident = msize = None
        survivors = set()
        mcs_error = str(e)
    means = {mm: float(np.average(np.asarray(L[mm], float), weights=w)) for mm in models}
    wins = _strict_win_counts(L)
    champ = pooled_winner(L, w)
    # Moved above VERDICT (round-6 fix, 2026-08-27): a reader trained on significance-testing
    # conventions (e.g. clinical/biostatistics audiences) can read "VERDICT: identified" in
    # isolation and stop before reaching RESOLUTION's power/sample-size caveat several lines below
    # -- MCS identification is a different statistic from RESOLUTION's power-based read and the two
    # CAN disagree. Computing `res` first lets VERDICT carry an inline pointer to RESOLUTION instead
    # of silently relying on the reader to keep scrolling.
    try:
        res = resolution_report(L, w, alpha=0.05, power=0.80, mcs_alpha=alpha)
    except ValueError:
        # Same non-uniform-weight MCS refusal, reached here via R2's `_identified(..., w=ww)` call
        # inside resolution_report() -- `resolved`/`observed_edge`/`mde` don't themselves need MCS,
        # but resolution_report() computes `is_identified` unconditionally before returning. Fall
        # back to the package's own existing "undetermined" contract (below) rather than duplicate
        # R1's formula here; _fmt_pct already renders NaN as "n/a".
        res = {"resolved": False, "observed_edge": np.nan, "mde": np.nan,
               "significance_boundary": np.nan, "binding_rival": None,
               "identified": None, "fragility_read": "undetermined"}
    lines = []
    if mcs_error is None:
        lines.append(f"VERDICT: {'identified' if ident else 'not identified'} (MCS size {msize} of {K})")
        if not res.get("resolved", False):
            lines.append("  (see RESOLUTION below -- identification alone does not mean the edge is")
            lines.append("   powered at this sample size)")
    else:
        lines.append(f"VERDICT: not available for this panel -- {mcs_error}")
    lines.append("")
    lines.append("LEADERBOARD")
    for mm in sorted(models, key=lambda x: means[x]):
        flag = "n/a" if mcs_error is not None else ("in MCS" if mm in survivors else "excluded")
        marker = " (champion)" if mm == champ else ""
        lines.append(f"  {mm:<15} mean={means[mm]:.4f}  {flag:<9}  periods won={wins[mm]}{marker}")
    lines.append("")
    lines.append("RESOLUTION")
    edge_txt = f"observed edge {_fmt_pct(res['observed_edge'])} vs MDE@80% power {_fmt_pct(res['mde'])}"
    if res["resolved"] and res["observed_edge"] is not None and np.isfinite(res["observed_edge"]) and res["observed_edge"] <= 0:
        # FIXED 2026-08-26 (independent ML-engineer review): on an exact tie (e.g. byte-identical
        # models) the champion is arbitrary, the observed edge is exactly 0, and the same-sample SE
        # is also exactly 0, so R1's `observed_edge >= significance_boundary` is trivially satisfied
        # and `resolved` comes out True -- mathematically defensible (the true edge IS resolved to
        # be exactly zero) but printed next to a named "(champion)" it reads as "we found a real
        # winner," which is the opposite of what a 0% edge means. State the tie explicitly instead
        # of the bare "-- resolved".
        lines.append(f"  {edge_txt} -- resolved: no edge exists (the champion ties its binding rival exactly)")
    elif res["resolved"]:
        lines.append(f"  {edge_txt} -- resolved")
    else:
        lines.append(f"  {edge_txt} -- cannot determine at this sample size")
    try:
        bound = mcb_bound(L, w)
        # FIXED 2026-08-26 (independent ML-engineer review): this text still said "...may be up to
        # X% WORSE than the binding rival" after resolution.py's own mcb_bound() docstring was
        # corrected on 2026-08-17 to say the opposite -- `bound = point + t_crit*SE` is by
        # construction >= the observed (positive) edge, so it is an upper bound on how much
        # BETTER/bigger the champion's true edge could be, never a bound on how much worse the
        # champion could be. Verified against the package's own worked example (edge=6.8%,
        # bound=44.8%, only sensible as "at least this good"). Math unchanged; wording corrected
        # to match what resolution.py's docstring already says was the fix, but was never applied
        # here -- CHANGELOG.md documents this as fixed; the shipped string still said "worse".
        lines.append(f"  MCB bound: the champion's true edge over the binding rival may be as large "
                     f"as {_fmt_pct(bound)}, at 95% confidence")
    except ValueError as e:
        lines.append(f"  MCB bound: not available -- {e}")
    lines.append("")
    lines.append("PIVOT")
    k, opp, removed = decision_breakdown(L, w, a=champ)
    if k == 0 or opp is None:
        lines.append("  k*=0 -- no strict pooled winner (an exact or effective tie); nothing is pivotal")
    else:
        # WHICH periods are responsible is a DETERMINISTIC FACT about the data (same tier as k* itself
        # -- decision_breakdown's `removed` list, not an interpretation of it), so it is always shown,
        # R1/R2 or not. Only built once and reused in both branches below.
        if k <= 5:
            # FIXED 2026-09-02 (10-agent code-review pass, CONFIRMED UX GAP): `removed` comes back
            # from decision_breakdown() in greedy-removal (largest-contribution-first) order, not
            # period order -- with real date/year labels that reads as a typo (e.g. "2018, 2017,
            # 2019, 2020, 2021"). Periods are stored chronologically by panel index (the same
            # assumption `top_i`/_fmt_label rely on elsewhere in this function), so sorting the
            # removed indices restores chronological display without touching which periods are
            # reported responsible or k* itself.
            resp = ", ".join(_disambiguate_labels([labels[i] for i in sorted(removed)]))
            period_txt = f"k*={k} of {T} periods ({resp})"
        else:
            la = np.asarray(L[champ], float)
            lb = np.asarray(L[opp], float)
            c = w * (lb - la)
            top_i = int(np.argmax(c))
            period_txt = f"k*={k} of {T} periods (largest single contributor: {_fmt_label(labels[top_i])})"
        if res["fragility_read"] == "undetermined":
            # R1/R2 REFUSAL, FIXED 2026-08-16 (independent review): this branch used to skip straight
            # to the confident Prop-2.2/concentration_share/pivot_agreement INTERPRETATION regardless
            # of identification or resolution status. Verified directly on every non-identified cell
            # in the real 26-cell panel: the report simultaneously printed "cannot determine at this
            # sample size" in RESOLUTION and a confident "certifies non-significance at the 5% level"
            # in PIVOT for the SAME undetermined decision (see
            # tests/test_stage6_report.py::test_pivot_never_contradicts_resolution_on_undetermined_cases
            # for the gated regression check, not a specific count here that could drift as the real
            # panel's identification status shifts) -- exactly the R2 violation
            # This package's own R1/R2 discipline states in words ("any fragility-adjacent read
            # must refuse... never a number dressed as a verdict"). What's withheld here is the
            # INTERPRETIVE layer (the Prop 2.2 certification claim, concentration_share,
            # pivot_agreement) -- all three answer "is this pivot real/trustworthy", which is exactly
            # what "fragility-adjacent read" means; the period_txt fact above is not. An earlier
            # version of this fix also withheld period_txt, which broke
            # test_small_kstar_lists_named_periods -- caught by running the fix against the suite
            # before trusting it. Reuses resolution_report's own combined R1+R2 flag rather than
            # re-deriving it, so the two sections can never disagree again.
            if mcs_error is not None:
                reason = "MCS unavailable for this panel (see VERDICT above)"
            elif not ident:
                reason = "not identified (MCS size > 1)"
            else:
                reason = "observed edge below MDE at this sample size"
            lines.append(f"  {period_txt} -- fragility read UNDETERMINED ({reason}); see "
                         f"VERDICT/RESOLUTION above before treating this as evidence either way")
        else:
            cert = prop22_certifies(k, T=T)
            if cert is True:
                cert_txt = "certifies non-significance at the 5% level (Prop. 2.2, no simulation)"
            elif cert is False:
                cert_txt = "does not certify non-significance at the 5% level"
            else:
                cert_txt = "no constraint from Proposition 2.2 at this k*"
            lines.append(f"  {period_txt} -- {cert_txt}")
            cs = concentration_share(L, w)
            pa = pivot_agreement(L, w, seed=0)
            lines.append(f"  concentration_share: {_fmt_pct(cs)} of the margin comes from the single "
                         f"largest-contribution period")
            if pa is not None:
                lines.append(f"  pivot_agreement: {_fmt_pct(pa)} of subsamples name the same pivotal period")
    return "\n".join(lines)
