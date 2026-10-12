"""v1.0.4 behavior locks. Three fixes, each pinned proactively (not just a regression-when-found test):

  Fix 1  fragility()'s `fragile` field no longer emits a 'retired/inverted' warning, because that label was
         wrong. VERIFIED by simulation (noise-rate > strong-winner-rate is the CORRECT direction for a
         fragility detector). This file re-runs a smaller version of that simulation every CI run so the
         direction can never silently flip.
  Fix 2  `screen` follows Table 1's fixed Fragile rule EXACTLY: stability < 0.60 OR (reversal AND k*/T < 0.25).
         v1.0.3 applied the k*/T cut on its own (no reversal conjunction) and over-flagged. Pinned two ways:
         (a) screen == the reconstructed Table 1 rule on many random panels; (b) on the REAL curated panel,
         screen == Table 1's published Fragile column for all 12 series (the authoritative check).
  Fix 3  mcs() documents the oversized first-step size (Section 4.2).

Plus a standing 12/12 reproduction guard: the released fragility()/model_confidence_set() reproduce Table 1's
winner, k*, and |MCS| (both elimination rules) for all 12 curated series -- the paper's own §12 claim.
"""
import warnings
import numpy as np
import pytest

from selection_fragility.fragility import fragility, exchangeable_benchmark
from selection_fragility import model_confidence_set
from selection_fragility.mcs import mcs as mcs_func


def _table1_rule(r):
    """Table 1's fixed Fragile rule, reconstructed from a fragility() result dict."""
    return bool(r["degenerate_margin"] or (r["winner_stability"] < 0.60)
                or (r["reversal"] and (r["k_over_T"] < 0.25)))


# --------------------------------------------------------------------------------------------------
# Fix 1 -- no warning, and the fire-rate DIRECTION is locked
# --------------------------------------------------------------------------------------------------
def test_fragility_emits_no_retired_or_inverted_warning():
    g = np.random.default_rng(0)
    L = {f"m{i}": g.normal(1.0, 0.3, 30) for i in range(6)}
    bench = exchangeable_benchmark(6, 30, n_rep=40, n_boot=150, seed=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = fragility(L, benchmark=bench, n_boot=200, seed=0)
    bad = [str(x.message) for x in w
           if any(t in str(x.message).lower() for t in ("retired", "inverted", "do not read", "do not act"))]
    assert bad == [], f"fragility() should emit no 'retired/inverted' warning; got: {bad}"
    assert res["fragile"] in (True, False)   # with a benchmark supplied it is a real bool, never None


def test_fragile_fire_rate_is_correct_direction():
    """A correct fragility detector fires MORE under pure noise than against a planted strong winner.
    This is the standing lock for Fix 1 -- if a future change inverts the field, this fails."""
    K, T, NB, N = 6, 30, 200, 120
    med, p05, p95 = exchangeable_benchmark(K, T, n_rep=120, n_boot=NB, seed=0)

    def rate(edge):
        hits = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for r in range(N):
                g = np.random.default_rng(10_000 + r)
                L = {f"m{i}": g.normal(1.0 - (edge if i == 0 else 0.0), 0.30, T) for i in range(K)}
                if fragility(L, benchmark=(med, p05, p95), n_boot=NB, seed=r)["fragile"]:
                    hits += 1
        return hits / N

    noise = rate(0.00)
    strong = rate(0.30)
    assert noise > strong, f"INVERTED: noise {noise:.3f} should exceed strong-winner {strong:.3f}"
    assert strong <= 0.04, f"a planted strong winner should almost never be flagged fragile; got {strong:.3f}"
    assert 0.01 <= noise <= 0.20, f"noise fire rate {noise:.3f} outside the near-nominal/oversized band [0.01, 0.20]"


# --------------------------------------------------------------------------------------------------
# Fix 2 -- screen follows Table 1's rule
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(40))
def test_screen_equals_table1_rule_random_panels(seed):
    g = np.random.default_rng(2_000 + seed)
    edge = (0.0, 0.15, 0.4)[seed % 3]
    K = 5
    L = {f"m{i}": g.normal(1.0 - (edge if i == 0 else 0.0), 0.3, 28) for i in range(K)}
    bench = exchangeable_benchmark(K, 28, n_rep=30, n_boot=120, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fragility(L, benchmark=bench, n_boot=150, seed=seed)
    assert r["screen"] == _table1_rule(r), (
        f"screen {r['screen']} != Table1 rule {_table1_rule(r)} "
        f"(stability={r['winner_stability']:.3f}, reversal={r['reversal']}, k/T={r['k_over_T']:.3f})")


@pytest.mark.parametrize("series", ["RSAFS", "BOPGEXP", "UNEMPLOY", "PAYEMS"])
def test_screen_behavior_change_on_real_flip_series(real_cells, series):
    """The exact behavior CHANGE from v1.0.3, on REAL frozen data: these four curated series have
    k*/T < 0.25, NO reversal, and stability >= 0.60, so v1.0.3's standalone k*/T cut flagged them
    (screen=True) while Table 1's Fragile column -- which v1.0.4 now matches -- says False."""
    L = real_cells[(series, "mase")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fragility(L, n_boot=1500, seed=0)
    assert r["reversal"] is False, f"{series}: expected no reversal"
    assert r["k_over_T"] < 0.25, f"{series}: expected k*/T < 0.25 (got {r['k_over_T']:.3f})"
    assert r["winner_stability"] >= 0.60, f"{series}: expected stability >= 0.60 (got {r['winner_stability']:.3f})"
    old_v103_screen = bool(r["degenerate_margin"] or (r["winner_stability"] < 0.60) or (r["k_over_T"] < 0.25))
    assert old_v103_screen is True, f"{series}: sanity -- v1.0.3 standalone rule should have fired here"
    assert r["screen"] is False, f"{series}: v1.0.4 screen must NOT fire (no reversal) -- matches Table 1 Fragile=no"


# --------------------------------------------------------------------------------------------------
# Fix 3 -- the first-step size is documented
# --------------------------------------------------------------------------------------------------
def test_mcs_docstring_documents_first_step_size():
    doc = mcs_func.__doc__ or ""
    assert "0.265" in doc and "0.367" in doc, "mcs() docstring must document the empirical first-step size"
    assert "lower bound" in doc.lower(), "docstring must note the non-identification count is a lower bound"


# --------------------------------------------------------------------------------------------------
# Standing 12/12 reproduction guard (paper §12) + Fix 2 on the REAL curated panel
# --------------------------------------------------------------------------------------------------
# series: (winner, k*, |MCS|, reversal, Fragile) -- Table 1, MASE, main text.
_TABLE1 = {
    "PI": ("ar1", 20, 1, False, False), "RSAFS": ("seasonal_naive", 2, 2, False, False),
    "BOPGEXP": ("ar1", 2, 3, False, False), "BOPGIMP": ("sarima", 1, 3, True, True),
    "PCE": ("ets", 2, 4, True, True), "UNEMPLOY": ("sarima", 4, 4, False, False),
    "BUSINV": ("ets", 1, 5, False, True), "DGORDER": ("ets", 1, 5, False, True),
    "HOUST": ("theta", 1, 5, True, True), "M2SL": ("ar1", 1, 5, True, True),
    "PAYEMS": ("ar1", 4, 5, False, False), "TCU": ("ar1", 1, 5, True, True),
}


@pytest.mark.parametrize("elimination", ["raw", "studentized"])
def test_curated_panel_reproduces_table1(real_cells, elimination):
    """fragility()/model_confidence_set() reproduce Table 1's winner, k*, |MCS|, reversal, and the
    (now Table-1-aligned) screen==Fragile for all 12 curated series -- the paper's §12 reproduction claim
    AND the authoritative check that Fix 2's screen matches the published Fragile column."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s, (win, kstar, mcs_n, rev, frag) in _TABLE1.items():
            L = real_cells[(s, "mase")]
            r = fragility(L, n_boot=1500, seed=0)
            surv, _p = model_confidence_set(L, elimination=elimination)
            assert r["pooled_winner"] == win, f"{s}: winner {r['pooled_winner']} != {win}"
            assert int(r["k_star"]) == kstar, f"{s}: k* {r['k_star']} != {kstar}"
            assert bool(r["reversal"]) == rev, f"{s}: reversal {r['reversal']} != {rev}"
            assert len(surv) == mcs_n, f"{s}: |MCS|({elimination}) {len(surv)} != {mcs_n}"
            assert bool(r["screen"]) == frag, f"{s}: screen {r['screen']} != Table1 Fragile {frag}"
