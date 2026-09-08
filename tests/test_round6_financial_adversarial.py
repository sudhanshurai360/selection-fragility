"""Round-6 financial/quant-adversarial review (release/selection-fragility).

Persona: a professionally paranoid quantitative researcher (systematic-fund macro forecasting /
bank economics desk) stress-testing whether this tool's diagnostics are safe to base a real capital
or publication decision on. Distinct from prior rounds' fuzzing/persona work -- targets regime
breaks, multiple-comparisons/overfitting risk, and fat-tailed financial-shaped data specifically.
"""
import numpy as np
import pytest

from selection_fragility.fragility import pooled_winner, decision_breakdown, winner_stability
from selection_fragility.mcs import model_confidence_set
from selection_fragility.panel import LossPanel
from selection_fragility.report import report


# ---------------------------------------------------------------------------
# FOUND THIS ROUND, FIXED BY A SIBLING FORK IN THE SAME SHARED TREE (owning
# fragility.py's weighted-average tie-break): pooled_winner's tie-break
# ("ties resolve by sorted model name," fragility.py) was not actually
# invariant to positive-scalar weight rescaling on a genuine exact
# real-number tie -- `min(sorted(L), key=lambda m: float(np.average(...)))`
# compared raw floats with no tolerance, and floating-point rounding during
# the weighted-average computation could make one model's computed average a
# few ULPs below the other's at certain weight scales, overriding the
# intended alphabetical tie-break. Reproduced directly (not just via
# hypothesis): L={'m0':[0,0,2,2.125], 'm1':[0,0,0,4.125]} (exact tie, both
# sum to 4.125) gave pooled_winner='m0' at c in {1e-3,1,1e3,1e6} but
# pooled_winner='m1' at c=1e-6 -- purely from FP rounding in np.average at
# that scale, not from any real difference. LOW-MODERATE severity: k* stayed
# 0 in every case (the tool correctly signalled "no clear winner, it's a
# tie"), so this never produced a false-confidence result -- it was a naming-
# consistency defect on an already-flagged-degenerate panel, not silently-
# wrong output. Same general bug CLASS as the round-4-fixed
# decision_breakdown weight-scale bug (float64 near-tie sensitivity), in a
# different function (pooled_winner's tie comparison) that fix didn't cover.
# Now fixed at the source; flipped to verify per this test's own instruction.
def test_pooled_winner_tiebreak_is_scale_invariant_on_exact_tie():
    L = {"m0": np.array([0.0, 0.0, 2.0, 2.125]), "m1": np.array([0.0, 0.0, 0.0, 4.125])}
    assert L["m0"].sum() == L["m1"].sum()  # confirm this IS a genuine exact real-number tie
    winners = {c: pooled_winner(L, np.ones(4) * c) for c in (1e-6, 1e-3, 1.0, 1e3, 1e6)}
    assert len(set(winners.values())) == 1, (
        f"pooled_winner's tie-break should depend only on sorted model name, never on weight "
        f"scale. Observed: {winners}"
    )
    assert set(winners.values()) == {"m0"}, f"expected the sorted-name winner 'm0'; got {winners}"


# ---------------------------------------------------------------------------
# Regime-change adversarial tests: does pooling silently average over a
# structural break and report false confidence, or does it correctly flag
# the winner as fragile/not-identified?
# ---------------------------------------------------------------------------
def test_symmetric_regime_flip_reports_not_identified_and_low_stability():
    """Two models that each dominate exactly half the sample (a genuine regime flip, not noise)
    plus a mediocre-but-stable third model. A tool that silently pools over the break would
    confidently crown a winner; this one must not."""
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(0.5, 0.1, 20), rng.normal(2.0, 0.1, 20)])
    b = np.concatenate([rng.normal(2.0, 0.1, 20), rng.normal(0.5, 0.1, 20)])
    c = rng.normal(1.2, 0.3, 40)
    L = {"model_a": a, "model_b": b, "model_c": c}
    surv, _ = model_confidence_set(L, B=500, block=3, seed=0)
    assert len(surv) >= 2, (
        f"regime-flip panel (no consistent winner) reported as identified (MCS size {len(surv)}) "
        f"-- would silently mislead a user pooling over a structural break"
    )
    stability, freq = winner_stability(L, w=np.ones(40), seed=0, block=3, n_boot=300)
    champion = pooled_winner(L, np.ones(40))
    assert freq[champion] < 0.6, (
        f"champion '{champion}' reported unrealistically stable (win share {freq[champion]:.2f}) "
        f"despite a genuine 50/50 regime flip in the underlying data"
    )


def test_asymmetric_tail_risk_regime_pivot_names_the_crisis_periods():
    """A model that looks best 90% of the time (long calm regime) but catastrophically worse in a
    short crisis regime, vs. a model that's always merely 'safe.' The tool's job: name the exact
    periods responsible for the apparent winner, not just declare a champion. This is the specific
    signal a risk-paranoid quant needs -- confirmed working, locked in as a regression."""
    rng = np.random.default_rng(1)
    calm = np.concatenate([rng.normal(0.5, 0.05, 36), rng.normal(8.0, 1.0, 4)])
    safe = np.concatenate([rng.normal(0.6, 0.05, 36), rng.normal(0.7, 0.1, 4)])
    L = {"model_calm_but_fragile": calm, "model_always_safe": safe}
    k, opp, removed = decision_breakdown(L, w=np.ones(40))
    # the 4 crisis periods are indices 36-39; removing them should be exactly what flips the winner
    assert k <= 4, f"k*={k} -- should be small (<=4), the whole apparent edge lives in the 4 crisis periods"
    assert set(removed).issubset(set(range(36, 40))), (
        f"removed periods {sorted(removed)} should be a subset of the 4 injected crisis periods "
        f"(36-39) -- PIVOT should point straight at the tail-risk periods, not diffuse credit"
    )


# ---------------------------------------------------------------------------
# Multiple-comparisons / overfitting-from-many-candidates: k*/MCS should get
# LESS confident (lower k*, larger surviving set) as the number of candidate
# models grows, all else held equal -- the classic "we tried 20 variants"
# post-hoc-selection risk a quant is professionally paranoid about.
# ---------------------------------------------------------------------------
def test_more_candidates_reduces_confidence_not_increases_it():
    rng = np.random.default_rng(7)
    T = 60

    def make_panel(K, seed):
        r = np.random.default_rng(seed)
        L = {"best": r.normal(1.000, 0.15, T)}
        for i in range(K - 1):
            L[f"cand_{i}"] = r.normal(1.055, 0.15, T)
        return L

    L_small = make_panel(3, seed=7)
    L_large = make_panel(20, seed=7)  # same generative process, just more draws from it
    _, surv_frac_small = model_confidence_set(L_small, B=1000, block=3, seed=0)
    surv_small, _ = model_confidence_set(L_small, B=1000, block=3, seed=0)
    surv_large, _ = model_confidence_set(L_large, B=1000, block=3, seed=0)
    k_small, _, _ = decision_breakdown(L_small, w=np.ones(T))
    k_large, _, _ = decision_breakdown(L_large, w=np.ones(T))
    assert len(surv_large) / 20 >= len(surv_small) / 3, (
        f"more candidates (20) gave a SMALLER survivor fraction than fewer (3) -- backwards for "
        f"multiple-comparisons risk: {len(surv_large)}/20 vs {len(surv_small)}/3"
    )


# ---------------------------------------------------------------------------
# Fat-tailed / volatility-clustered (GARCH-shaped) financial data: no crash,
# no NaN/inf, and PIVOT correctly attributes an apparent winner driven by
# tail events to those specific events rather than diffusing credit.
# ---------------------------------------------------------------------------
def test_garch_fat_tailed_panel_no_crash_and_pivot_names_outliers():
    def garch_losses(mu, omega, alpha, beta, T, seed):
        r = np.random.default_rng(seed)
        h = omega / (1 - alpha - beta)
        losses = np.empty(T)
        for t in range(T):
            z = r.standard_t(df=4)
            losses[t] = abs(mu + np.sqrt(h) * z)
            h = omega + alpha * (np.sqrt(h) * z) ** 2 + beta * h
        return losses

    T = 250
    a = garch_losses(0.5, 0.02, 0.10, 0.85, T, seed=10)
    b = garch_losses(0.55, 0.015, 0.08, 0.88, T, seed=11)
    outlier_idx = [50, 120, 200]
    a[outlier_idx] *= 30
    L = {"model_a": a, "model_b": b}
    assert np.isfinite(a).all() and np.isfinite(b).all()

    panel = LossPanel.from_losses(L)
    out = report(panel)  # must not raise on fat-tailed/outlier-injected real-shaped data
    assert "VERDICT" in out

    k, opp, removed = decision_breakdown(L, w=np.ones(T))
    assert 0 <= k <= T
    # the injected outliers should dominate the pivotal-period story
    assert set(removed).issubset(set(outlier_idx)), (
        f"removed periods {sorted(removed)} should be drawn from the injected outlier periods "
        f"{outlier_idx} -- PIVOT should point at the tail events, not ordinary periods"
    )
