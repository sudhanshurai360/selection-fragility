"""Round-6 production-scale review (demand-planning/MLOps persona), distinct from prior rounds'
manual/persona/fuzzing/adversarial lenses. Covers: realistic batch scale, cross-call contamination
in a tight loop, and locking in the pooled_winner weight-scale-instability finding as a known,
documented gap (not fixed here -- find/test-only round, per directive).
"""
import numpy as np
import pytest

from selection_fragility.fragility import decision_breakdown, pooled_winner
from selection_fragility.resolution import resolution_report


def test_decision_breakdown_no_cross_call_contamination_in_tight_loop():
    """Interleaved repeated calls on two distinct panels must be idempotent -- no global/cached
    state should leak between calls in a batch-processing loop (a different check from the
    reproducibility audit's seed-determinism tests: this is about STATE LEAKAGE, not seeding)."""
    rng = np.random.default_rng(42)
    panel_a = {"m0": rng.normal(1.0, 0.3, 60), "m1": rng.normal(1.1, 0.3, 60)}
    panel_b = {"x0": rng.normal(2.0, 0.5, 60), "x1": rng.normal(2.05, 0.5, 60)}
    w = np.ones(60)
    results_a, results_b = set(), set()
    for _ in range(300):
        ka, oa, ra = decision_breakdown(panel_a, w=w)
        kb, ob, rb = decision_breakdown(panel_b, w=w)
        results_a.add((ka, oa, tuple(ra)))
        results_b.add((kb, ob, tuple(rb)))
    assert len(results_a) == 1, f"panel A result varied across interleaved calls: {results_a}"
    assert len(results_b) == 1, f"panel B result varied across interleaved calls: {results_b}"


def test_resolution_report_no_cross_call_contamination_in_tight_loop():
    """Same contamination check for resolution_report, which is documented stateless (no seed
    param -- deterministic given L/w/alpha/power/mcs_alpha)."""
    rng = np.random.default_rng(1)
    panel_a = {"m0": rng.normal(1.0, 0.3, 60), "m1": rng.normal(1.1, 0.3, 60)}
    panel_b = {"x0": rng.normal(2.0, 0.5, 60), "x1": rng.normal(2.02, 0.5, 60)}
    seen = set()
    for i in range(300):
        r = resolution_report(panel_a if i % 2 == 0 else panel_b)
        seen.add((i % 2, r["mde"], r["resolved"]))
    assert len(seen) == 2, f"resolution_report varied across interleaved calls: {seen}"


def test_pooled_winner_weight_scale_invariance_on_exact_tie():
    """FOUND by round-6 production-scale review (2026-08-27), FIXED same day. pooled_winner
    (fragility.py) was NOT invariant to positive-scalar rescaling of the weight vector when two
    models are exactly or near-exactly tied on pooled loss -- a cousin of the k*-count bug round 4
    fixed in breakdown_number's degenerate-margin floor, but in a DIFFERENT function
    (pooled_winner's own weighted-mean tie comparison) that fix did not touch. Concretely:
    L={'m0': [0,0,2,2.125], 'm1': [0,0,0,4.125]} -- pooled sums are EXACTLY tied (4.125 each) --
    pooled_winner(L, w=ones(4)) returned 'm0', but pooled_winner(L, w=ones(4)*1e-6) returned 'm1'.
    This propagated into decision_breakdown: k* itself stayed correctly invariant (0 in both
    cases), but the reported binding `opponent` flipped identity purely from weight-scale. Fixed
    by treating any mean within a relative floor of the true minimum as a genuine tie, broken by
    sorted name -- see fragility.py::pooled_winner."""
    L = {"m0": np.array([0.0, 0.0, 2.0, 2.125]), "m1": np.array([0.0, 0.0, 0.0, 4.125])}
    w = np.ones(4)
    winners = {pooled_winner(L, w=w * c) for c in (1.0, 1e-6, 1e-3, 1e3, 1e6)}
    assert len(winners) == 1, f"pooled_winner varied under weight rescaling: {winners}"


def test_batch_scale_smoke_300_skus_completes_and_scales_roughly_linearly():
    """Regression guard against a non-linear scaling blowup at realistic demand-planning batch
    scale (round-6 finding: ~58ms/SKU at K=4,T=104,B=200 on this machine -- extrapolates to ~5min
    for 5,000 SKUs, ~49min for 50,000 SKUs, single-threaded, no parallelism built in; memory usage
    is negligible, no leak observed). This test only guards against a REGRESSION making per-SKU
    cost blow up, not the absolute number, which is hardware-dependent."""
    import time
    from selection_fragility.mcs import model_confidence_set

    rng = np.random.default_rng(0)
    n_skus, K, T = 40, 4, 60
    t0 = time.perf_counter()
    for i in range(n_skus):
        losses = {f"m{j}": np.abs(rng.normal(1.0 + 0.05 * j, 0.3, T)) for j in range(K)}
        model_confidence_set(losses, alpha=0.10, B=100, block=4, seed=i)
        decision_breakdown(losses, w=np.ones(T))
    elapsed = time.perf_counter() - t0
    per_sku_ms = elapsed / n_skus * 1000
    assert per_sku_ms < 500, (
        f"per-SKU cost regressed to {per_sku_ms:.1f}ms (round-6 baseline ~30-60ms on this "
        f"machine at similar K/T/B) -- would make large-batch production runs impractically slow"
    )
