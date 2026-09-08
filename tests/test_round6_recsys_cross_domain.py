"""Round-6 cross-domain review: a recommender-systems ML engineer repurposing the tool for offline
ranking-algorithm evaluation (NDCG/recall/MRR-based losses), distinct from this project's own
forecasting domain. Covers two gaps the forecasting-focused suite doesn't specifically exercise:
bounded, densely-packed-near-a-boundary losses (ranking metrics live in [0,1], unlike often-unbounded
forecast percentage errors), and a cold-start-style single pivotal segment among many normal ones.
"""
import numpy as np
import pytest

from selection_fragility.fragility import pooled_winner, decision_breakdown, fragility
from selection_fragility.mcs import model_confidence_set
from selection_fragility.panel import LossPanel
from selection_fragility.report import report


def _bounded_panel(rng, centers, T, scale=0.01, floor=1e-4):
    """{model: array(T)} with values clipped into (0, 1) -- a 1-NDCG/1-recall-style loss shape."""
    return {
        name: np.clip(c + rng.normal(0, scale, T), floor, 1 - floor)
        for name, c in centers.items()
    }


def test_densely_packed_near_zero_bounded_losses_no_crash():
    """Several strong algorithms all scoring 1-NDCG in [0.02, 0.09] (near-zero-loss, near-perfect
    ranking quality) -- a realistic recsys shape the forecasting-domain fixtures don't cover."""
    rng = np.random.default_rng(7)
    L = _bounded_panel(rng, {"algo_a": 0.05, "algo_b": 0.045, "algo_c": 0.06}, T=20)
    k, opp, removed = decision_breakdown(L)
    assert 0 <= k <= 20
    surv, p = model_confidence_set(L, B=300, block=1, seed=0)
    assert 1 <= len(surv) <= 3
    frag = fragility(L)
    assert frag["pooled_winner"] in L
    # smoke: full report renders without raising on this bounded, tightly-clustered shape
    report(LossPanel.from_losses(L))


def test_extreme_near_zero_losses_at_numeric_floor_no_crash():
    """1-NDCG in [0.0003, 0.0007] -- NDCG > 0.999, an elite-tier ranking system regime. Losses this
    close to the numeric floor could plausibly interact badly with relative-percentage computations
    (MDE, significance_boundary) that divide by a near-zero denominator -- verify they don't."""
    rng = np.random.default_rng(7)
    L = _bounded_panel(rng, {"algo_a": 0.0005, "algo_b": 0.0004}, T=20, scale=0.0001, floor=1e-6)
    k, opp, removed = decision_breakdown(L)
    assert 0 <= k <= 20
    surv, p = model_confidence_set(L, B=300, block=1, seed=0)
    assert 1 <= len(surv) <= 2
    assert np.isfinite(p)


def test_cold_start_style_single_pivotal_segment_identified_as_driver():
    """Analogous to this project's own 'is the shock period doing all the work' framing (a 2020-style
    outlier), but for a different real mechanism: 3 algorithms evaluated across 12 normal weekly
    slices plus one cold-start segment.

    Non-obvious finding worth locking in: a shock period only shows up as pivotal when it
    DIFFERENTIALLY separates the two closest competitors, not merely when it's extreme in absolute
    terms. A first version of this fixture had both top algorithms degrade by roughly the SAME
    amount at cold-start (a realistic "MF-style collapse" for both) -- decision_breakdown correctly
    did NOT flag it as pivotal, because the MARGIN between the top two barely moved even though both
    scores cratered. Real cold-start pivotality requires one contender holding up better than its
    closest rival specifically at that segment (e.g. a hybrid/content-based fallback vs. a pure
    collaborative-filtering model with no signal for new users) -- that's what this fixture encodes."""
    rng = np.random.default_rng(42)
    segments = [f"week{i}" for i in range(1, 13)] + ["new_users_coldstart"]
    T = len(segments)
    coldstart_idx = segments.index("new_users_coldstart")
    base = {"popularity_baseline": 0.62, "als_mf": 0.48, "two_tower_nn": 0.45}
    # als_mf (pure CF) collapses hard at cold-start; two_tower_nn (has content features) degrades
    # too but much less -- a real, differential cold-start effect on the two closest competitors.
    coldstart = {"popularity_baseline": 0.70, "als_mf": 0.91, "two_tower_nn": 0.75}
    L = {}
    for a in base:
        vals = []
        for i, seg in enumerate(segments):
            c = coldstart[a] if i == coldstart_idx else base[a]
            vals.append(float(np.clip(c + rng.normal(0, 0.01), 1e-4, 1 - 1e-4)))
        L[a] = np.array(vals)
    k, opp, removed = decision_breakdown(L)
    assert 0 <= k <= T
    assert removed[0] == coldstart_idx, (
        f"cold-start segment (index {coldstart_idx}) should be the FIRST period removed (i.e. the "
        f"single largest margin contributor between the pooled winner and its binding opponent), "
        f"given it's constructed to differentially separate the two closest competitors; got "
        f"removed={removed}"
    )


def test_multi_metric_panels_can_disagree_on_pooled_winner():
    """Real recsys evaluation reports several metrics simultaneously and a model that wins one often
    loses another (the classic accuracy-vs-diversity tradeoff). The tool has no native cross-metric
    reconciliation -- this locks in that running it once per metric on the SAME algorithm panel can
    legitimately produce DIFFERENT pooled winners, which is correct behavior (not a bug) but worth a
    permanent regression guard so a future change doesn't silently start conflating metrics."""
    rng = np.random.default_rng(11)
    algos = ["item_cf", "two_tower_nn", "seq_transformer"]
    T = 15
    accuracy_losses = _bounded_panel(
        rng, {"item_cf": 0.50, "two_tower_nn": 0.40, "seq_transformer": 0.42}, T, scale=0.02)
    diversity_losses = _bounded_panel(
        rng, {"item_cf": 0.80, "two_tower_nn": 0.88, "seq_transformer": 0.70}, T, scale=0.02)
    accuracy_winner = pooled_winner(accuracy_losses)
    diversity_winner = pooled_winner(diversity_losses)
    assert accuracy_winner == "two_tower_nn"
    assert diversity_winner == "seq_transformer"
    assert accuracy_winner != diversity_winner, (
        "fixture should demonstrate a real cross-metric disagreement; if this now matches, the "
        "fixture's margins need revisiting, not this assertion"
    )
