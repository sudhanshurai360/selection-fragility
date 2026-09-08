"""Round-7 domain review: LLM benchmark leaderboard evaluation (task-based scores across a fixed
benchmark suite -- e.g. MMLU-style subcategories -- NOT the pairwise-preference-vote approach of
arXiv 2508.11847, which this project already cites/differentiates from in paper/draft.md). This is
a genuinely new domain relative to rounds 1-6: candidates are LLM checkpoints/versions, periods are
individual benchmark task categories within one fixed suite, and loss is 1-accuracy (or an
equivalent scalar quality metric) per task.

All four scenarios below are grounded in real, well-documented leaderboard phenomena:
  1. A realistic 6-model x 40-task suite -- does the abstraction map cleanly at all.
  2. Low volatility: one model genuinely dominant everywhere -- must read as robust.
  3. High volatility: the "leaderboard illusion" pattern (two models with near-equal averages but
     each dominating a disjoint subset of task categories) -- must read as fragile, not falsely
     confident, per the project's own docs' concern about exactly this pattern.
  4. Benchmark contamination -- one model suspiciously perfect on a single task. This is the most
     nuanced finding of this round, given real code behavior, not assumed: k*/concentration_share
     surface the contaminated task ONLY when that task is actually decisive for the winner (4c
     below). If contamination merely inflates an already-losing model's position without flipping
     the champion (4a/4b), the pivotal-period diagnostic correctly points at whatever DOES decide
     the outcome instead -- which is the right behavior for a "why did the winner flip" tool, but
     means it is NOT a general-purpose single-task-contamination scanner on its own. Documented
     honestly as a real, precise capability boundary, not a bug.
"""
import numpy as np
import pytest

from selection_fragility import (
    LossPanel, report, decision_breakdown, concentration_share, model_confidence_set,
)


def _tasks(n, prefix="task"):
    return [f"{prefix}_{i:02d}" for i in range(n)]


# ---------------------------------------------------------------------------
# Scenario 1: realistic long-running fixed benchmark suite, 6 candidate models.
# ---------------------------------------------------------------------------
def test_llm_leaderboard_fixed_suite_identifies_the_consistently_best_model():
    rng = np.random.default_rng(42)
    n = 40
    base = rng.uniform(0.15, 0.45, n)
    L = {
        "model_alpha":   np.clip(base - 0.03 + rng.normal(0, 0.02, n), 0.01, 0.9),
        "model_beta":    np.clip(base - 0.01 + rng.normal(0, 0.02, n), 0.01, 0.9),
        "model_gamma":   np.clip(base + 0.00 + rng.normal(0, 0.02, n), 0.01, 0.9),
        "model_delta":   np.clip(base + 0.02 + rng.normal(0, 0.02, n), 0.01, 0.9),
        "model_epsilon": np.clip(base + 0.05 + rng.normal(0, 0.02, n), 0.01, 0.9),
        "model_zeta":    np.clip(base + 0.08 + rng.normal(0, 0.02, n), 0.01, 0.9),
    }
    panel = LossPanel.from_losses(L, labels=_tasks(n))
    out = report(panel)
    assert "identified (MCS size 1 of 6)" in out
    assert "model_alpha" in out.split("LEADERBOARD")[1].split("\n")[1]  # champion line names alpha
    k, opp, removed = decision_breakdown(L)
    assert 0 < k <= n  # a real, finite breakdown point, not degenerate either direction


# ---------------------------------------------------------------------------
# Scenario 2: LOW volatility -- one model dominant on nearly every task must read as robust
# (large k*, high resolved power), not fragile.
# ---------------------------------------------------------------------------
def test_llm_leaderboard_dominant_model_reads_as_robust_not_fragile():
    rng = np.random.default_rng(7)
    n = 35
    base = rng.uniform(0.2, 0.5, n)
    L = {
        "strong_model": np.clip(base - 0.15 + rng.normal(0, 0.01, n), 0.01, 0.95),
        "rival_a":      np.clip(base + rng.normal(0, 0.015, n), 0.01, 0.95),
        "rival_b":      np.clip(base + 0.02 + rng.normal(0, 0.015, n), 0.01, 0.95),
        "rival_c":      np.clip(base + 0.04 + rng.normal(0, 0.015, n), 0.01, 0.95),
    }
    panel = LossPanel.from_losses(L, labels=_tasks(n))
    out = report(panel)
    assert "identified (MCS size 1 of 4)" in out
    assert "resolved" in out and "cannot determine" not in out
    k, opp, removed = decision_breakdown(L)
    # a truly dominant model requires removing (close to) every favoring period to flip -- k* should
    # be large relative to T, the signature of genuine robustness, not a fragile few-period result.
    assert k >= int(0.8 * n)


# ---------------------------------------------------------------------------
# Scenario 3: HIGH volatility -- the "leaderboard illusion" pattern. Two models with near-equal
# overall averages but each dominating a disjoint subset of task categories (one strong on
# math/code, the other strong on knowledge/reasoning) must NOT be reported as falsely identified.
# ---------------------------------------------------------------------------
def test_llm_leaderboard_split_strength_illusion_reads_as_not_identified():
    rng = np.random.default_rng(7)
    n_math, n_know, n_other = 15, 15, 5
    base_math = rng.uniform(0.25, 0.5, n_math)
    base_know = rng.uniform(0.25, 0.5, n_know)
    base_other = rng.uniform(0.3, 0.4, n_other)

    model_math = np.concatenate([
        base_math - 0.12 + rng.normal(0, 0.02, n_math),
        base_know + 0.12 + rng.normal(0, 0.02, n_know),
        base_other + rng.normal(0, 0.02, n_other),
    ])
    model_know = np.concatenate([
        base_math + 0.12 + rng.normal(0, 0.02, n_math),
        base_know - 0.12 + rng.normal(0, 0.02, n_know),
        base_other + rng.normal(0, 0.02, n_other),
    ])
    model_mid = np.concatenate([
        base_math + 0.02 + rng.normal(0, 0.02, n_math),
        base_know + 0.02 + rng.normal(0, 0.02, n_know),
        base_other + 0.05 + rng.normal(0, 0.02, n_other),
    ])
    labels = [f"math_{i}" for i in range(n_math)] + [f"know_{i}" for i in range(n_know)] + \
             [f"other_{i}" for i in range(n_other)]
    L = {
        "model_math": np.clip(model_math, 0.01, 0.95),
        "model_know": np.clip(model_know, 0.01, 0.95),
        "model_mid":  np.clip(model_mid, 0.01, 0.95),
    }
    # confirm this really is the "near-equal averages, opposite strengths" shape before asserting
    # anything about the tool's output -- a fixture that silently drifted wouldn't test this pattern.
    means = {k: v.mean() for k, v in L.items()}
    assert abs(means["model_math"] - means["model_know"]) < 0.01, "fixture drifted: not near-equal"

    panel = LossPanel.from_losses(L, labels=labels)
    out = report(panel)
    assert "not identified" in out, (
        "split-strength / leaderboard-illusion panel must NOT be falsely reported as identified"
    )
    surv, p = model_confidence_set(L, seed=0)
    assert len(surv) >= 2, "near-equal split-strength models should not collapse to a lone survivor"


# ---------------------------------------------------------------------------
# Scenario 4: benchmark contamination. Three sub-cases, precisely characterizing when the
# concentration/pivot diagnostics do and do not surface a contaminated task.
# ---------------------------------------------------------------------------
def _contaminated_panel(gap, seed=3, n=30, contaminated_task=7):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.25, 0.5, n)
    model_a = np.clip(base + rng.normal(0, 0.02, n), 0.01, 0.95)
    model_c = np.clip(base + gap + rng.normal(0, 0.02, n), 0.01, 0.95)
    model_c[contaminated_task] = 0.0005
    return {"model_a": model_a, "model_c_contaminated": model_c}, n, contaminated_task


def test_llm_leaderboard_contamination_that_does_not_win_is_not_named_as_pivotal():
    """Contamination on a task that isn't decisive for the ranking (model_c still loses overall)
    must NOT be reported as the pivotal period -- the diagnostic correctly names whatever period
    actually drives the real winner decision instead. This is the precise, non-obvious boundary of
    what k*/pivot answers ("which periods would flip the winner"), not "which task looks
    suspicious" -- a real capability distinction worth locking in, not a bug."""
    L, n, contaminated_task = _contaminated_panel(gap=0.015)
    assert L["model_a"].mean() < L["model_c_contaminated"].mean(), "fixture must keep model_a champion"
    k, opp, removed = decision_breakdown(L)
    assert contaminated_task not in removed, (
        "contaminated task should NOT be named pivotal when it isn't what decides the winner"
    )


def test_llm_leaderboard_contamination_that_flips_the_winner_is_named_as_pivotal_with_high_concentration():
    """Contamination severe/positioned enough to actually flip the champion IS correctly surfaced:
    k* is minimal (1), the contaminated task is exactly the named pivotal period, and
    concentration_share is high -- together a real, usable contamination red flag."""
    L, n, contaminated_task = _contaminated_panel(gap=0.006)
    assert L["model_c_contaminated"].mean() < L["model_a"].mean(), (
        "fixture must make the contaminated model the (illegitimate) champion"
    )
    k, opp, removed = decision_breakdown(L)
    assert k == 1, f"a single suspiciously-perfect task flipping the winner should give k*=1, got {k}"
    assert removed == [contaminated_task], (
        f"pivotal period must be exactly the contaminated task {contaminated_task}, got {removed}"
    )
    cs = concentration_share(L)
    assert cs > 0.25, (
        f"concentration_share should flag the contaminated task as disproportionately responsible "
        f"for the ranking, got {cs:.3f}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
