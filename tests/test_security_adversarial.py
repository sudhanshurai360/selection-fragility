"""Round-5 adversarial security stress test (deliberately hostile input, not random fuzzing).

Distinct from round 4's hypothesis-based property fuzzing (random-but-valid input, looking for
accidental crashes) -- this file locks in DELIBERATE attacks an adversary or a hostile security
auditor would craft by hand: hand-forged malicious LossPanel files, boundary-exact numeric attacks,
and a static guard against dangerous deserialization patterns ever being introduced.

Every attack in this file was independently verified to already be blocked as of round 5 (2026-08-27)
-- these are regression locks, not bug reports, EXCEPT test_unbounded_bootstrap_count_is_a_known_gap,
which documents a real, currently-unfixed finding: `B` has no upper bound.
"""
import json
import os
import re
import tempfile
import warnings

import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.fragility import decision_breakdown, winner_stability, _MAX_ABS_LOSS
from selection_fragility.mcs import mcs, _MAX_ABS_LOSS as _MCS_MAX_ABS_LOSS


def _write_raw(payload_str):
    path = tempfile.mktemp(suffix=".json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload_str)
    return path


# ---------------------------------------------------------------------------
# Hand-forged hostile LossPanel files -- bypass .save(), attack .load() directly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_load_rejects_bare_json_extension_tokens(token):
    path = _write_raw(
        '{"losses": {"a": [1.1,2.2,3.3,4.4], "b": [1.5,2.1,3.9,4.2]}, '
        f'"weights": [1,1,1,{token}], "labels": ["1","2","3","4"], "labels_are_positional": false}}'
    )
    try:
        with pytest.raises(ValueError, match="not valid JSON"):
            LossPanel.load(path)
    finally:
        os.remove(path)


def test_load_rejects_negative_weight_in_hand_forged_file():
    path = _write_raw(json.dumps({
        "losses": {"a": [1.1, 2.3, 0.7, 4.4], "b": [2.2, 3.1, 4.9, 5.2]},
        "weights": [-1, 1, 1, 1], "labels": ["1", "2", "3", "4"], "labels_are_positional": False,
    }))
    try:
        with pytest.raises(ValueError, match="negative"):
            LossPanel.load(path)
    finally:
        os.remove(path)


def test_load_rejects_oversized_weight_vector_in_hand_forged_file():
    """A file claiming 1,000,000 weights for a 2-period panel -- a shape-mismatch attack that must
    be caught before any downstream code assumes weights/loss arrays are consistently sized."""
    path = _write_raw(json.dumps({
        "losses": {"a": [1.1, 2.3], "b": [2.2, 3.1]},
        "weights": [1] * 1_000_000, "labels": ["1", "2"], "labels_are_positional": False,
    }))
    try:
        with pytest.raises(ValueError, match="length"):
            LossPanel.load(path)
    finally:
        os.remove(path)


def test_load_rejects_oversized_labels_in_hand_forged_file():
    path = _write_raw(json.dumps({
        "losses": {"a": [1.1, 2.3, 0.7, 4.4], "b": [2.2, 3.1, 4.9, 5.2]},
        "weights": [1, 1, 1, 1], "labels": ["1", "2", "3", "4", "5", "6"], "labels_are_positional": False,
    }))
    try:
        with pytest.raises(ValueError, match="length"):
            LossPanel.load(path)
    finally:
        os.remove(path)


def test_load_rejects_truncated_json():
    path = _write_raw('{"losses": {"a": [1,2,')
    try:
        with pytest.raises(json.JSONDecodeError):
            LossPanel.load(path)
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# Deliberately crafted (not random) numeric boundary attacks.
# ---------------------------------------------------------------------------

def test_breakdown_number_accepts_exactly_at_magnitude_boundary():
    T = 20
    L = {"a": np.full(T, _MAX_ABS_LOSS), "b": np.full(T, _MAX_ABS_LOSS * 0.99)}
    k, opp, removed = decision_breakdown(L, np.ones(T))
    assert k >= 0  # must not raise/crash exactly at the documented boundary


def test_breakdown_number_rejects_just_over_magnitude_boundary():
    T = 20
    L = {"a": np.full(T, _MAX_ABS_LOSS * 1.0000001), "b": np.full(T, _MAX_ABS_LOSS * 0.99)}
    with pytest.raises(ValueError, match="exceeds"):
        decision_breakdown(L, np.ones(T))


def test_mcs_own_magnitude_guard_rejects_overflow():
    """ROUND-7 MUTATION-TESTING GAP (2026-08-27): `mcs.py`'s own `_MAX_ABS_LOSS` overflow guard
    (mcs.py, inside mcs() itself) is a SEPARATE check from fragility.py's `_MAX_ABS_LOSS` guard
    exercised by the two tests above -- decision_breakdown()'s guard does not protect a direct
    mcs() call, and nothing in the suite called mcs() with an overflowing panel. Confirmed by
    actually disabling mcs.py's guard and re-running the full suite: 0 failures. This is exactly
    the scenario the guard's own docstring warns about -- float64 silently overflows the bootstrap
    se computation and can crown an astronomical-magnitude model a spurious MCS survivor depending
    on column order (the round-1 bug this guard was originally added to fix)."""
    T = 10
    L = np.column_stack([
        np.full(T, 1e200),
        np.random.default_rng(0).normal(1.0, 0.2, T),
    ])
    with pytest.raises(ValueError, match=r"(?i)exceeds"):
        mcs(L, B=50, block=1, seed=0)


def test_mcs_own_magnitude_guard_accepts_at_boundary():
    T = 10
    L = np.column_stack([
        np.full(T, _MCS_MAX_ABS_LOSS),
        np.random.default_rng(0).normal(1.0, 0.2, T),
    ])
    surv, p = mcs(L, B=50, block=1, seed=0)
    assert len(surv) >= 1  # must not raise/crash exactly at the documented boundary


def test_breakdown_number_survives_catastrophic_cancellation_weights():
    """Alternating 1e15/1e-15 weights are engineered to maximize float64 cancellation error in the
    weighted margin sum -- must not crash or hang, even if the numeric answer has reduced precision."""
    rng = np.random.default_rng(0)
    w = np.array([1e15, 1e-15] * 10)
    la = rng.normal(1, 0.1, 20)
    lb = rng.normal(1.001, 0.1, 20)
    k, opp, removed = decision_breakdown({"a": la, "b": lb}, w)
    assert 0 <= k <= 20


def test_winner_stability_rejects_degenerate_block_equal_T():
    rng = np.random.default_rng(0)
    T = 10
    L = {"a": rng.normal(1, 0.1, T), "b": rng.normal(1.05, 0.1, T)}
    with pytest.raises(ValueError, match="rotation"):
        winner_stability(L, block=T)


# ---------------------------------------------------------------------------
# FIXED (round 5, same day as the finding above): bootstrap resample count is now capped.
# ---------------------------------------------------------------------------

def test_bootstrap_count_ceiling_enforced():
    """Round-5 adversarial security review found B (bootstrap resample count) had no upper-bound
    validation -- B=1,000,000 on a small (T=500, K=5) panel ran for 25+ seconds with no cap.
    mcs.py now enforces `_MAX_BOOTSTRAP_B = 100_000` (50x the documented default of 2000). This test
    supersedes `test_unbounded_bootstrap_count_is_a_known_gap` per that test's own stated intent
    ("if a future fix adds validation, this test should start failing and its assertion should flip
    to pytest.raises, not be deleted") -- the gap is now closed, not just documented."""
    import inspect
    from selection_fragility.mcs import model_confidence_set, _MAX_BOOTSTRAP_B
    sig = inspect.signature(model_confidence_set)
    assert sig.parameters["B"].default == 2000
    assert _MAX_BOOTSTRAP_B == 100_000
    rng = np.random.default_rng(0)
    T, K = 20, 3
    L = {f"m{i}": rng.normal(1, 0.1, T) for i in range(K)}
    # at/under the ceiling: still works (this is the value round 5 confirmed completes fine)
    surv, p = model_confidence_set(L, B=50_000)
    assert len(surv) >= 1
    # over the ceiling: rejected immediately, not run to completion
    with pytest.raises(ValueError, match="bootstrap-resample ceiling"):
        model_confidence_set(L, B=_MAX_BOOTSTRAP_B + 1)
    # the originally-reported pathological value is rejected instantly, not after 25+ seconds
    import time
    t0 = time.time()
    with pytest.raises(ValueError, match="bootstrap-resample ceiling"):
        model_confidence_set(L, B=1_000_000)
    assert time.time() - t0 < 1.0, "B=1,000,000 should be rejected before any bootstrap work starts"


def test_large_T_times_B_warns():
    """FIXED 2026-09-10 (round-6 stress-review, security_resource_exhaustion lens): the B ceiling
    above notes "cost scales with T*B" but only ever bounded B -- T (period count) had no
    corresponding guard anywhere in the package, unlike K, which got exactly this treatment in an
    earlier round. mcs()/model_confidence_set() are public, callable directly on a raw array with no
    LossPanel gate in between, so an ordinary-looking large-T input reached this cost (measured:
    minutes and multiple GB at T=200,000, B=2,000 -- still the DEFAULT B) with zero warning."""
    from selection_fragility.mcs import mcs

    # normal usage: no warning
    rng = np.random.default_rng(0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mcs(rng.normal(size=(50, 5)), B=2000)
    assert not any("T=" in str(x.message) and "bootstrap resamples" in str(x.message) for x in w)

    # T*B past the threshold: warns, does not raise (large-but-real T is a supported use case)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mcs(rng.normal(size=(15000, 5)), B=2000)
    assert any("bootstrap resamples" in str(x.message) for x in w)


def test_wrong_type_L_and_w_give_clear_typeerror_not_raw_internal_error():
    """FIXED 2026-09-10 (round-6 stress-review, error_message_quality lens): the raw-array tier
    (pooled_winner, decision_breakdown, fragility(), resolution_report and siblings, identified()/
    mcs_size()) had no duck-type check on L and no wrapped conversion for w -- a plausible mistake
    (passing a list/string instead of a {model: array} dict, or a dict/bad-string for weights)
    escaped as a raw AttributeError/TypeError/ValueError from deep inside numpy or Python, naming
    neither the argument nor the calling function. Worst case: identified([1,2,3]) did not raise
    cleanly at all -- `sorted(L)`/`L[m]` silently reinterpreted the list's VALUES as dict keys,
    producing a message that falsely implied a real model named '1' was passed."""
    import selection_fragility as sf

    L = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([2.0, 3.0, 4.0])}

    for bad_L in ("xyz", [1, 2, 3], None, 42):
        with pytest.raises(TypeError, match=r"(?i)L must be a"):
            sf.resolution_report(bad_L)
        with pytest.raises(TypeError, match=r"(?i)L must be a"):
            sf.pooled_winner(bad_L)
    with pytest.raises(TypeError, match=r"(?i)L must be a"):
        sf.identified([1, 2, 3])   # the actively-misleading case: sorted(L)/L[m] used to reinterpret
                                     # list VALUES as dict keys instead of raising cleanly

    for bad_w in ({"a": 1, "b": 2, "c": 3}, ["1", "2", "x"]):
        with pytest.raises(TypeError, match=r"(?i)weights must be"):
            sf.pooled_winner(L, w=bad_w)
    with pytest.raises(TypeError, match=r"(?i)weights must be"):
        sf.LossPanel.from_losses(L, weights={"a": 1, "b": 2})
    with pytest.raises(TypeError, match=r"(?i)weights must be.*string"):
        sf.LossPanel.from_losses(L, weights="count")


# ---------------------------------------------------------------------------
# Static guard: no dangerous deserialization/execution pattern should ever be introduced.
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS = [
    r"\bpickle\b", r"\beval\s*\(", r"\bexec\s*\(", r"yaml\.load\s*\((?!.*Loader=)",
    r"\bsubprocess\b", r"os\.system\s*\(", r"__import__\s*\(",
]


def test_no_dangerous_deserialization_or_execution_patterns():
    """Static guard: as of round 5, this package's source contains zero pickle/eval/exec/subprocess/
    os.system/unsafe-yaml usage -- LossPanel.save()/.load() is plain JSON. If a future change
    introduces any of these on data that could originate outside the caller's own trusted Python
    process (a loaded file, a network response), that's a real security regression this test exists
    to catch before it ships."""
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "selection_fragility")
    hits = []
    for root, _dirs, files in os.walk(src_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for pat in _DANGEROUS_PATTERNS:
                for m in re.finditer(pat, text):
                    hits.append(f"{path}: {pat!r} matched {m.group(0)!r}")
    assert not hits, "dangerous pattern(s) found:\n" + "\n".join(hits)


# ---------------------------------------------------------------------------
# 10-agent code-review pass (2026-09-02): scaling/DoS-shaped gaps found beyond mcs.py's own B cap.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mcs_native_worst_case_elimination_cost_grows_faster_than_quadratic():
    """KNOWN GAP, documented not fixed here: mcs.py's native mcs()/model_confidence_set() has no
    cost warning or K ceiling (only B is capped, see test_bootstrap_count_ceiling_enforced above).
    In the realistic worst case (a genuinely non-identified panel that forces near-complete
    elimination, one model dropped per round), cost is empirically O(B*K^3): measured K=50->400
    (doubling 3x) gives ~470x wall-clock growth, not the ~64x quadratic-in-K growth a caller might
    expect from "a bootstrap MCS". Extrapolated to this project's own real K~2949 LLM leaderboard
    (see paper/note.md) at the documented default B=2000, this is an estimated multi-day silent
    hang with zero warning -- directly relevant since mcs.py's own module docstring recommends
    calling it as an independent cross-check on exactly this kind of large panel. This test only
    locks in the scaling SHAPE (superquadratic) on a fast, small K so it stays cheap in CI; it does
    NOT assert a ceiling exists, since none does today -- if a future fix adds a K/cost guard
    (mirroring _MAX_BOOTSTRAP_B), revisit this test's ratio assertion alongside it."""
    import time

    from selection_fragility.mcs import model_confidence_set

    rng = np.random.default_rng(0)
    T = 30

    def timed(K):
        means = np.linspace(0, 20, K)
        L = {f"m{j}": means[j] + rng.normal(0, 0.05, T) for j in range(K)}
        t0 = time.time()
        surv, p = model_confidence_set(L, B=50, block=3, seed=0)
        return time.time() - t0, len(surv)

    t_small, surv_small = timed(40)
    t_large, surv_large = timed(80)   # 2x K
    assert surv_small == 1 and surv_large == 1, "fixture must force near-complete elimination"
    ratio = t_large / max(t_small, 1e-6)
    assert ratio > 5.0, (
        f"expected superquadratic (~O(K^3)) growth doubling K, got {ratio:.1f}x "
        f"({t_small:.3f}s -> {t_large:.3f}s) -- if this now passes near 4x, the scaling gap "
        f"documented here may have been fixed; update this test's docstring accordingly."
    )


def test_winner_stability_has_no_bootstrap_count_ceiling_unlike_mcs_B():
    """KNOWN GAP, documented not fixed here: round 5's adversarial security review found and fixed
    exactly this class of bug for mcs.py's `B` (see test_bootstrap_count_ceiling_enforced above).
    The identical "innocuous-looking large count hangs the process, unbounded" shape remains open
    in winner_stability/exchangeable_benchmark/fragility() (all via winner_stability) and in
    pivot_agreement/compare()'s n_perm. Demonstrates the absence of an upfront rejection for a
    value comfortably past mcs.py's own chosen ceiling, on a fast panel so it stays cheap; does NOT
    assert this SHOULD be rejected (that's the fix this documents the need for)."""
    from selection_fragility.fragility import winner_stability
    from selection_fragility.mcs import _MAX_BOOTSTRAP_B, model_confidence_set

    T = 5
    L = {"a": np.array([1.0, 2.0, 1.5, 1.2, 1.8]), "b": np.array([1.1, 1.9, 1.6, 1.3, 1.7])}
    with pytest.raises(ValueError, match="bootstrap-resample ceiling"):
        model_confidence_set({"a": L["a"], "b": L["b"]}, B=_MAX_BOOTSTRAP_B + 1)
    # ...but the sibling function accepts the same order-of-magnitude count with no check at all:
    stability, freq = winner_stability(L, n_boot=_MAX_BOOTSTRAP_B + 50_000, block=1)
    assert 0.0 <= stability <= 1.0   # it "worked" -- silently, with no cost warning ever given


def test_compare_rejects_negative_n_perm_instead_of_fabricating_churn_base_rate():
    from selection_fragility import LossPanel, compare

    T = 20
    rng = np.random.default_rng(0)
    Lprev = {"a": rng.normal(1, 0.1, T), "b": rng.normal(1.05, 0.1, T)}
    Lcurr = {"a": rng.normal(1, 0.1, T), "b": rng.normal(1.05, 0.1, T)}
    prev = LossPanel.from_losses(Lprev)
    curr = LossPanel.from_losses(Lcurr)
    with pytest.raises(ValueError):
        compare(prev, curr, n_perm=-5)


# ---- Round-2 10-agent review (2026-09-02): mixed-magnitude panel attacks --------------------

def test_pooled_winner_is_argmin_under_mixed_magnitude_panel():
    """CONFIRMED CRITICAL (2026-09-02, round-2 review): pooled_winner's degenerate-tie floor used a
    WHOLE-PANEL loss scale (mean |loss| over every model), so one large-magnitude column inflated the
    tie tolerance for every unrelated pair and a genuinely worse model that sorted earlier was
    returned as the winner. Reproduced exactly as below: z_best is strictly better in ALL 12 periods
    and was still not the reported winner. Same whole-panel-vs-per-pair defect already fixed twice in
    this package (fragility()'s concentration floor 2026-08-27, identify._run_mcs's jitter
    2026-08-26); pooled_winner shipped with the old form because its tests only used single-magnitude
    panels."""
    from selection_fragility.fragility import pooled_winner

    noise = np.random.default_rng(1).normal(0, 0.01, 12)
    L = {"z_best": 1.000 + noise, "a_worse": 1.001 + noise, "dollars": np.full(12, 5e9)}
    assert np.all(L["z_best"] < L["a_worse"]), "fixture broken: z_best must win every period"
    means = {m: float(np.mean(v)) for m, v in L.items()}
    assert pooled_winner(L, np.ones(12)) == min(means, key=means.get) == "z_best"


def test_pooled_winner_tie_tolerance_is_per_pair_not_whole_panel():
    """The invariant the fix establishes: adding an unrelated large-magnitude model must never change
    which of two OTHER models is reported as the winner."""
    from selection_fragility.fragility import pooled_winner

    noise = np.random.default_rng(7).normal(0, 0.01, 20)
    base = {"z_best": 1.000 + noise, "a_worse": 1.002 + noise}
    w = np.ones(20)
    for outlier in (1e2, 1e6, 1e10, 1e12, 1e20):
        contaminated = dict(base, dollars=np.full(20, outlier))
        assert pooled_winner(contaminated, w) == pooled_winner(base, w) == "z_best", (
            f"an unrelated {outlier:g}-magnitude column changed the winner between two other models"
        )


def test_pooled_winner_still_breaks_genuine_weight_scale_ties_by_name():
    """REGRESSION GUARD for the ORIGINAL defect the floor was added to fix (2026-08-27): an exactly
    tied pair must resolve to the same name at every weight scale, since np.average's sum(w*x)/sum(w)
    is mathematically but not float64-invariant to w's magnitude. The 2026-09-02 per-pair narrowing
    must not have reopened this."""
    from selection_fragility.fragility import pooled_winner

    L = {"m0": np.array([0.0, 0.0, 2.0, 2.125]), "m1": np.array([0.0, 0.0, 0.0, 4.125])}
    winners = {pooled_winner(L, np.full(4, s)) for s in (1e-6, 1e-3, 1.0, 1e3, 1e6, 1e12)}
    assert winners == {"m0"}, f"weight scale changed the reported winner: {winners}"


def test_observed_edge_and_mcb_bound_are_never_negative_on_mixed_magnitude_panels():
    """Downstream consequence of the same bug: because the reported champion was not the pooled
    argmin, resolution_report() produced a NEGATIVE observed_edge and mcb_bound -- directly violating
    mcb_bound's own documented invariant that it is 'by construction never tighter than the observed
    point-estimate edge, since the champion's mean is the global minimum by definition'."""
    from selection_fragility.resolution import resolution_report

    rng = np.random.default_rng(3)
    for outlier in (1e6, 1e9, 1e12):
        noise = rng.normal(0, 0.01, 16)
        L = {"z_best": 1.000 + noise, "a_worse": 1.001 + noise, "big": np.full(16, outlier)}
        rep = resolution_report(L, np.ones(16))
        assert rep["observed_edge"] >= 0.0, f"negative observed_edge at outlier={outlier:g}: {rep}"
        if rep.get("mcb_bound") is not None:
            assert rep["mcb_bound"] >= rep["observed_edge"] - 1e-12


def test_identify_shrinks_block_at_least_T_instead_of_faking_point_identification():
    """CONFIRMED SEVERE (2026-09-02): mcs.py shrinks the block when T <= block and
    fragility.winner_stability RAISES on block >= T, but identify._run_mcs -- the recommended engine,
    and the one report() uses -- had neither guard, with block_size an unvalidated public kwarg.
    Measured false 'point-identified' rate on PURE NOISE (K=3 iid, alpha=0.10, 120 trials): T=3 at
    the DEFAULT block_size=3 was 0.650, T=4/block=6 0.908, T=6/block=6 0.983, against 0.03-0.17 at
    block=1. Post-fix all block>=T cells sit near the block=1 baseline."""
    import warnings as _w
    from selection_fragility.identify import mcs_size

    rng = np.random.default_rng(0)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        rate = sum(mcs_size({f"m{j}": rng.normal(1, 0.2, 6) for j in range(3)},
                            alpha=0.10, block_size=6) == 1 for _ in range(120)) / 120
    assert rate < 0.40, f"block>=T still manufactures point-identification on noise: rate={rate}"

    with pytest.warns(UserWarning, match=r"(?i)shrink|block"):
        mcs_size({f"m{j}": rng.normal(1, 0.2, 6) for j in range(3)}, block_size=6)


def test_identify_rejects_nonpositive_reps_and_block_size():
    from selection_fragility.identify import mcs_size

    rng = np.random.default_rng(0)
    L = {f"m{j}": rng.normal(1, 0.2, 20) for j in range(3)}
    with pytest.raises(ValueError, match=r"(?i)reps"):
        mcs_size(L, reps=0)
    with pytest.raises(ValueError, match=r"(?i)block_size"):
        mcs_size(L, block_size=0)


def test_pivot_agreement_rejects_nonpositive_n_boot_and_bad_frac():
    """CONFIRMED (2026-09-02): n_boot<=0 and an out-of-range frac both silently returned None --
    indistinguishable from pivot_agreement's TWO documented None meanings (no strict winner;
    subsample too small). Same defect class hardened in winner_stability and _churn_base_rate the
    same day; this sibling was missed."""
    from selection_fragility.pivot import pivot_agreement

    rng = np.random.default_rng(0)
    L = {"a": rng.normal(1, 0.2, 30), "b": rng.normal(1.05, 0.2, 30)}
    w = np.ones(30)
    for kwargs in ({"n_boot": 0}, {"n_boot": -5}, {"frac": 5.0}, {"frac": -1.0}, {"frac": 1.0}):
        with pytest.raises(ValueError):
            pivot_agreement(L, w, **kwargs)
    assert 0.0 <= pivot_agreement(L, w) <= 1.0     # a valid call must still work


def test_d_max_over_median_is_finite_for_negative_losses():
    """CONFIRMED (2026-09-02): the guard was a SIGN test (`b > 0`) rather than a ZERO test, so any
    panel with a negative median difficulty returned inf even though max(d)/b is finite. Negative
    losses pass _validate_losses (finiteness only) and are routine: log scores, skill scores,
    negative log-likelihood. Reproduced: b=-5.175, max(d)=-3.958, true ratio 0.765, reported inf."""
    from selection_fragility.fragility import surprise_concentration

    rng = np.random.default_rng(0)
    L = {"a": rng.normal(-5.0, 0.2, 30), "b": rng.normal(-5.1, 0.2, 30)}
    out = surprise_concentration(L, np.ones(30))
    assert np.isfinite(out["d_max_over_median"]), out


def test_d_max_over_median_is_nan_for_mixed_sign_baseline_and_max():
    """CONFIRMED (2026-09-05, round-3 audit): the 2026-09-02 b!=0 fix above only handles the
    same-sign negative case correctly. Because max(d) >= b always holds, a negative b paired with a
    non-negative max(d) is reachable and the old fix silently returned a negative or zero ratio --
    not an error, but not a coherent "times normal" reading either. Reproduced directly:
    L={'a':[-10,-9,-8,5],'b':[-9,-8,-7,6]} -> d=[-10,-9,-8,5], b=-8.5, max(d)=5.0 (opposite signs).
    The fixed function must report this as NaN (undefined), not a negative number."""
    from selection_fragility.fragility import surprise_concentration

    L = {"a": [-10, -9, -8, 5], "b": [-9, -8, -7, 6]}
    out = surprise_concentration(L)
    assert np.isnan(out["d_max_over_median"]), out


def test_d_max_over_median_same_sign_negative_case_still_matches_reference_ratio():
    """Companion to the mixed-sign test above: confirm the fix did NOT regress the already-correct
    same-sign negative case from the 2026-09-02 fix (b=-5.175, max(d)=-3.958, true ratio 0.765)."""
    from selection_fragility.fragility import surprise_concentration

    rng = np.random.default_rng(0)
    L = {"a": rng.normal(-5.0, 0.2, 30), "b": rng.normal(-5.1, 0.2, 30)}
    out = surprise_concentration(L)
    d = np.vstack([np.asarray(L[m], float) for m in L]).min(axis=0)
    b = float(np.median(d))
    assert b < 0 and float(np.max(d)) < 0        # confirms this really is the same-sign-negative case
    assert out["d_max_over_median"] == pytest.approx(float(np.max(d)) / b)
