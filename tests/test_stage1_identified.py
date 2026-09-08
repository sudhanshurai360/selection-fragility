"""Stage 1 -- identified/mcs_size via arch delegation. release/TOOL_TEST_PLAN.md S1.1-S1.5.

Two of these test classes CAN run against current code (arch itself, and the existing
model_confidence_set()) -- these serve as the pre-registered ground truth the new wrapper must
match. The rest wait on the new `identified`/`mcs_size` convenience API (Stage 1 of the design doc).
"""
import warnings
import numpy as np
import pandas as pd
import pytest
from itertools import combinations

from selection_fragility import model_confidence_set

try:
    from arch.bootstrap import MCS as ArchMCS
    HAVE_ARCH = True
except ImportError:
    HAVE_ARCH = False

try:
    from selection_fragility import identified, mcs_size
    HAVE_NEW_API = True
except ImportError:
    HAVE_NEW_API = False


# ---- S1.1: the two guards arch was verified against, PRE-REGISTERED against arch directly --------
@pytest.mark.skipif(not HAVE_ARCH, reason="arch not installed")
class TestArchGuardsDirect:
    """These run TODAY, against arch itself, independent of our wrapper -- they are the ground
    truth the wrapper (once built) must reproduce exactly."""

    def test_deterministic_dominance(self):
        L = np.column_stack([np.full(30, 1.0), np.full(30, 6.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ArchMCS(L, size=0.10, reps=300, block_size=3, bootstrap="circular", seed=0)
            m.compute()
        assert list(m.included) == [0]

    def test_T_near_block_length(self):
        rng = np.random.default_rng(0)
        L = rng.normal(1, 0.3, (4, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ArchMCS(L, size=0.10, reps=300, block_size=3, bootstrap="circular", seed=0)
            m.compute()
        assert set(m.included) <= {0, 1, 2}   # no crash, a well-formed subset


@pytest.mark.skipif(not HAVE_NEW_API, reason="identified()/mcs_size() not implemented yet (Stage 1)")
class TestGuardsViaWrapper:
    def test_deterministic_dominance_via_wrapper(self):
        L = {"a": np.full(30, 1.0), "b": np.full(30, 6.0)}
        assert identified(L) is True
        assert mcs_size(L) == 1

    def test_T_near_block_length_via_wrapper(self):
        rng = np.random.default_rng(0)
        L = {f"m{i}": rng.normal(1, 0.3, 4) for i in range(3)}
        mcs_size(L)   # must not raise

    def test_large_magnitude_model_does_not_corrupt_small_scale_tiebreak(self):
        # FIXED (2026-08-26): the internal exact-tie jitter's std used to be scaled from
        # `mean(abs(WHOLE PANEL))`, so one model with a legitimately large-but-valid magnitude
        # (e.g. raw-dollar loss alongside a normalized metric -- explicitly the scenario this
        # module's own docstring warns about) inflated the jitter applied to EVERY column,
        # including two small-scale models with a real, decisive edge between them. Reproduced
        # directly: at a magnitude ratio of 1e10 (still far under the 1e100 cap), the panel-wide
        # formula flipped the correctly-identified small-scale winner; a per-column jitter scale
        # does not. This test locks in that the reported winner tracks the TRUE small-scale
        # winner regardless of which one it is, with a large model sharing the panel.
        rng = np.random.default_rng(1)
        T = 40
        noise = rng.normal(0, 1e-6, T)
        big_model = 1e11 + rng.normal(0, 1e7, T)   # large-but-valid magnitude, decisively worst

        from selection_fragility.identify import _run_mcs

        L_b_wins = {"small_a": 1.0 + 3e-5 + noise, "small_b": 1.0 + noise, "big_model": big_model}
        assert mcs_size(L_b_wins) == 1
        assert identified(L_b_wins) is True
        assert _run_mcs(L_b_wins, alpha=0.10, seed=0).included == ["small_b"]

        L_a_wins = {"small_a": 1.0 + noise, "small_b": 1.0 + 3e-5 + noise, "big_model": big_model}
        assert mcs_size(L_a_wins) == 1
        assert identified(L_a_wins) is True
        assert _run_mcs(L_a_wins, alpha=0.10, seed=0).included == ["small_a"]

    def test_byte_identical_pair_never_split_inside_larger_panel(self):
        """CONFIRMED REGRESSION (Phase-2 holistic review, finding A7, 2026-08-27). A byte-identical
        PAIR inside a larger, otherwise-distinct panel could get OPPOSITE MCS verdicts -- verified
        directly before the fix: 16/200 seeds on this exact construction split the tied pair (one
        included, one excluded), which is impossible on the actual statistics (two byte-identical
        loss series have an EXACTLY ZERO difference; nothing in the data can distinguish them).
        Root cause: the old jitter was one shared `rng.normal(size=M.shape)` draw filled
        positionally, so two identical columns still received two different noise draws. Fixed by
        seeding each column's jitter from that column's own data content (see _col_seed) -- two
        byte-identical columns now draw the identical jitter and stay tied with each other."""
        from selection_fragility.identify import _run_mcs
        split_count = 0
        trials = 200
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            T = 30
            good = {f"good{i}": rng.normal(1.0, 0.2, T) for i in range(2)}
            identical_vals = rng.normal(1.0, 0.2, T)
            L = {**good, "tie_a": identical_vals.copy(), "tie_b": identical_vals.copy(),
                 "bad": rng.normal(3.0, 0.2, T)}
            res = _run_mcs(L)
            a_in, b_in = "tie_a" in res.included, "tie_b" in res.included
            if a_in != b_in:
                split_count += 1
        assert split_count == 0, f"{split_count}/{trials} trials split a byte-identical pair's verdict"

    def test_rename_invariance_survives_a_near_machine_precision_tie(self):
        """CONFIRMED REGRESSION (Phase-2 holistic review, finding A7, 2026-08-27). `_to_frame`'s
        `sorted(L)` (added for a DIFFERENT, earlier bug) only guarantees INSERTION-order
        invariance for a FIXED set of names -- renaming a model changes its alphabetical rank,
        which changes its position in the sorted column order, which (with the old
        position-filled jitter) changed which noise draw it received. Verified directly before the
        fix: renaming a single model (same data, new name) in a 6-model, T=24 panel with one
        manufactured ~1e-13-relative near-tie flipped the MCS verdict in 186/200 trials. Fixed by
        the same _col_seed change as the byte-identical-pair test above: seeding by data content
        makes the draw genuinely identity-invariant."""
        from selection_fragility.identify import _run_mcs
        disagree = 0
        trials = 200
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            K, T = 6, 24
            base = rng.normal(1.0, 0.3, (T, K))
            base[:, 1] = base[:, 0] * (1 + 1e-13)   # manufactured near-machine-precision tie
            L = {f"m{i}": base[:, i] for i in range(K)}
            set1 = set(_run_mcs(L).included)

            L2 = dict(L)
            L2["z0"] = L2.pop("m0")                 # rename, same data
            set2 = {("m0" if x == "z0" else x) for x in _run_mcs(L2).included}
            if set1 != set2:
                disagree += 1
        assert disagree == 0, f"{disagree}/{trials} trials changed verdict purely from a rename"


# ---- S1.2: K/T sweep ------------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_NEW_API, reason="identified()/mcs_size() not implemented yet (Stage 1)")
@pytest.mark.parametrize("K,T", [(2, 2), (2, 8), (2, 30), (6, 32), (13, 30), (20, 30), (100, 1000)])
def test_KT_sweep_no_crash(K, T):
    rng = np.random.default_rng(0)
    L = {f"m{i}": rng.normal(1, 0.3, T) for i in range(K)}
    r = identified(L)
    assert isinstance(r, (bool, np.bool_))
    assert 1 <= mcs_size(L) <= K


# ---- S1.3: structural properties -------------------------------------------------------------------
# These test model_confidence_set(), which already exists -- they run TODAY and serve as the
# pre-registered ground truth Stage 1's arch-backed wrapper must preserve. Only the one test that
# names identified()/mcs_size() directly is skipped until Stage 1 ships.
class TestStructuralProperties:
    def test_alpha_nesting(self, random_panel):
        """A LARGER alpha is a LOWER bar to reject equal-predictive-ability, so it eliminates more
        readily -> a larger alpha gives a SMALLER-OR-EQUAL survivor set. Direction verified directly
        against model_confidence_set() before writing this assertion (seed=6, K=6, T=30: alpha=0.05
        and 0.10 both give size 6, alpha=0.25 gives size 5) -- do not flip this without re-verifying,
        the flipped version silently passes far more seeds than it should and is the wrong claim."""
        for seed in range(60):
            L = random_panel(seed=seed, T=30, K=6)
            from selection_fragility import model_confidence_set as mcs_fn
            s05, _ = mcs_fn(L, alpha=0.05, seed=0)
            s10, _ = mcs_fn(L, alpha=0.10, seed=0)
            s25, _ = mcs_fn(L, alpha=0.25, seed=0)
            assert set(s25) <= set(s10) <= set(s05), f"nesting violated at seed={seed}"

    def test_best_mean_always_survives(self, random_panel):
        for seed in range(400):
            L = random_panel(seed=seed, T=30, K=6)
            best = min(L, key=lambda m: L[m].mean())
            surv, _ = model_confidence_set(L, alpha=0.10)
            assert best in surv, f"best model excluded at seed={seed}"

    def test_column_permutation_invariance(self, random_panel):
        L = random_panel(seed=0, T=30, K=6)
        base_surv, _ = model_confidence_set(L, alpha=0.10, seed=0)
        keys = list(L)
        rng = np.random.default_rng(1)
        for _ in range(40):
            shuffled_keys = rng.permutation(keys)
            L2 = {k: L[k] for k in shuffled_keys}
            surv2, _ = model_confidence_set(L2, alpha=0.10, seed=0)
            assert set(surv2) == set(base_surv)

    def test_dict_key_order_invariance(self, random_panel):
        L = random_panel(seed=0, T=30, K=4)
        items = list(L.items())
        surv_a, _ = model_confidence_set(dict(items), alpha=0.10, seed=0)
        surv_b, _ = model_confidence_set(dict(reversed(items)), alpha=0.10, seed=0)
        assert set(surv_a) == set(surv_b)

    def test_identical_columns_all_survive(self, identical_triple_panel):
        surv, _ = model_confidence_set(identical_triple_panel, alpha=0.10)
        assert set(surv) == {"a", "b", "c"}

    def test_mixed_exact_tie_survivor_set_invariant_to_column_order(self):
        """REGRESSION-style positive control (round-2 strict-code-correctness review, 2026-08-26):
        the existing order-invariance tests above use either all-identical columns or a generic
        random panel -- neither exercises the mixed case of an exact BYTE-IDENTICAL tie between
        some models alongside a genuinely worse one, which is the actual risk zone for an
        insertion-order-dependent tie-break (the exact bug class this codebase already found and
        fixed once in social_choice.py's plurality winner). Verified clean on this construction
        across 3 orderings; kept as a permanent positive control."""
        rng = np.random.default_rng(7)
        T = 30
        tied = rng.normal(1.0, 0.3, T)
        worse = rng.normal(1.6, 0.3, T)
        base = {"m1": tied.copy(), "m2": tied.copy(), "m3": worse}
        for order in (["m1", "m2", "m3"], ["m3", "m2", "m1"], ["m2", "m3", "m1"]):
            surv, _ = model_confidence_set({k: base[k] for k in order}, seed=0)
            assert sorted(surv) == ["m1", "m2"], f"order {order} gave {sorted(surv)}"

    def test_many_near_tied_models_mcs_gives_plausible_multi_survivor_set(self):
        """M-competition-veteran review (round 3, 2026-08-26): only a 2-model tie case is exercised
        anywhere in the suite, but a real top-heavy competition leaderboard has many (10+) closely
        clustered models. A tight cluster of 10 near-identical-mean models plus 3 clearly-worse ones
        should survive as a genuine multi-model MCS (not spuriously collapse to a singleton), and
        decision_breakdown's return_ties=True should surface more than one binding opponent."""
        rng = np.random.default_rng(3)
        T = 60
        cluster = {f"m{i}": rng.normal(1.0, 0.05, T) for i in range(10)}   # tight cluster, same mean
        worse = {f"bad{i}": rng.normal(1.8, 0.1, T) for i in range(3)}     # clearly separated
        L = {**cluster, **worse}
        surv, _ = model_confidence_set(L, alpha=0.10, seed=0)
        assert len(surv) >= 2, f"expected a genuine multi-survivor MCS, got singleton {surv}"
        assert set(surv) <= set(cluster), "a clearly-worse model spuriously survived"
        assert not (set(worse) & set(surv)), "a clearly-worse model spuriously survived"

        from selection_fragility import decision_breakdown
        w = np.ones(T)
        k, opp, resp, ties = decision_breakdown(L, w, return_ties=True)
        assert len(ties) >= 2, f"expected multiple binding opponents in a tight cluster, got {ties}"

    @pytest.mark.skipif(not HAVE_NEW_API, reason="identified()/mcs_size() not implemented yet (Stage 1)")
    def test_identical_columns_via_new_api(self, identical_triple_panel):
        assert mcs_size(identical_triple_panel) == 3
        assert identified(identical_triple_panel) is False

    def test_pvalue_floor(self, random_panel):
        L = random_panel(seed=0, T=30, K=2)
        for B in (99, 999, 2000):
            _, p = model_confidence_set(L, alpha=0.10, B=B, seed=0)
            assert p >= 1.0 / (B + 1) - 1e-12


# ---- S1.4: golden values on the real panel --------------------------------------------------------
@pytest.mark.skipif(not HAVE_NEW_API, reason="identified()/mcs_size() not implemented yet (Stage 1)")
class TestRealPanelGolden:
    def test_PI_is_identified_both_metrics(self, real_cells):
        for metric in ("mase", "rmsse"):
            L = real_cells[("PI", metric)]
            assert identified(L) is True, f"PI/{metric} should be identified"

    def test_non_PI_series_generally_not_identified(self, real_cells, real_series_list):
        non_identified_count = sum(
            1 for sid in real_series_list if sid != "PI"
            for metric in ("mase", "rmsse")
            if not identified(real_cells[(sid, metric)])
        )
        # matches the session's repeatedly-verified 12-of-13 (mase) / 11-of-13 (rmsse) pattern
        assert non_identified_count >= 20, (
            f"only {non_identified_count} of 24 non-PI cells non-identified; "
            "expected the vast majority, matching 4 independent re-derivations this session"
        )


# ---- S1.5: failure modes --------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_NEW_API, reason="identified()/mcs_size() not implemented yet (Stage 1)")
class TestFailureModes:
    def test_K1_raises(self):
        with pytest.raises(ValueError, match=r"(?i)at least 2"):
            identified({"a": np.ones(10)})

    def test_nan_raises_naming_offender(self):
        L = {"a": np.array([1., np.nan, 3.]), "b": np.array([1., 2., 3.])}
        with pytest.raises(ValueError, match=r"(?i)nan"):
            identified(L)

    def test_int_dtype_works(self):
        L = {"a": np.array([1, 2, 3, 4, 5] * 6), "b": np.array([2, 3, 4, 5, 6] * 6)}
        identified(L)   # must not raise

    def test_float32_matches_float64_qualitatively(self, random_panel):
        L64 = random_panel(seed=0, T=30, K=4)
        L32 = {k: v.astype(np.float32) for k, v in L64.items()}
        assert identified(L64) == identified(L32)
        assert mcs_size(L64) == mcs_size(L32)


@pytest.mark.skipif(not HAVE_NEW_API, reason="identified/mcs_size not available")
class TestLossPanelAcceptance:
    """CONFIRMED HIGH (round-4 8-lens PyPI-preflight audit, 2026-09-07): following the README's own
    Staged v1.0 API table literally -- Stage 0 `LossPanel.from_losses(data)`, Stage 1 immediately
    after `identified(L)`/`mcs_size(L)`, nothing in the table suggesting a different input type is
    needed -- raised a raw internal `TypeError: object of type 'LossPanel' has no len()` three stack
    frames deep inside `identify.py`'s `_to_frame`. Every OTHER stage's function in that same table
    (resolution_report, concentration_share, pivot_agreement, selection_regret, mcb_bound,
    significance_boundary, certified_tied_subset, minimum_detectable_edge) already accepted a
    LossPanel directly -- identified()/mcs_size() were the outliers, and the identical bug class had
    already been found and fixed in resolution_report() on 2026-08-26. Fixed by routing both through
    the same shared `_unwrap_panel` helper every other LossPanel-accepting function already uses."""

    def from_losses(self, L):
        from selection_fragility import LossPanel
        return LossPanel.from_losses(L)

    def test_identified_accepts_a_losspanel_matching_dict_result(self, random_panel):
        L = random_panel(seed=1, T=25, K=3)
        panel = self.from_losses(L)
        assert identified(panel) == identified(L)

    def test_mcs_size_accepts_a_losspanel_matching_dict_result(self, random_panel):
        L = random_panel(seed=2, T=25, K=5)
        panel = self.from_losses(L)
        assert mcs_size(panel) == mcs_size(L)

    def test_readme_stage0_then_stage1_literally_does_not_raise(self, random_panel):
        """The exact sequence README.md's Staged v1.0 API table documents: build with Stage 0, feed
        straight into Stage 1, nothing else. This is the literal reproduction of the reported bug."""
        L = random_panel(seed=3, T=20, K=4)
        panel = self.from_losses(L)
        result = identified(panel)   # must not raise TypeError
        assert isinstance(result, bool)

    def test_losspanel_weights_are_honored_when_w_not_given(self, random_panel):
        """_unwrap_panel pulls w=panel.weights when the caller doesn't pass w explicitly -- confirm
        this actually wires through mcs_size, not just that the call doesn't crash."""
        from selection_fragility import LossPanel
        L = random_panel(seed=4, T=10, K=3)
        panel = LossPanel.from_losses(L)
        # a panel built from_losses() has uniform weights by construction -- passing w=None (the
        # default) through mcs_size(panel) must agree with explicitly passing that same uniform w.
        assert mcs_size(panel) == mcs_size(L, w=panel.weights)
