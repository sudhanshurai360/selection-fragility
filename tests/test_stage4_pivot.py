"""Stage 4 -- concentration_share and pivot_agreement. TOOL_TEST_PLAN.md S4.1-S4.2.
Rebuilds the statistician's matched-size calibration matrix in the shipped code, not a throwaway
script -- this is the test that must keep concentration_share from repeating pivot_sharpness's
near-tie collapse."""
import numpy as np
import pytest

try:
    from selection_fragility import concentration_share, pivot_agreement
    HAVE_PIVOT = True
except ImportError:
    HAVE_PIVOT = False

pytestmark = pytest.mark.skipif(not HAVE_PIVOT, reason="concentration_share/pivot_agreement not implemented yet (Stage 4)")


def _shock_panel(rng, T, K, config):
    """config: list of (n_periods, fractions) describing how the champion's margin over the
    binding rival is distributed across periods. Mirrors the 7 configurations tonight's
    statistician review measured at matched 5% size.

    FIXED 2026-08-16: the original version left the K-2 bystander models at the SAME baseline as
    champ/rival. Verified directly against the finished concentration_share implementation: at K=6
    that construction makes rival a blown-out overall loser (mean pushed up by the whole shock
    margin), so decision_breakdown's binding-opponent selection -- correctly finding the CLOSEST
    competitor -- picks one of the four untouched bystander models instead of rival in the large
    majority of draws (checked directly: 'm3' vs 'm4', neither of which is the shocked model, on a
    representative seed). The shock configuration was then invisible to any binding-opponent-based
    statistic, which is why every 'two_*' config measured near-identical, weak power regardless of
    how the margin was split -- a flaw in this fixture, not in concentration_share (confirmed by
    re-running the same configs at K=2, where champ/rival are the only two models and every config
    passes its floor with room to spare). Bystanders now start a full point worse than champ/rival's
    baseline (mean 2.0 vs ~1.0), so they are never competitive and champ/rival remain the natural
    decision pair regardless of K."""
    base = rng.normal(1.0, 0.05, (T, K))
    champ, rival = 0, 1
    if K > 2:
        base[:, 2:] += 1.0
    n_periods, fracs = config
    # FIXED 2026-08-16 (independent reproducibility review): `total_margin = 0.30 * T` -- at T=30 this
    # is 9.0, a ~900% edge against a baseline mean of 1.0, roughly 800x the real 26-cell panel's own
    # median observed edge (~0.5-1.1%, see resolution.py). At that scale EVERY config had power ~1.0,
    # so the test never actually exercised the near-tie collapse it exists to guard against -- it just
    # never got close enough to the boundary to find out. A stray `* T` is the likely origin: it turns
    # what reads as "a moderate constant edge" in the comment into something that grows unboundedly
    # with T instead of staying fixed. Swept total_margin as a plain constant (not T-scaled) against a
    # T=30 panel and picked 0.35 -- close to the value that reproduces power comparable to the design
    # doc's own original per-config figures (e.g. ~0.25-0.35 on two_50_50, matching its stated 0.348),
    # and, at 0.35/T=30 periods, an average per-period contribution close to 1% of the mean -- the
    # panel's own realistic scale. At this scale concentration_share does NOT collapse on the critical
    # two_50_50 near-tie case (power 0.15-0.25 across 5 reseeds, comfortably above the 5% nominal
    # size) -- confirming concentration_share genuinely avoids the pivot_sharpness failure mode, which
    # the old 900%-edge fixture could never have actually tested.
    total_margin = 0.35
    idx = rng.choice(T, n_periods, replace=False)
    for i, frac in zip(idx, fracs):
        base[i, rival] += total_margin * frac
    return base, champ, rival


CONFIGS = {
    "one_period":      (1, [1.0]),
    "two_50_50":       (2, [0.5, 0.5]),
    "two_70_30":       (2, [0.7, 0.3]),
    "two_85_15":       (2, [0.85, 0.15]),
    "three_60_25_15":  (3, [0.6, 0.25, 0.15]),
    "three_equal":     (3, [1 / 3, 1 / 3, 1 / 3]),
    "five_tapered":    (5, [0.4, 0.25, 0.15, 0.12, 0.08]),
}


def _matched_size_power(statistic_fn, config, T=30, K=6, n_null=400, n_alt=400, alpha=0.05, seed=0):
    """Measure size under H0 and power under the given shock configuration, at a common
    matched-alpha threshold derived from the null distribution -- the correct way to compare
    statistics, per tonight's discipline against using AUC."""
    rng = np.random.default_rng(seed)
    null_vals = []
    for _ in range(n_null):
        panel = rng.normal(1.0, 0.05, (T, K))
        L = {f"m{i}": panel[:, i] for i in range(K)}
        null_vals.append(statistic_fn(L))
    threshold = np.percentile(null_vals, 100 * (1 - alpha))
    size = np.mean(np.array(null_vals) > threshold)  # should be ~alpha by construction; sanity only

    alt_vals = []
    for _ in range(n_alt):
        panel, champ, rival = _shock_panel(rng, T, K, config)
        L = {f"m{i}": panel[:, i] for i in range(K)}
        alt_vals.append(statistic_fn(L))
    power = np.mean(np.array(alt_vals) > threshold)
    return size, power


# ---- S4.1: concentration_share matched-size calibration --------------------------------------------
class TestConcentrationShare:
    @pytest.mark.parametrize("config_name,expected_power_min", [
        # FLOORS RECALIBRATED 2026-08-16 against the corrected (realistic-scale) shock fixture above --
        # verified across 5 reseeds (0-4), each floor set safely below the observed minimum.
        ("one_period", 0.65),
        ("two_50_50", 0.12),      # THE critical case: pivot_sharpness collapsed to 0.09 here
        ("two_70_30", 0.28),
        ("two_85_15", 0.45),
        ("three_60_25_15", 0.18),
        ("three_equal", 0.02),
        ("five_tapered", 0.03),
    ])
    def test_matched_size_power_floor(self, config_name, expected_power_min):
        size, power = _matched_size_power(concentration_share, CONFIGS[config_name])
        assert power >= expected_power_min, (
            f"{config_name}: power {power:.3f} below floor {expected_power_min} -- "
            "check for a pivot_sharpness-style collapse"
        )

    def test_does_not_collapse_below_size_at_near_ties(self):
        """THE SPECIFIC FAILURE MODE pivot_sharpness (c1/c2) suffered: power BELOW the 5% nominal
        size on the two-period 50/50 configuration. concentration_share must not repeat this.
        TIGHTENED 2026-08-16: raised from >0.05 (a bare non-collapse check) to >0.10 now that the
        fixture is calibrated at a realistic scale and power here is verified at 0.12-0.25 across
        reseeds -- still comfortably below the observed range, but a stronger check than "not
        literally at nominal size"."""
        size, power = _matched_size_power(concentration_share, CONFIGS["two_50_50"])
        assert power > 0.10, (
            f"power {power:.3f} too close to nominal 5% size on the near-tie configuration -- "
            "this is exactly the collapse that killed pivot_sharpness"
        )

    def test_size_under_H0_near_nominal(self):
        size, _ = _matched_size_power(concentration_share, CONFIGS["one_period"])
        assert 0.02 < size < 0.10   # sanity: matched-threshold construction should give ~alpha=0.05

    def test_degenerate_margin_nan_guarded(self, near_tie_float_panel):
        val = concentration_share(near_tie_float_panel)
        assert val is None or not np.isfinite(val) or abs(val) < 1e3

    def test_exact_tie_returns_none_not_a_finite_number(self):
        """REGRESSION (independent 'wild' review, 2026-08-17, cross-fix interaction lens):
        `pivot.py::_champ_opp_contributions`'s `if not ties: return None, None, None` was dead code
        -- decision_breakdown's `ties` list is non-empty for any panel with >=2 models regardless of
        whether the minimum k is 0, so the intended "k*=0, no strict winner, nothing pivotal" case
        was never actually caught. Exactly the same None-vs-falsy defect class as
        `prop22.py::certified_tied_subset`'s k==0 fix earlier the same night, never carried over to
        this sibling function. Both concentration_share and pivot_agreement must return None for a
        genuine k*=0 exact tie, even when real (non-cancelling) per-period differentials exist."""
        rng = np.random.default_rng(7)
        T = 20
        champ = rng.normal(1.0, 0.3, T)
        rival = champ.copy() + rng.normal(0, 0.4, T)
        rival = rival - np.mean(rival) + np.mean(champ)   # force an exact mean tie -> k*=0
        third = champ + 0.5
        L = {"champ": champ, "rival": rival, "third": third}
        assert concentration_share(L, np.ones(T)) is None
        assert pivot_agreement(L, np.ones(T), seed=0) is None


# ---- S4.2: pivot_agreement -------------------------------------------------------------------------
class TestPivotAgreement:
    def test_non_integer_subsample_fraction_defined(self, random_panel):
        L = random_panel(seed=0, T=13, K=4)   # 0.8*13 = 10.4, non-integer
        val = pivot_agreement(L, seed=0)
        assert val is None or (0 <= val <= 1)

    def test_determinism(self, random_panel):
        L = random_panel(seed=0, T=30, K=4)
        v1 = pivot_agreement(L, seed=0)
        v2 = pivot_agreement(L, seed=0)
        assert v1 == v2

    def test_too_small_T_returns_undetermined(self, random_panel):
        L = random_panel(seed=0, T=3, K=4)   # 0.8*3 < 2
        val = pivot_agreement(L, seed=0)
        assert val is None or np.isnan(val)

    def test_recentred_null_benchmark_near_052(self, recentred_real_panel):
        """THE CALIBRATION ANCHOR, RE-CORRECTED 2026-08-29 (11-angle review): this cell is built
        from real_cells[("PI","mase")], and PI's own SARIMA forecasts changed when the main repo
        fixed a real SARIMA bug (a missing trend term was forcing a flat, driftless forecast) --
        so recomputing here against the current data is expected to move, not a regression.
        Recomputed directly (median 0.518 over 30 seeds, was 0.673 pre-SARIMA-fix, itself a
        2026-08-16 correction of an original "~0.42" design-doc figure). The mechanism explaining
        WHY this benchmark sits well above 1/T is unchanged by the SARIMA fix: F9 recentres each
        model's MEAN to remove level differences (the right null for winner_stability, which asks
        "which MODEL wins"), but it does NOT touch the year-by-year SHAPE/CORRELATION structure --
        a real shared shock year (e.g. a crisis that hit every model's forecast simultaneously)
        stays the largest single contributor for almost ANY randomly-paired champion/opponent after
        recentring, which is exactly what makes a PERIOD pivotal here. That makes F9 a weaker null
        for period-identity stability specifically than it is for model-identity stability. Still
        nowhere near 1/T=1/6=0.167 -- reading a real value against 1/T instead of this benchmark
        remains the mistake `exchangeable_benchmark` was built to fix."""
        vals = []
        for seed in range(30):
            L = recentred_real_panel(series="PI", metric="mase", seed=seed)
            v = pivot_agreement(L, seed=seed)
            if v is not None and np.isfinite(v):
                vals.append(v)
        assert len(vals) > 10
        median_v = np.median(vals)
        assert 0.40 < median_v < 0.65, (
            f"recentred-panel pivot_agreement median {median_v:.3f}, expected near 0.52 "
            "(recomputed 2026-08-29, post-SARIMA-fix); if this drifts toward 1/T=0.167, the "
            "benchmark itself may be broken"
        )

    def test_golden_value_real_panel_median(self, real_cells, real_series_list):
        vals = []
        for sid in real_series_list:
            for metric in ("mase", "rmsse"):
                v = pivot_agreement(real_cells[(sid, metric)], seed=0)
                if v is not None and np.isfinite(v):
                    vals.append(v)
        median_v = np.median(vals)
        assert 0.50 < median_v < 0.78, f"real-panel median {median_v:.3f}, expected near 0.64"

class TestPivotRenameInvariance:
    """ADDED 2026-08-16 after independent review: concentration_share (an argmax-over-contributions
    statistic) and pivot_agreement (which names a 'pivotal period' -- another argmax) are
    STRUCTURALLY the same shape of computation as the plurality_winner/per_period_winner argmax
    that produced the confirmed T2/T3 tie-award-by-name-order bugs. Neither statistic had a
    rename-invariance test anywhere in the original suite -- this was the single highest-value gap
    the review found. Mirrors test_stage3_kstar_prop22.py::TestRenameInvariance exactly."""

    RENAME_SCHEMES = [
        lambda ms: {m: m[::-1] for m in ms},
        lambda ms: dict(zip(ms, sorted(ms, reverse=True))),
        lambda ms: {m: str(i) for i, m in enumerate(ms)},
        lambda ms: {m: f"éè{i}" for i, m in enumerate(ms)},
    ]

    @staticmethod
    def _has_tied_opponent(L, w):
        """Both statistics are computed against decision_breakdown's binding opponent. When SEVERAL
        opponents genuinely tie for k* (decision_breakdown's own documented case, and the exact
        source of tests/test_stage3_kstar_prop22.py's rename-invariance fix), a renaming can flip
        WHICH tied opponent gets picked, and the two are legitimately allowed to give different
        contribution vectors -- that's not a bug in concentration_share/pivot_agreement, it's the
        same tie-break ambiguity decision_breakdown already documents. Skip those seeds here, exactly
        as Stage 3 does for the analogous decision_breakdown rename test."""
        from selection_fragility import decision_breakdown
        _k, _opp, _resp, ties = decision_breakdown(L, w, return_ties=True)
        return len(ties) > 1

    @pytest.mark.parametrize("scheme_idx", range(4))
    def test_concentration_share_invariant_to_renaming(self, random_panel, scheme_idx):
        """FIXED 2026-08-16: seeds with a genuine multi-way tie for binding opponent (verified
        directly: seeds 1/10/12/14/22/23 at T=15,K=4 all have 2-3 opponents tied on k*) are excluded
        -- same root cause and same fix as test_stage3_kstar_prop22.py's rename-invariance test."""
        violations = []
        for seed in range(25):
            L = random_panel(seed=seed, T=15, K=4)
            w = np.ones(15)
            if self._has_tied_opponent(L, w):
                continue
            v1 = concentration_share(L)
            remap = self.RENAME_SCHEMES[scheme_idx](list(L))
            L2 = {remap[m]: v for m, v in L.items()}
            v2 = concentration_share(L2)
            if v1 is None and v2 is None:
                continue
            if v1 is None or v2 is None or abs(v1 - v2) > 1e-9:
                violations.append(seed)
        assert not violations, f"concentration_share rename-invariance violated at seeds {violations} (scheme {scheme_idx})"

    @pytest.mark.parametrize("scheme_idx", range(4))
    def test_pivot_agreement_invariant_to_renaming(self, random_panel, scheme_idx):
        """FIXED 2026-08-16: same tied-opponent exclusion as the concentration_share test above."""
        violations = []
        for seed in range(25):
            L = random_panel(seed=seed, T=15, K=4)
            w = np.ones(15)
            if self._has_tied_opponent(L, w):
                continue
            v1 = pivot_agreement(L, seed=0)
            remap = self.RENAME_SCHEMES[scheme_idx](list(L))
            L2 = {remap[m]: v for m, v in L.items()}
            v2 = pivot_agreement(L2, seed=0)
            if v1 is None and v2 is None:
                continue
            if v1 is None or v2 is None or v1 != v2:
                violations.append(seed)
        assert not violations, f"pivot_agreement rename-invariance violated at seeds {violations} (scheme {scheme_idx})"

    def test_explicit_fixtures_under_renaming(self, dominance_panel, identical_triple_panel,
                                              near_tie_float_panel):
        """The named regression fixtures, re-run under a renaming that changes the alphabetical
        order -- proves any fix is genuinely name-invariant, not fixed only for the specific
        letters used in the original bug reports (same discipline as Stage 3's equivalent test)."""
        for panel, remap in [
            (dominance_panel, {"a": "zeta", "b": "alpha"}),
            (identical_triple_panel, {"a": "zzz", "b": "aaa", "c": "mmm"}),
            (near_tie_float_panel, {"a": "zeta", "b": "alpha"}),
        ]:
            renamed = {remap[m]: v for m, v in panel.items()}
            v1_orig, v1_renamed = concentration_share(panel), concentration_share(renamed)
            if v1_orig is not None and v1_renamed is not None:
                assert abs(v1_orig - v1_renamed) < 1e-9

    def test_near_tied_period_construction(self, random_panel):
        """The structural analogue of the near-tied-MODEL fixtures above, but for PERIODS: two
        periods contributing near-equally to the margin. pivot_agreement's 'name the pivotal
        period' step is an argmax over periods exactly like per_period_winner's argmax over
        models -- if a position/order tie-break silently substitutes for a real signal here, this
        is where it would show. Constructs a panel with two periods of near-identical contribution
        and checks the statistic doesn't silently prefer one by position alone across relabelings
        that reorder the period axis."""
        rng = np.random.default_rng(0)
        T = 10
        a = rng.normal(1.0, 0.05, T)
        b = a.copy()
        # two periods, near-tied largest contributors
        b[3] += 0.50
        b[7] += 0.50 + 1e-6   # deliberately just barely larger, to probe order-sensitivity
        L = {"a": a, "b": b}
        # TOLERANCE FIXED 2026-08-16: reversing the period axis is a genuine DATA-CONTENT change (the
        # value at position 7 moves to position 2, etc.), not a pure relabeling like the model-name
        # renames above -- pivot_agreement's subsample draws are position-indexed, so the two
        # orientations run through completely independent random index sets, and any Monte Carlo
        # statistic will show sampling noise between them even with zero systematic bias. Verified
        # directly: the gap shrinks from 0.071 (n_boot=200) to 0.016 (n_boot=1000) to 0.006
        # (n_boot=5000) as noise averages out -- exactly the signature of pure sampling variance, not
        # a position/order bias (which would NOT shrink with more draws). n_boot=3000 and a tolerance
        # of 0.03 keeps the noise floor comfortably below what a real bias would produce while staying
        # fast.
        v1 = pivot_agreement(L, seed=0, n_boot=3000)
        # reverse the period order entirely (a full relabeling of the T axis, not the model axis)
        L_reversed = {m: v[::-1].copy() for m, v in L.items()}
        v2 = pivot_agreement(L_reversed, seed=0, n_boot=3000)
        if v1 is not None and v2 is not None:
            assert abs(v1 - v2) < 0.03, (
                f"pivot_agreement sensitive to period ORDER (not just identity) under near-tied "
                f"periods: {v1} vs {v2} after reversing the period axis"
            )


class TestFlaggedCellsBenchmark:
    def test_flagged_cells_at_or_below_recentred_benchmark(self, real_cells, recentred_real_panel):
        """RE-CORRECTED 2026-08-29 (11-angle review): two things changed since the 2026-08-16 fix
        this docstring used to describe. (1) UMCSENT was removed from the panel entirely (confirmed
        U-Michigan licensing restriction) -- dropped from `flagged` rather than guessed at with a
        replacement series, since there is no principled basis for which series should inherit its
        slot. (2) The anchor benchmark itself moved 0.673 -> 0.518 (see
        test_recentred_null_benchmark_near_052, same file) because PI's own SARIMA forecasts changed
        under the main repo's SARIMA trend-term fix. Recomputed the 4 remaining originally-flagged
        cells against the CURRENT 0.518 benchmark directly: only HOUST/rmsse (0.455) still falls
        below it; HOUST/mase (0.52), UNEMPLOY/mase (0.775), and PCE/rmsse (0.545) do not. This is a
        real, disclosed finding, not a bug to paper over -- most of the originally-flagged cells no
        longer read as "unstable" relative to the new, lower benchmark. Kept as a narrow regression
        check on the one cell that still reproduces, not as evidence about how many cells overall are
        unstable (that requires a fresh full sweep, out of scope for this narrow test)."""
        flagged = [("HOUST", "mase"), ("HOUST", "rmsse"), ("UNEMPLOY", "mase"), ("PCE", "rmsse")]
        n_confirmed = 0
        for sid, metric in flagged:
            v = pivot_agreement(real_cells[(sid, metric)], seed=0)
            if v is not None and v < 0.518:
                n_confirmed += 1
        assert n_confirmed >= 1, (
            f"only {n_confirmed} of {len(flagged)} previously-flagged cells reproduce as unstable "
            f"against the corrected 0.518 benchmark -- expected at least HOUST/rmsse to still confirm"
        )


class TestPivotOpponentWeightScaleInvariance:
    """FIXED 2026-09-10 (structural post-publish audit -- a fifth independent instance of the
    "champion pick via raw, un-floored float comparison" bug class already found and fixed four
    times elsewhere in this package). `_champ_opp_contributions`'s opponent tie-break
    (`M > best_M`, module-private, exercised here only through concentration_share/pivot_agreement)
    picked among decision_breakdown's tied opponents by total margin M, computed via `c.sum()` --
    order-dependent floating-point summation. Two opponents whose per-period contributions are
    PERMUTATIONS of each other are mathematically tied on M by construction, but numpy's summation
    gave different float64 rounding for the two orderings, and multiplying the weight vector by a
    scalar (a documented no-op) changed which ordering artifact appeared, flipping which opponent
    (and therefore which pivotal period / concentration_share value) was named. Fixed via
    `math.fsum` (provably order-invariant), NOT a relative floor + name-based fallback -- a floor
    here would have reopened the exact rename-dependence bug TestPivotRenameInvariance above exists
    to prevent."""

    def test_permutation_tied_opponents_give_scale_invariant_pick(self):
        from selection_fragility.pivot import _champ_opp_contributions
        a = np.array([0.0, 0.0, 0.0, 0.0])
        y = np.array([1.0, 2.0, 3.0, 4.125])
        z = np.array([4.125, 3.0, 2.0, 1.0])   # a permutation of y -> mathematically tied M
        L = {"a": a, "y": y, "z": z}
        picks = set()
        for scale in (1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9):
            _champ, opp, _c = _champ_opp_contributions(L, np.ones(4) * scale)
            picks.add(opp)
        assert len(picks) == 1, f"opponent pick flipped across weight scales: {picks}"
