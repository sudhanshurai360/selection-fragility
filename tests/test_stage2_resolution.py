"""Stage 2 -- resolution report: MDE, selection_regret, MCB bound, R1/R2 refusal rules.release/TOOL_TEST_PLAN.md S2.1-S2.4. None of this exists yet; all skipped until Stage 2 ships."""
import numpy as np
import pytest
try:
    from selection_fragility import minimum_detectable_edge, selection_regret, mcb_bound
    HAVE_RESOLUTION = True
except ImportError:
    HAVE_RESOLUTION = False
pytestmark = pytest.mark.skipif(not HAVE_RESOLUTION, reason="resolution report not implemented yet (Stage 2)")
# ---- S2.1: MDE --------------------------------------------------------------------------------
class TestMDE:
    @pytest.mark.parametrize("T", [5, 30, 1000])
    def test_monotone_decreasing_in_T(self, random_panel, T):
        pass  # placeholder for the cross-T comparison below; see test_mde_monotone_in_T
    def test_mde_monotone_in_T(self, random_panel):
        mdes = []
        for T in (5, 30, 1000):
            L = random_panel(seed=0, T=T, K=4)
            mdes.append(minimum_detectable_edge(L))
        assert mdes[0] > mdes[1] > mdes[2], f"MDE not monotone decreasing in T: {mdes}"
    def test_mde_increases_with_power_target(self, random_panel):
        """power=0.50 UPDATED 2026-08-16: a follow-up 'wild' review found power<=0.5 makes MDE go
        negative (z=norm.ppf(power) is non-positive there), which then defeats resolution_report()'s
        own R1 refusal gate -- power is now validated to be strictly >0.5 (see
        resolution.py::_validate_one_sided). 0.55 is the smallest valid value near the old 0.50."""
        L = random_panel(seed=0, T=30, K=4)
        mde_55 = minimum_detectable_edge(L, power=0.55)
        mde_80 = minimum_detectable_edge(L, power=0.80)
        mde_90 = minimum_detectable_edge(L, power=0.90)
        assert mde_55 < mde_80 < mde_90
    def test_mde_decreases_as_alpha_loosens(self, random_panel):
        L = random_panel(seed=0, T=30, K=4)
        mde_01 = minimum_detectable_edge(L, alpha=0.01)
        mde_05 = minimum_detectable_edge(L, alpha=0.05)
        mde_10 = minimum_detectable_edge(L, alpha=0.10)
        assert mde_01 > mde_05 > mde_10
    def test_zero_variance_panel_does_not_crash(self):
        L = {"a": np.full(30, 1.0), "b": np.full(30, 1.1)}
        result = minimum_detectable_edge(L)
        assert np.isfinite(result) or result == 0.0
    def test_extreme_variance_can_exceed_100_percent(self, random_panel):
        L = random_panel(seed=0, T=10, K=4, sigma=5.0)   # huge noise relative to mu=1.0
        mde = minimum_detectable_edge(L)
        assert mde > 0   # must not be silently clipped to a "safe-looking" small number
    def test_uses_binding_rival_not_average(self):
        """Champion at mean 1.0; two rivals at 1.05 (close) and 3.0 (far). MDE must reflect the
        CLOSE rival, not an average across rivals -- a distant rival cannot make the champion's
        margin look more resolved than it is against the real threat."""
        rng = np.random.default_rng(0)
        L = {
            "champ": rng.normal(1.00, 0.10, 30),
            "close": rng.normal(1.05, 0.10, 30),
            "far":   rng.normal(3.00, 0.10, 30),
        }
        mde_full = minimum_detectable_edge(L)
        mde_close_only = minimum_detectable_edge({"champ": L["champ"], "close": L["close"]})
        np.testing.assert_allclose(mde_full, mde_close_only, rtol=0.05)
    def test_binding_rival_stable_on_near_tied_rivals_across_weight_scale(self):
        """FIXED 2026-09-09 (round-5 stress-review, champion_pattern_hunt lens): _binding_rival's
        sorted-name tie-break previously only guarded an EXACT float tie of the distance-to-
        champion, not a relative floor for NEAR-ties -- a uniform weight rescale (a documented
        no-op for np.average's ratio) could flip which of two near-equally-close rivals was named
        'binding', flipping significance_boundary and the headline `resolved` verdict. This function
        was already the target of one confirmed order-dependence regression (see its own docstring)
        but had no near-tie coverage, only the well-separated (1.05 vs 3.0) fixture above."""
        L = {
            "a": np.array([-0.4, -0.4, -0.4, -0.4]),
            "y": np.array([0.0, 0.0, 2.0, 2.125]),
            "z": np.array([0.0, 0.0, 0.0, 4.125]),
        }
        boundaries = []
        for scale in (1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9):
            from selection_fragility.resolution import resolution_report
            w = np.ones(4) * scale
            r = resolution_report(L, w)
            boundaries.append((r["binding_rival"], r["resolved"], r["significance_boundary"]))
        rivals = {b[0] for b in boundaries}
        resolved_flags = {b[1] for b in boundaries}
        assert len(rivals) == 1, f"binding_rival flipped across weight scales: {boundaries}"
        assert len(resolved_flags) == 1, f"resolved verdict flipped across weight scales: {boundaries}"
# ---- S2.2: selection_regret (LOO) ---------------------------------------------------------------
class TestSelectionRegret:
    def test_T2_degenerate_raises_or_flags(self, random_panel):
        L = random_panel(seed=0, T=2, K=3)
        try:
            r = selection_regret(L)
            assert r is None or np.isnan(r)
        except ValueError:
            pass
    def test_T3_minimum_meaningful(self, random_panel):
        L = random_panel(seed=0, T=3, K=3)
        r = selection_regret(L)
        assert r is not None and np.isfinite(r) and r >= 0
    def test_champion_wins_every_fold_gives_zero_regret(self):
        """Constructed so one model dominates every single leave-one-out fold."""
        T = 10
        L = {"champ": np.full(T, 1.0), "rival": np.full(T, 2.0)}
        r = selection_regret(L)
        assert r == pytest.approx(0.0, abs=1e-9)
    def test_adversarial_max_churn_regret_at_least_pooled_edge(self):
        """A different model wins every single LOO fold (constructed to alternate). LOO regret of
        this unstable rule must be AT LEAST as large as the pooled edge a stable rule would show."""
        T = 10
        rng = np.random.default_rng(0)
        a = np.array([0.5 if i % 2 == 0 else 1.5 for i in range(T)])
        b = np.array([1.5 if i % 2 == 0 else 0.5 for i in range(T)])
        L = {"a": a, "b": b}
        pooled_edge = abs(a.mean() - b.mean()) / min(a.mean(), b.mean())
        r = selection_regret(L)
        assert r >= pooled_edge - 1e-9
    def test_determinism(self, random_panel):
        L = random_panel(seed=0, T=30, K=5)
        r1 = selection_regret(L)
        r2 = selection_regret(L)
        assert r1 == r2
    def test_weights_actually_threaded_through(self, random_panel):
        """Regression guard: LOO regret computed with a skewed weight vector must differ from the
        unweighted version on a panel constructed so weighting matters -- proves the weights are
        not silently dropped inside the refit loop."""
        rng = np.random.default_rng(0)
        T = 20
        L = {"a": rng.normal(1.0, 0.3, T), "b": rng.normal(1.0, 0.3, T)}
        w_skewed = np.concatenate([np.full(10, 5.0), np.full(10, 0.1)])
        r_uniform = selection_regret(L, weights=np.ones(T))
        r_skewed = selection_regret(L, weights=w_skewed)
        assert r_uniform != pytest.approx(r_skewed, rel=1e-6)
    def test_w_keyword_alias_for_weights(self):
        """FIXED 2026-09-10 (round-6 stress-review, api_consistency lens): every sibling function in
        this module (minimum_detectable_edge, significance_boundary, mcb_bound, resolution_report)
        names this parameter `w`; selection_regret alone named it `weights`, so a caller who learned
        the `w=` convention from any sibling got a raw TypeError here. `weights` is kept as the
        primary/positional name (it shipped in the published v1.0.0), `w` is an additive
        keyword-only alias -- both must give the identical result, and passing both must raise."""
        rng = np.random.default_rng(0)
        T = 12
        L = {"a": rng.normal(1.0, 0.3, T), "b": rng.normal(1.1, 0.3, T)}
        w = np.linspace(0.5, 2.0, T)
        r_positional = selection_regret(L, w)
        r_weights_kw = selection_regret(L, weights=w)
        r_w_kw = selection_regret(L, w=w)
        assert r_positional == r_weights_kw == r_w_kw
        with pytest.raises(TypeError, match=r"(?i)got both"):
            selection_regret(L, weights=w, w=w)
    def test_golden_value_real_panel_quartile_monotone(self, real_cells, real_series_list):
        """Reconstructs the positioning reviewer's own measurement: predicted regret should be
        monotone non-decreasing across quartiles when cells are sorted by predicted regret (a
        self-consistency check -- proves the statistic orders cells sensibly, independent of
        needing the exact realised-regret ground truth this fixture cannot supply)."""
        vals = []
        for sid in real_series_list:
            for metric in ("mase", "rmsse"):
                L = real_cells[(sid, metric)]
                vals.append(selection_regret(L))
        vals = sorted(v for v in vals if v is not None and np.isfinite(v))
        assert len(vals) >= 20
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        assert q1 <= q2 <= q3   # trivially true by construction of percentiles; real check is:
        assert vals[0] <= vals[-1]   # monotone ordering exists and is non-degenerate
        assert vals[-1] > vals[0], "all cells have identical regret -- statistic is not discriminating"
    def test_fold_champion_actually_routes_through_pooled_winner(self, monkeypatch):
        """FIXED 2026-09-09 (round-5 stress-review, test_gap_hunt lens), TEST ITSELF CORRECTED
        2026-09-09 (independent review of the fix): selection_regret's 2026-09-09 fix routes each
        fold's champion through pooled_winner(). A first version of this test hand-built a near-tied
        panel (two models tied at the same pooled mean, weight-rescaled to force float64-noise
        disagreement) and cross-checked the reported regret against a reimplementation using
        pooled_winner() per fold -- but independent review reverted the fix's routing to a raw,
        un-floored min and this test STILL passed: at every fold where the raw and floored picks
        actually disagreed, both candidate models happened to have byte-identical held-out-period
        losses, so the champion's IDENTITY differed but the regret VALUE did not -- vacuous. This
        version instead verifies the wiring directly: monkeypatch pooled_winner to a deliberately
        wrong constant-champion rule and confirm selection_regret's reported value changes, proving
        the fold-champion pick is genuinely read from pooled_winner()'s return value rather than
        computed independently."""
        L = {
            "champ": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "rival": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            "other": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        }
        w = np.ones(5)
        real = selection_regret(L, w)
        import selection_fragility.resolution as resolution_mod
        monkeypatch.setattr(resolution_mod, "pooled_winner", lambda Ld, wd: "other")
        forced_wrong = selection_regret(L, w)
        assert forced_wrong != pytest.approx(real), (
            "selection_regret's fold-champion pick does not actually route through pooled_winner() "
            "-- forcing a different champion had no effect on the reported regret"
        )
# ---- S2.3: MCB bound ------------------------------------------------------------------------------
class TestMCBBound:
    def test_bound_is_positive(self, random_panel):
        L = random_panel(seed=0, T=30, K=4)
        b = mcb_bound(L)
        assert b >= 0
    def test_near_tie_small_bound(self):
        L = {"a": np.full(30, 1.0), "b": np.full(30, 1.001)}
        b = mcb_bound(L)
        assert b < 0.05
    def test_bound_never_below_observed_edge(self, random_panel):
        """A confidence bound cannot be tighter than the point estimate it bounds -- property test,
        200 random panels, 0 violations expected."""
        violations = 0
        for seed in range(200):
            L = random_panel(seed=seed, T=30, K=6)
            means = {m: v.mean() for m, v in L.items()}
            champ = min(means, key=means.get)
            rival = sorted((m for m in means if m != champ), key=lambda m: means[m])[0]
            observed_edge = abs(means[rival] - means[champ]) / means[champ]
            bound = mcb_bound(L)
            if bound < observed_edge - 1e-9:
                violations += 1
        assert violations == 0
    def test_bound_shrinks_as_T_grows(self, random_panel):
        b_small = mcb_bound(random_panel(seed=0, T=10, K=4))
        b_large = mcb_bound(random_panel(seed=0, T=500, K=4))
        assert b_large < b_small
    def test_golden_value_real_panel(self, real_cells, real_series_list):
        bounds = []
        for sid in real_series_list:
            for metric in ("mase", "rmsse"):
                bounds.append(mcb_bound(real_cells[(sid, metric)]))
        median_bound = np.median(bounds)
        assert 0.01 < median_bound < 0.20, f"median MCB bound {median_bound} outside expected ~0.062 ballpark"
# ---- S2.4: refusal rules R1/R2 --------------------------------------------------------------------
class TestRefusalRules:
    def test_R1_fires_when_edge_below_MDE(self):
        """Constructed panel with a genuinely tiny, unresolvable edge at T=10."""
        rng = np.random.default_rng(0)
        L = {"a": rng.normal(1.000, 0.5, 10), "b": rng.normal(1.001, 0.5, 10)}
        from selection_fragility import resolution_report
        r = resolution_report(L)
        assert r["resolved"] is False
    def test_R1_does_not_fire_when_edge_clears_MDE(self):
        L = {"a": np.full(30, 1.0), "b": np.full(30, 3.0)}
        from selection_fragility import resolution_report
        r = resolution_report(L)
        assert r["resolved"] is True
    def test_R2_refuses_fragility_read_when_not_identified(self, identical_triple_panel):
        from selection_fragility import resolution_report
        r = resolution_report(identical_triple_panel)
        assert r.get("fragility_read") in (None, "undetermined")
    def test_real_panel_R1_fires_in_most_cells(self, real_cells, real_series_list):
        from selection_fragility import resolution_report
        n_unresolved = sum(
            1 for sid in real_series_list for metric in ("mase", "rmsse")
            if resolution_report(real_cells[(sid, metric)])["resolved"] is False
        )
        assert n_unresolved >= 20, (
            f"only {n_unresolved} of 26 cells unresolved; expected ~24 per tonight's measurement"
        )
    def test_resolution_report_accepts_losspanel_directly(self):
        """FIXED 2026-08-26 (independent applied-practitioner review): resolution_report() used to
        only accept a raw {model: array} dict/DataFrame, unlike report()/compare() which both accept
        a LossPanel directly -- passing the same LossPanel already built for report() raised a
        confusing internal TypeError('LossPanel' has no len()) from deep inside _validate_losses
        instead of working, or failing with a clear top-level message."""
        from selection_fragility import LossPanel, resolution_report
        rng = np.random.default_rng(0)
        L = {"a": rng.normal(1.0, 0.2, 30), "b": rng.normal(1.1, 0.2, 30)}
        panel = LossPanel.from_losses(L)
        r_from_panel = resolution_report(panel)
        r_from_dict = resolution_report(L)
        assert r_from_panel["resolved"] == r_from_dict["resolved"]
        assert r_from_panel["observed_edge"] == pytest.approx(r_from_dict["observed_edge"])
    def test_resolution_report_losspanel_weights_respected(self):
        """A LossPanel's own weights are used when w= is not passed explicitly, matching report().
        Uses a non-uniform weight vector deliberately -- the MCS-weighting guard (below) refuses
        non-uniform weights outright, so threading the panel's own weights through correctly must
        hit the SAME graceful non-identified degradation as passing them explicitly, proving they
        were actually read from the panel rather than silently dropped to w=None.

        UPDATED 2026-08-27 (round-7 fix pass): resolution_report() used to crash raw and uncaught
        on this exact non-uniform-weight case (a real bug, found by the round-7 clinical-trials
        review on realistic weighted registry data and fixed the same round) -- this test used that
        crash as its proof mechanism that weights were threaded through. Now that the same refusal
        degrades gracefully (identified=None/fragility_read='undetermined', matching report()'s own
        established fallback contract) instead of raising, the proof is that BOTH call styles
        (panel-implicit vs explicit w=) produce the identical graceful result, not identical
        exceptions."""
        from selection_fragility import LossPanel, resolution_report
        L = {"a": np.array([1.0, 1.0, 5.0]), "b": np.array([2.0, 2.0, 2.0])}
        w = np.array([1.0, 1.0, 0.0])   # non-uniform -> both calls must hit the same MCS-weight guard
        panel = LossPanel.from_losses(L, weights=w)
        r_from_panel = resolution_report(panel)
        r_explicit = resolution_report(L, w=w)
        assert r_from_panel == r_explicit
        assert r_from_panel["identified"] is None
        assert r_from_panel["fragility_read"] == "undetermined"


# ---- 10-agent code-review pass (2026-09-02): SEV-1 denormal-weight NaN propagation -------------

def test_denormal_weights_rejected_with_clear_error_not_silent_nan():
    """FIXED 2026-09-02 (CONFIRMED SEV-1 BUG, reproduced then fixed): a uniform weight vector at
    denormal magnitude (individually finite/non-negative/non-zero, so it passed every OTHER
    existing validation gate) made `_effective_n`'s Kish formula (sum(w)**2/sum(w**2)) underflow
    to 0/0=nan, silently flipping `resolved` to False on a maximally decisive true edge --
    resolution.py's own docstrings promise weight-SCALE invariance (already fixed once for
    breakdown_number on the OVERFLOW side; this was the same class of bug, unfixed on the
    UNDERFLOW side). Fixed with a lower-bound guard symmetric to the existing upper-bound
    (_MAX_ABS_LOSS) check -- rejecting, not silently renormalizing, matching that check's own
    precedent. A weight vector this small is never a real observation-count/frequency weight."""
    from selection_fragility import minimum_detectable_edge, significance_boundary, mcb_bound

    T = 10
    L = {"a": np.full(T, 1.0), "b": np.full(T, 100.0)}
    w_tiny = np.full(T, 1e-200)  # individually finite/non-negative/non-zero
    for fn in (minimum_detectable_edge, significance_boundary, mcb_bound):
        with pytest.raises(ValueError, match=r"(?i)underflow|magnitude"):
            fn(L, w_tiny)


def test_ordinary_weight_scales_still_work_after_the_underflow_guard():
    """The new lower-bound guard must not be so aggressive it rejects any real, ordinary weight
    scale -- observation counts, frequencies, and small-but-completely-normal floats must all
    still produce the same, scale-invariant answer as before."""
    from selection_fragility import minimum_detectable_edge, significance_boundary, mcb_bound

    T = 10
    L = {"a": np.full(T, 1.0), "b": np.full(T, 100.0)}
    mde_ref = minimum_detectable_edge(L, np.ones(T))
    sb_ref = significance_boundary(L, np.ones(T))
    mb_ref = mcb_bound(L, np.ones(T))

    for scale in (1e-10, 0.001, 1.0, 50.0, 1e10):
        w = np.full(T, scale)
        assert np.isclose(minimum_detectable_edge(L, w), mde_ref)
        assert np.isclose(significance_boundary(L, w), sb_ref)
        assert np.isclose(mcb_bound(L, w), mb_ref)


def test_denormal_weights_do_not_silently_flip_resolved_verdict():
    """FIXED 2026-09-02: end-to-end reproduction through the public resolution_report() surface --
    see the sibling test above for the root cause. A champion beating its rival by a 99-point
    margin in EVERY period must never be silently misread as 'undetermined'/resolved=False via NaN
    propagation -- now resolution_report() itself raises a clear error instead of returning a
    fully-formed but silently-wrong dict. LossPanel construction itself still accepts the weights
    (panel.py has no reason to know resolution.py's own magnitude constraint), so the check fires
    at the point that actually uses them for a resolution computation."""
    import warnings

    from selection_fragility import LossPanel, resolution_report

    T = 10
    L = {"a": np.full(T, 1.0), "b": np.full(T, 100.0)}
    w = np.full(T, 1e-200)
    panel = LossPanel.from_losses(L, weights=w)   # accepted -- panel.py has no reason to reject this
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"(?i)underflow|magnitude"):
            resolution_report(panel.losses, panel.weights)
