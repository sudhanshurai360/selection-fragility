"""Stage 3 -- k*, Proposition 2.2 inversion, certified-tied subset. TOOL_TEST_PLAN.md S3.1-S3.5.

S3.1 (k* exactness) and S3.2/S3.3 (invariance properties) run TODAY against the existing
decision_breakdown()/fragility() -- these are regression tests for confirmed bugs in the CURRENT
shipped code, and some are EXPECTED TO FAIL until Stage 3 fixes land. S3.4 (Prop 2.2 inversion) and
S3.5 (certified subset) wait on new functions.
"""
import math
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
import numpy as np
import pytest

from selection_fragility import fragility, decision_breakdown, breakdown_number
from selection_fragility import pooled_winner, winner_stability

try:
    from selection_fragility import prop22_certifies, certified_tied_subset
    HAVE_PROP22 = True
except ImportError:
    HAVE_PROP22 = False


def brute_force_k_exact(L, w):
    """Exhaustive subset search in EXACT rational arithmetic -- the ground truth greedy k* must match."""
    w = [Fraction(x) for x in w]
    models = list(L)
    means = {}
    for m in models:
        vals = [Fraction(str(x)) for x in L[m]]
        num = sum(wi * vi for wi, vi in zip(w, vals))
        den = sum(w)
        means[m] = num / den
    champ = min(models, key=lambda m: means[m])
    T = len(w)
    for rival in models:
        if rival == champ:
            continue
        c = [w[t] * (Fraction(str(L[rival][t])) - Fraction(str(L[champ][t]))) for t in range(T)]
        total = sum(c)
        if total <= 0:
            continue  # champ doesn't even win this pair pooled -- k*=0 territory, handled separately
        found = None
        for k in range(0, T + 1):
            ok = False
            if k == 0:
                ok = total <= 0
            else:
                for S in combinations(range(T), k):
                    if total - sum(c[i] for i in S) <= 0:
                        ok = True
                        break
            if ok:
                found = k
                break
        if found is not None:
            yield rival, found


# ---- S3.1: k* exactness, brute-force cross-check, and confirmed regressions ----------------------
class TestKStarExactness:
    def test_brute_force_cross_check(self):
        rng = np.random.default_rng(0)
        n_checked = 0
        for trial in range(200):
            T = rng.integers(3, 9)
            K = rng.integers(2, 5)
            L = {chr(97 + i): rng.integers(0, 20, size=T).astype(float) for i in range(K)}
            w = np.ones(T)
            k_greedy, opp, _ = decision_breakdown(L, w)
            for rival, k_exact in brute_force_k_exact(L, w):
                if rival == opp:
                    assert k_greedy == k_exact, f"trial {trial}: greedy={k_greedy} exact={k_exact}"
                    n_checked += 1
        assert n_checked > 100, f"only {n_checked} pairs actually checked -- test is too weak"

    def test_greedy_stopping_condition_is_relative_floor_not_exact_zero(self):
        """REGRESSION (independent 'wild' review, 2026-08-17, blind re-implementation differential
        test): the greedy loop's mid-loop stopping check was `if s <= 0: break` -- an EXACT-ZERO
        comparison on a value built by repeatedly subtracting individually-rounded float64 terms.
        The sibling `test_brute_force_cross_check` above uses INTEGER losses (exactly representable
        in float64), so it structurally cannot trigger this -- decimal-quantized losses (a realistic
        shape for real-world reported metrics, e.g. MASE to 1 decimal place) can. Minimal deterministic
        repro found by an independent from-scratch reimplementation + Decimal-exact brute force:
        la=[6.1,2.1,0.1,5.1], lb=[7.1,5.1,6.1,4.1] has true minimum k*=2 (removing the top two
        contributions leaves a margin that is mathematically exactly 0), but the old code computed
        that margin as 4.44e-16 (positive, float residue) and removed a THIRD, unnecessary period,
        reporting k*=3 -- always overstating k* by exactly 1, i.e. making the decision look MORE
        robust than it truly is, the unsafe direction for a fragility-detection tool. Quantified at
        4.7-4.9% of trials on decimal-quantized random panels before the fix; 0/400 after."""
        la = np.array([6.1, 2.1, 0.1, 5.1])
        lb = np.array([7.1, 5.1, 6.1, 4.1])
        k, removed = breakdown_number(la, lb, np.ones(4))
        assert k == 2, f"expected the true minimum k*=2, got {k} (removed={removed})"

        # broader stress sweep against Decimal-exact brute-force ground truth
        from decimal import Decimal
        from itertools import combinations

        def brute_force_k(la, lb, w):
            la_d = [Decimal(str(x)) for x in la]
            lb_d = [Decimal(str(x)) for x in lb]
            w_d = [Decimal(str(x)) for x in w]
            T = len(la)
            M = sum(w_d[t] * (lb_d[t] - la_d[t]) for t in range(T))
            if M <= 0:
                return 0
            for kk in range(T + 1):
                for subset in combinations(range(T), kk):
                    s = M - sum(w_d[t] * (lb_d[t] - la_d[t]) for t in subset)
                    if s <= 0:
                        return kk
            return T

        rng = np.random.default_rng(0)
        mismatches = 0
        n = 150
        for i in range(n):
            r = np.random.default_rng(1000 + i)
            T = int(r.integers(3, 9))
            la_i = np.round(r.uniform(0, 10, T), 1)
            lb_i = np.round(r.uniform(0, 10, T), 1)
            w_i = np.ones(T)
            k_impl, _ = breakdown_number(la_i, lb_i, w_i)
            k_true = brute_force_k(la_i, lb_i, w_i)
            if k_impl != k_true:
                mismatches += 1
        assert mismatches == 0, f"{mismatches}/{n} decimal-quantized trials disagreed with brute-force ground truth"

    def test_exact_tie_gives_kstar_zero(self, exact_tie_panel):
        """THE CONFIRMED REGRESSION: an exact pooled tie must give k*=0 (no strict winner exists),
        not k*>=1. This is EXPECTED TO FAIL against current code until Stage 3 fixes it."""
        w = np.ones(4)
        k, opp, responsible = decision_breakdown(exact_tie_panel, w)
        assert k == 0, (
            f"got k*={k} on an EXACT tie (verified equal via Decimal); the function's own "
            "documented invariant says k*==0 means no strict winner exists"
        )

    def test_degenerate_margin_does_not_produce_huge_concentration(self, near_tie_float_panel):
        """THE CONFIRMED REGRESSION: fragility() emitted concentration=4.4e15 on a near-tie panel.

        TIGHTENED 2026-08-27 (Phase-2 holistic review, Section C -- this test's own docstring
        already predicted the fix): concentration used a bare `M > 0` guard, missing the same
        degenerate-margin floor breakdown_number()/`degenerate_margin` already apply -- so a tiny
        POSITIVE M (at or below floating-point noise) still divided through to a huge ratio. Now
        gated on the identical floor, so a degenerate margin produces exactly NaN, not merely
        "small" -- asserted directly instead of the old looser `abs(conc) < 1e6` bound."""
        w = np.ones(len(near_tie_float_panel["a"]))
        r = fragility(near_tie_float_panel, w)
        assert r.get("degenerate_margin") is True, (
            "fixture is expected to trip the degenerate-margin guard -- if this fails, either the "
            "fixture changed or the guard itself regressed"
        )
        conc = r.get("concentration", r.get("concentration_share"))
        assert conc is None or (isinstance(conc, float) and np.isnan(conc)), (
            f"degenerate margin must produce exactly NaN concentration, got {conc!r}"
        )

    def test_unrelated_large_magnitude_model_does_not_falsely_trigger_degenerate(self):
        """CONFIRMED REGRESSION (third-agent review of the fix above, same day). The first version
        of that fix gave `concentration`/`degenerate` a WHOLE-PANEL scale (`_loss_scale(L)`,
        averaged over every model including ones that never bind), inconsistent with
        breakdown_number() itself (the function that actually computes k*), which has always used
        a PER-PAIR scale. Demonstrated directly: a panel with a real, decisive, non-degenerate
        breakdown between two models plus an UNRELATED third model at a much larger magnitude
        reported degenerate_margin=True / concentration=NaN, even though breakdown_number() on the
        real pair alone correctly found a genuine k*. Fixed by giving concentration/degenerate the
        same per-pair scale breakdown_number() uses, so they can no longer disagree with the
        function that actually determines k*."""
        T = 20
        rng = np.random.default_rng(3)
        a = rng.normal(1.0, 0.05, T)
        b = a + 0.3                          # a real, decisive, non-degenerate margin at scale ~1
        z = np.full(T, 1e14)                 # unrelated, never-winning, huge-magnitude model --
                                              # large enough to actually inflate a whole-panel
                                              # scale past this margin's own floor (verified: 1e10
                                              # is NOT extreme enough to trigger the old bug here)
        L = {"a": a, "b": b, "z": z}
        w = np.ones(T)

        k_direct, _ = breakdown_number(a, b, w)
        r = fragility(L, w)

        assert r["degenerate_margin"] is False, (
            "an unrelated large-magnitude model in the panel must not make a real, decisive "
            "margin look degenerate"
        )
        assert r["k_star"] == k_direct, (
            f"fragility()'s k_star ({r['k_star']}) must match breakdown_number()'s own direct "
            f"computation on the same pair ({k_direct})"
        )
        assert np.isfinite(r["concentration"]), (
            f"expected a real, finite concentration on a non-degenerate margin, got "
            f"{r['concentration']!r}"
        )

    def test_dominance_no_fabricated_reversal(self, dominance_panel):
        """THE CONFIRMED REGRESSION: b weakly dominates a (never worse, strictly better once).
        No voting rule can disagree. per_period_winner must be 'b', reversal must be False."""
        w = np.ones(3)
        r = fragility(dominance_panel, w)
        assert r["pooled_winner"] == "b"
        assert r["per_period_winner"] == "b", (
            f"per_period_winner={r['per_period_winner']!r}, expected 'b' -- "
            "b is never worse and strictly better once, so a cannot win the plurality"
        )
        assert r["reversal"] is False, "reversal fabricated on a dominance panel"

    def test_identical_triple_no_fabricated_unique_winner(self, identical_triple_panel):
        """THE CONFIRMED REGRESSION: three byte-identical models must report a genuine tie, not a
        unique 'winner' picked by argmin-on-alphabetically-sorted-names."""
        w = np.ones(3)
        r = fragility(identical_triple_panel, w)
        plurality = r.get("plurality_winners", [r.get("per_period_winner")])
        assert len(plurality) == 3 or set(plurality) == {"a", "b", "c"}, (
            f"plurality_winners={plurality!r} on three identical models -- expected a 3-way tie"
        )

    def test_breakdown_number_tied_margin_prefers_earlier_period(self):
        """CONFIRMED REGRESSION (2026-08-28, found in the paper repo's Phase-2 holistic review,
        present unfixed in this package's own copy of breakdown_number too): `np.argsort(c)[::-1]`
        is not guaranteed stable for equal margin contributions, and the trailing [::-1] reverses
        the whole result -- two periods with the EXACT same contribution could be removed/named in
        an order that is an accidental byproduct of the sort algorithm, not a stated convention.

        5 periods with contributions [20, 5, 5, 5, -7]: period 0 removed first unambiguously;
        periods 1-3 tied at 5; period 4 negative (never removed). The greedy loop stops strictly
        INSIDE the tied group (removing 2 of the 3 -- k*=3 either way, only the SET differs). Fixed
        via `np.argsort(-c, kind="stable")`, which must remove {0, 1, 2} (the earlier two tied
        periods), leaving period 3 (the later one) un-removed -- not {0, 3, 2}, which the old
        reversed-unstable-sort convention produced."""
        la = np.zeros(5)
        lb = np.array([20.0, 5.0, 5.0, 5.0, -7.0])
        w = np.ones(5)
        k, removed = breakdown_number(la, lb, w)
        assert k == 3, f"k* must be 3 regardless of tie order, got {k}"
        assert removed == [0, 1, 2], (
            f"expected the EARLIER two tied periods (1, 2) removed and the LATER (3) left, got {removed!r}"
        )


# ---- S3.2: rename-invariance (the master property test) -------------------------------------------
class TestRenameInvariance:
    RENAME_SCHEMES = [
        lambda ms: {m: m[::-1] for m in ms},                                    # reversed strings
        lambda ms: dict(zip(ms, sorted(ms, reverse=True))),                     # reverse-alpha remap
        lambda ms: {m: str(i) for i, m in enumerate(ms)},                       # numeric-string names
        lambda ms: {m: f"éè{i}" for i, m in enumerate(ms)},           # unicode names
    ]

    @pytest.mark.parametrize("scheme_idx", range(4))
    def test_kstar_and_winner_invariant_to_renaming(self, random_panel, scheme_idx):
        """FIXED 2026-08-16 (caught by running the finished Stage 3 implementation against this test,
        before trusting the failure): the original compared the POINT-pick's `resp` (responsible
        periods) directly across a renaming. At seed=1/scheme=1 this legitimately differs -- k*=1 in
        BOTH cases, but TWO opponents tie for that k* ('m2'/'m3' before renaming), and the renaming
        flips which of the tied names sorts first, so a DIFFERENT (but equally valid) tied opponent
        is reported as the point pick, with its own distinct responsible period. That is not a bug:
        decision_breakdown's own docstring documents this exact case and recommends
        return_ties=True precisely because the point pick is an arbitrary tie-break among several
        opponents. The genuinely name-invariant facts are (1) k* itself, and (2) the SET of models
        tied for binding opponent, once mapped back through the renaming -- verified directly against
        model_confidence_set()-style set comparisons before writing this fix. Checking those instead
        of the single point-pick's `resp` is both a correct test AND still catches the original bug
        class (a data-dependent tie set computed differently depending on name order would still
        show up as a set mismatch here)."""
        violations = []
        for seed in range(25):
            L = random_panel(seed=seed, T=15, K=4)
            w = np.ones(15)
            k1, opp1, resp1, ties1 = decision_breakdown(L, w, return_ties=True)
            remap = self.RENAME_SCHEMES[scheme_idx](list(L))
            L2 = {remap[m]: v for m, v in L.items()}
            k2, opp2, resp2, ties2 = decision_breakdown(L2, w, return_ties=True)
            inv = {v: k for k, v in remap.items()}
            ties2_mapped_back = sorted(inv[t] for t in ties2)
            if k1 != k2 or sorted(ties1) != ties2_mapped_back:
                violations.append(seed)
        assert not violations, f"rename-invariance violated at seeds {violations} (scheme {scheme_idx})"

    def test_explicit_regressions_under_renaming(self, dominance_panel, identical_triple_panel):
        """The two confirmed name-order bugs, re-run under a renaming that changes the alphabetical
        winner -- proves the fix (once applied) is genuinely name-invariant, not just fixed for the
        specific letters 'a'/'b'/'c' used in the original bug reports."""
        w3 = np.ones(3)
        renamed_dom = {"zeta": dominance_panel["a"], "alpha": dominance_panel["b"]}
        r = fragility(renamed_dom, w3)
        assert r["per_period_winner"] == "alpha", "reversal bug persists under a name that sorts LAST"

        renamed_triple = {"zzz": identical_triple_panel["a"], "aaa": identical_triple_panel["b"],
                          "mmm": identical_triple_panel["c"]}
        r2 = fragility(renamed_triple, w3)
        plurality = r2.get("plurality_winners", [r2.get("per_period_winner")])
        assert len(plurality) == 3 or set(plurality) == {"zzz", "aaa", "mmm"}


# ---- S3.3: scale/shift invariance -----------------------------------------------------------------
class TestScaleShiftInvariance:
    @pytest.mark.parametrize("scale", [0.001, 1.0, 100.0, 1e6])
    def test_positive_scale_invariant(self, random_panel, scale):
        L = random_panel(seed=0, T=20, K=4)
        w = np.ones(20)
        k1, opp1, _ = decision_breakdown(L, w)
        L_scaled = {m: v * scale for m, v in L.items()}
        k2, opp2, _ = decision_breakdown(L_scaled, w)
        assert k1 == k2 and opp1 == opp2

    def test_additive_shift_invariant(self, random_panel):
        L = random_panel(seed=0, T=20, K=4)
        w = np.ones(20)
        k1, opp1, _ = decision_breakdown(L, w)
        shift = 500.0
        L_shifted = {m: v + shift for m, v in L.items()}
        k2, opp2, _ = decision_breakdown(L_shifted, w)
        assert k1 == k2 and opp1 == opp2

    @pytest.mark.xfail(reason="negative-scale sign validation not implemented yet -- Stage 0/3 shared "
                              "validator. Currently a negative scale silently flips 'lower is better' "
                              "with no warning; this should start passing once that validator lands.",
                       strict=False)
    def test_negative_scale_raises(self, random_panel):
        L = random_panel(seed=0, T=20, K=4)
        w = np.ones(20)
        L_flipped = {m: v * -1.0 for m, v in L.items()}
        with pytest.raises(ValueError):
            decision_breakdown(L_flipped, w)


# ---- S3.4: Proposition 2.2 inversion ---------------------------------------------------------------
@pytest.mark.skipif(not HAVE_PROP22, reason="prop22_certifies() not implemented yet (Stage 3)")
class TestProp22Inversion:
    @pytest.mark.parametrize("T,expected_k", [(10, 2), (13, 2), (14, 3), (20, 3), (30, 3),
                                              (36, 3), (120, 3), (1000, 3), (2000, 3)])
    def test_threshold_by_T(self, T, expected_k):
        """FIXED 2026-08-17 (independent 'wild' review, claims-vs-implementation audit): this used to
        reimplement the threshold formula inline rather than calling the shipped
        `_prop22_threshold` -- a regression in the ACTUAL implementation would not have been caught
        here, only a regression in this test's own parallel copy of the math. Now calls the real
        function directly, with the independent reimplementation kept alongside as a genuine
        cross-check (both must agree, not just the shipped function against its own expectation)."""
        from selection_fragility.prop22 import _prop22_threshold
        real_k = _prop22_threshold(T, z=1.96)
        assert real_k == expected_k, f"T={T}: shipped _prop22_threshold gave {real_k}, expected {expected_k}"

        # independent cross-check: the closed-form threshold reimplemented from the docstring's own
        # formula, not read from the source -- must agree with the shipped function above.
        z = 1.96
        k = 0
        while k + 1 < T and math.sqrt(T * (k + 1) / (T - (k + 1))) < z:
            k += 1
        assert k == real_k, f"T={T}: independent reimplementation ({k}) disagrees with shipped code ({real_k})"

    def test_kstar_leq_3_certifies_at_T14_plus(self, random_panel):
        L = random_panel(seed=0, T=36, K=2)
        w = np.ones(36)
        k, opp, _ = decision_breakdown(L, w)
        cert = prop22_certifies(k, T=36)
        if k <= 3:
            assert cert is True
        else:
            assert cert in (False, None)

    def test_kstar_zero_not_applicable(self, exact_tie_panel):
        assert prop22_certifies(0, T=4) in (None, "not_applicable")

    def test_kstar_equals_T_no_constraint(self):
        assert prop22_certifies(30, T=30) in (None, "not_applicable", False)

    def test_real_panel_19_of_24_certified(self, real_cells, real_series_list):
        """UPDATED 2026-08-29 (11-angle review finding): the panel shrank 13->12 series / 26->24
        cells when the main repo dropped UMCSENT for a confirmed U-Michigan licensing restriction
        (this package's own fixture was stale until this same pass). Recomputed directly against the
        real, current CSV (per this project's own [[feedback_recompute_from_source]] rule): 19 of 24,
        not the pre-UMCSENT-removal 21 of 26. The 5 non-certified cells: PI/mase (k=23), PI/rmsse
        (k=25) -- PI is the one genuinely IDENTIFIED series, consistent with a large, robust k* --
        plus PAYEMS/mase (k=4), PAYEMS/rmsse (k=4), UNEMPLOY/mase (k=4), all just above the T>=14
        threshold of 3. (BOPGEXP/mase, non-certified pre-removal, is now certified -- the panel's
        SARIMA fix in the same main-repo session changed its k* -- and is not one of the 5 above.)
        Provenance is not correctness: trust the direct recomputation over any prior count."""
        n_certified = 0
        for sid in real_series_list:
            for metric in ("mase", "rmsse"):
                L = real_cells[(sid, metric)]
                T = len(next(iter(L.values())))
                w = np.ones(T)
                k, opp, _ = decision_breakdown(L, w)
                if prop22_certifies(k, T=T):
                    n_certified += 1
        assert n_certified == 19, f"expected 19 of 24 certified (recomputed 2026-08-29), got {n_certified}"

    def test_zero_violations_against_realised_t(self, real_cells, real_series_list):
        """The empirical proof the certificate is honest: whenever it certifies k*<=3 as
        non-significant, the ACTUAL measured |t| on that panel must be < 1.96."""
        violations = 0
        for sid in real_series_list:
            for metric in ("mase", "rmsse"):
                L = real_cells[(sid, metric)]
                T = len(next(iter(L.values())))
                w = np.ones(T)
                k, opp, _ = decision_breakdown(L, w)
                if prop22_certifies(k, T=T):
                    means = {m: np.average(v, weights=w) for m, v in L.items()}
                    champ = min(means, key=means.get)
                    d = L[opp] - L[champ]
                    t_stat = d.mean() / (d.std(ddof=1) / math.sqrt(T))
                    if abs(t_stat) >= 1.96:
                        violations += 1
        assert violations == 0


# ---- S3.5: certified-tied subset ------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_PROP22, reason="certified_tied_subset() not implemented yet (Stage 3)")
class TestCertifiedSubset:
    def test_never_exceeds_K(self, random_panel):
        L = random_panel(seed=0, T=30, K=6)
        subset = certified_tied_subset(L)
        assert len(subset) <= 6

    def test_never_negative_size(self, random_panel):
        for seed in range(30):
            L = random_panel(seed=seed, T=30, K=6)
            assert len(certified_tied_subset(L)) >= 0

    def test_non_nesting_with_mcs_is_expected(self, real_cells, real_series_list):
        """EXPLICIT NON-REGRESSION: on at least one real cell, the certified subset is NOT a subset
        of the MCS (or vice versa) -- this is a KNOWN, CORRECT property (different criteria: iid
        population-sd vs block-bootstrap). A future 'fix' that forces nesting should be caught HERE
        as a deliberate behaviour change, not silently accepted.

        SERIES LIST CORRECTED 2026-08-16: the original hardcoded 5-series list (from an earlier
        exploratory pass) shows nesting on all 5 under this fresh Stage 3 implementation -- verified
        directly. RSAFS (already in this project's real_series_list, previously not checked here)
        genuinely diverges: certified_tied_subset includes 'theta', which the alpha=0.10 MCS excludes
        -- confirmed directly before widening this loop. Scanning the full real_series_list rather
        than a stale hand-picked subset both fixes the immediate failure and makes the test more
        thorough than the original."""
        from selection_fragility import model_confidence_set
        found_non_nesting = False
        for sid in real_series_list:
            L = real_cells[(sid, "mase")]
            certified = set(certified_tied_subset(L))
            mcs_surv, _ = model_confidence_set(L, alpha=0.10)
            if not (certified <= set(mcs_surv)) and not (set(mcs_surv) <= certified):
                found_non_nesting = True
                break
            if certified and not certified.issubset(set(mcs_surv)):
                found_non_nesting = True
                break
        assert found_non_nesting, (
            "expected at least one real cell where the certified subset and the MCS diverge "
            "(known property from tonight's measurement, 2 of the checked cells) -- if this now "
            "fails, the two criteria may have been silently unified, which is a behaviour change "
            "requiring explicit review, not a bug in this test"
        )

    def test_exact_tie_is_included_not_excluded(self, random_panel):
        """REGRESSION (independent 'wild' review, 2026-08-16): prop22_certifies(k=0, T) correctly
        returns None ("not_applicable" -- an exact tie has nothing for Prop 2.2 to certify), but the
        old `if prop22_certifies(...)` treated None as falsy and skipped `subset.add(m)` entirely for
        an exact tie -- the single most tied rival possible was silently excluded from its own
        "certified tied subset," while a rival the champion barely, fragilely beats (k*=1-2) was
        correctly included. An exact tie must never be excluded."""
        L = random_panel(seed=0, T=30, K=3)
        models = list(L)
        champ_key = models[0]
        L2 = dict(L)
        L2["exact_tie"] = np.asarray(L[champ_key], float).copy()
        subset = certified_tied_subset(L2)
        assert "exact_tie" in subset, f"exact tie silently excluded from {subset}"


class TestSurpriseConcentration:
    def test_d_max_over_median_is_one_not_inf_for_perfectly_boring_panel(self):
        """REGRESSION (independent 'wild' review, 2026-08-17, numerical-extremes lens): when every
        model achieves exactly 0 loss every period (b=0 and max(d)=0, the true 0/0 case), the old
        `else np.inf` fallback reported "infinitely surprising" for a panel with no surprise
        anywhere. b==0 with max(d)==0 means "exactly at baseline" -- the correct ratio is 1.0."""
        from selection_fragility import surprise_concentration
        boring = {"a": np.zeros(10), "b": np.zeros(10)}
        r = surprise_concentration(boring)
        assert r["d_max_over_median"] == 1.0

    def test_d_max_over_median_stays_inf_when_genuinely_unbounded(self):
        """b==0 but max(d)>0 (some period had real difficulty against a zero baseline) IS a
        genuinely unbounded ratio and must stay inf -- only the true 0/0 case changes."""
        from selection_fragility import surprise_concentration
        mixed = {"a": np.array([0., 0., 0., 5.]), "b": np.array([0., 0., 0., 6.])}
        r = surprise_concentration(mixed)
        assert r["d_max_over_median"] == float("inf")


class TestExtremeMagnitudeRefusal:
    def test_decision_breakdown_refuses_extreme_magnitude(self):
        """REGRESSION (independent 'wild' review, 2026-08-17, numerical-extremes lens): loss
        magnitudes beyond ~1e150-1e300 silently overflow float64 arithmetic in breakdown_number's
        margin/scale computation (`inf <= inf` evaluates True, misclassifying a maximally decisive
        win as k*=0 -- 'no strict winner'). A magnitude floor now refuses this cleanly instead."""
        huge = {"champ": np.full(30, 1.0), "rival": np.full(30, 1e300)}
        with pytest.raises(ValueError, match=r"(?i)magnitude"):
            decision_breakdown(huge, np.ones(30))


class TestBreakdownNumberWeightValidation:
    def test_negative_weight_raises(self):
        """REGRESSION (own-review pass, architecture lens, 2026-08-17): `breakdown_number` is a
        public, exported function taking raw arrays directly (not a dict), so it never went through
        `_validate_losses`'s weight checks the way every dict-based caller does. Verified directly: a
        negative weight placed exactly on model `la`'s one catastrophic period silently ran to
        completion and reported k*=10 built on an inverted contribution, instead of refusing."""
        T = 10
        A = np.full(T, 1.0); A[3] = 100.0
        B = np.full(T, 1.1)
        w = np.ones(T); w[3] = -1.0
        with pytest.raises(ValueError, match=r"(?i)negative"):
            breakdown_number(A, B, w)

    def test_all_zero_weight_raises(self):
        T = 10
        A = np.full(T, 1.0)
        B = np.full(T, 1.1)
        with pytest.raises(ValueError, match=r"(?i)zero"):
            breakdown_number(A, B, np.zeros(T))

    def test_nan_loss_raises(self):
        """REGRESSION (independent ML-engineer review, 2026-08-26): breakdown_number's weight
        validation (test_negative_weight_raises above) did not extend to the loss arrays themselves
        -- a NaN in `la`/`lb` sorted to the end of np.argsort rather than raising, and the greedy
        loop ran to completion on a margin that was never real, reporting a confident k* instead of
        refusing. `decision_breakdown` catches this indirectly via `_validate_losses`, but
        breakdown_number is public and reachable directly, bypassing that guard entirely."""
        T = 4
        la = np.array([1.0, np.nan, 1.0, 1.0])
        lb = np.full(T, 2.0)
        with pytest.raises(ValueError, match=r"(?i)finite|nan"):
            breakdown_number(la, lb, np.ones(T))


class TestFragilityModuleWeightAndBlockValidation:
    """REGRESSION (independent ML-engineer review, 2026-08-26): pooled_winner/winner_stability
    (and therefore decision_breakdown/fragility, which route through the same _validate_losses(L,
    w)) checked weight finiteness but not sign -- unlike breakdown_number, which panel.py's
    _validate_weights already covered, and unlike LossPanel-built panels, which are never exposed to
    raw negative weights at all. winner_stability's block>=T degeneracy (a circular block bootstrap
    at least as long as the sample is a pure rotation, so the winner can never change) was likewise
    unguarded and silently returned exactly 1.0."""

    L = {"a": np.array([1.0, 1.0, 1.0, 1.0]), "b": np.array([2.0, 2.0, 2.0, 2.0])}

    def test_pooled_winner_negative_weight_raises(self):
        with pytest.raises(ValueError, match=r"(?i)negative"):
            pooled_winner(self.L, w=np.array([-1.0, 1.0, 1.0, 1.0]))

    def test_winner_stability_negative_weight_raises(self):
        with pytest.raises(ValueError, match=r"(?i)negative"):
            winner_stability(self.L, w=np.array([-1.0, 1.0, 1.0, 1.0]), n_boot=20)

    def test_pooled_winner_all_zero_weight_raises(self):
        """REGRESSION (round-2 strict-code-correctness review, 2026-08-26): the negative-weight
        guard above closed one gap in _validate_losses, but an all-zero weight vector is
        all-finite and non-negative, so it sailed through and crashed later inside np.average
        with a bare, opaque ZeroDivisionError instead of the package's own domain error -- unlike
        panel.py's _validate_weights, which has rejected this exact case since 2026-08-17."""
        with pytest.raises(ValueError, match=r"(?i)all zero"):
            pooled_winner(self.L, w=np.zeros(4))

    def test_winner_stability_all_zero_weight_raises(self):
        with pytest.raises(ValueError, match=r"(?i)all zero"):
            winner_stability(self.L, w=np.zeros(4), n_boot=20)

    def test_decision_breakdown_all_zero_weight_raises(self):
        with pytest.raises(ValueError, match=r"(?i)all zero"):
            decision_breakdown(self.L, w=np.zeros(4))

    def test_fragility_all_zero_weight_raises(self):
        with pytest.raises(ValueError, match=r"(?i)all zero"):
            fragility(self.L, w=np.zeros(4))

    def test_winner_stability_block_equal_T_raises(self):
        with pytest.raises(ValueError, match=r"(?i)block"):
            winner_stability(self.L, n_boot=20, block=4)

    def test_winner_stability_block_greater_than_T_raises(self):
        with pytest.raises(ValueError, match=r"(?i)block"):
            winner_stability(self.L, n_boot=20, block=10)

    def test_winner_stability_default_block_still_works(self):
        """The new block>=T guard must not break the common, unguarded default-block call path."""
        stability, freq = winner_stability(self.L, n_boot=50, seed=0)
        assert 0.0 <= stability <= 1.0


# ---- 10-agent code-review pass (2026-09-02): additional adversarial cross-checks against the ------
# exact-Fraction brute-force oracle. The core greedy algorithm was independently stress-tested
# (500,000+ trials across two separate reviews) and found exact with zero mismatches; these three
# extend that coverage to shapes the existing S3.1 suite doesn't specifically target.

def _brute_force_k_pair(la, lb, w):
    """Exact-Fraction brute force, independent of the package's own implementation."""
    la = [Fraction(str(x)) for x in la]
    lb = [Fraction(str(x)) for x in lb]
    w = [Fraction(str(x)) for x in w]
    T = len(la)
    c = [w[t] * (lb[t] - la[t]) for t in range(T)]
    M = sum(c)
    if M <= 0:
        return 0
    for k in range(1, T + 1):
        for S in combinations(range(T), k):
            if M - sum(c[i] for i in S) <= 0:
                return k
    return T


def test_kstar_exact_under_negative_and_mixed_sign_losses():
    """The existing brute-force cross-check draws losses from a non-negative range. Nothing in
    breakdown_number/decision_breakdown forbids negative losses (only weights are sign-checked),
    and a real metric (e.g. a signed forecast-error difference, or a log-loss delta) can
    legitimately go negative. Cross-validates the greedy algorithm against exact-Fraction brute
    force on negative and mixed-sign panels specifically, since the degenerate-margin floor uses
    abs() and could plausibly behave differently in sign than in magnitude-only regimes."""
    rng = np.random.default_rng(2026)
    n_checked = 0
    for trial in range(200):
        T = int(rng.integers(3, 9))
        lo, hi = rng.choice([(-50, -1), (-20, 20), (1, 50)])
        la = rng.uniform(lo, hi, T)
        lb = rng.uniform(lo, hi, T)
        w = np.ones(T)
        k_pkg, _removed = breakdown_number(la, lb, w)
        k_exact = _brute_force_k_pair(la, lb, w)
        assert k_pkg == k_exact, f"trial {trial}: greedy={k_pkg} exact={k_exact} (la={la}, lb={lb})"
        n_checked += 1
    assert n_checked > 150


def test_kstar_exact_with_partially_zero_weighted_periods():
    """Existing weight tests check all-zero (rejected) and all-uniform weights, but not a MIX of
    zero- and positive-weight periods within the same panel -- a realistic shape (e.g. a period
    with zero observations still present in the array). A zero-weight period must never be
    removable (its margin contribution is always exactly 0) and must never count toward k*."""
    rng = np.random.default_rng(99)
    n_checked = 0
    for trial in range(150):
        T = int(rng.integers(4, 9))
        la = rng.uniform(0, 10, T)
        lb = rng.uniform(0, 10, T)
        w = rng.integers(0, 3, T).astype(float)   # some exact zeros
        if w.sum() == 0:
            w[0] = 1.0
        k_pkg, removed = breakdown_number(la, lb, w)
        k_exact = _brute_force_k_pair(la, lb, w)
        assert k_pkg == k_exact, f"trial {trial}: greedy={k_pkg} exact={k_exact}"
        assert all(w[i] > 0 for i in removed), "a zero-weighted period was removed"
        n_checked += 1
    assert n_checked > 100


def test_kstar_deterministic_and_correct_under_massive_tie_block():
    """Stresses the existing tie-break rule (prefer earlier periods) at scale (1000 periods tied
    at an identical contribution) to confirm: (1) k* is the exact ceiling-division count needed,
    not off by one from accumulated float summation over a wide tied block, (2) the removed set is
    deterministic across repeated calls, (3) ties are broken toward the EARLIEST periods as
    documented, not an artifact of sort-algorithm internals at scale."""
    T = 2000
    la = np.zeros(T)
    lb = np.zeros(T)
    lb[:1000] = 3.0          # 1000 periods tied at contribution 3.0
    lb[1000:1999] = -1.0     # never removable (negative contribution)
    lb[1999] = 0.5
    w = np.ones(T)

    k1, removed1 = breakdown_number(la, lb, w)
    k2, removed2 = breakdown_number(la, lb, w)
    assert k1 == k2 and removed1 == removed2, "breakdown_number is not deterministic across repeated calls"

    # true M = 1000*3 + 999*(-1) + 0.5 = 2001.5; need ceil(2001.5 / 3) = 668 removals of the tied block
    assert k1 == 668
    assert sorted(removed1) == list(range(668)), (
        "tie-break must prefer the EARLIEST periods within a tied contribution block"
    )


# ---- 10-agent code-review pass (2026-09-02): statistical/null-calibration layer -------------------

def test_prop22_extremal_construction_achieves_bound_with_equality():
    """The proof claims the bound sqrt(T*k/(T-k)) is SHARP, achieved in the limit by a panel with k
    periods of equal positive margin and the remaining T-k periods just below zero. Taking that
    limit to its boundary (T-k periods at exactly 0) and using the i.i.d. POPULATION-sd studentized
    statistic (ddof=0 -- the statistic the theorem is stated for, matching
    code/pipeline/realised_t.py's own `sd = d.std(ddof=0)`), the realised |t| must equal the bound
    to float precision at every valid (T,k). The strongest available affirmative check for the
    paper's one proven theorem."""
    def bound(T, k):
        return math.sqrt(T * k / (T - k))

    for T in (4, 5, 6, 10, 14, 20, 30, 50, 100, 500, 1000):
        for k in range(1, T):
            la = np.zeros(T)
            lb = np.concatenate([np.full(k, 1.0), np.zeros(T - k)])
            kstar, _ = breakdown_number(la, lb, np.ones(T))
            assert kstar == k, f"T={T},k={k}: breakdown_number gave k*={kstar}, expected {k}"
            d = lb - la
            t = abs(math.sqrt(T) * d.mean() / d.std(ddof=0))
            b = bound(T, k)
            assert abs(t - b) < 1e-9, f"T={T},k={k}: |t|={t} != bound={b} at the extremal construction"


@pytest.mark.skipif(not HAVE_PROP22, reason="prop22_certifies not implemented yet")
def test_prop22_bound_never_violated_random_search_equal_weight():
    """The bound must never be violated for ANY panel, not just a handful of fixed real cells.
    Random search across T, magnitude, and shape, using the theorem's own statistic (population
    sd). A single violation here would falsify the paper's proven theorem."""
    rng = np.random.default_rng(20260821)
    n_checked = 0
    for _ in range(50_000):
        T = int(rng.integers(2, 80))
        la = rng.normal(0, rng.uniform(0.01, 100), T)
        lb = rng.normal(0, rng.uniform(0.01, 100), T)
        w = np.ones(T)
        if np.average(lb, weights=w) <= np.average(la, weights=w):
            la, lb = lb, la
        k, _ = breakdown_number(la, lb, w)
        if k <= 0 or k >= T:
            continue
        d = lb - la
        sd = d.std(ddof=0)
        if sd <= 0:
            continue
        t = abs(math.sqrt(T) * d.mean() / sd)
        bound = math.sqrt(T * k / (T - k))
        n_checked += 1
        assert t <= bound + 1e-9, f"VIOLATION: T={T}, k*={k}, |t|={t} > bound={bound}"
        if prop22_certifies(k, T) is True:
            assert t < 1.96, f"false certification: T={T}, k*={k}, certified non-sig but |t|={t}"
    assert n_checked > 10_000, "search filtered out too many trials; widen the sampling ranges"


@pytest.mark.skipif(not HAVE_PROP22, reason="certified_tied_subset not implemented yet")
def test_certified_tied_subset_rejects_non_uniform_weights():
    """CONFIRMED CRITICAL BUG (2026-09-02 stress test), FIXED: certified_tied_subset(L, w=...) used
    to accept arbitrary non-uniform weights (a documented, supported feature elsewhere in this
    package) and thread them into a WEIGHTED breakdown_number(), then certify that weighted k*
    using prop22_certifies()'s threshold, which is only proved for the UNWEIGHTED i.i.d.
    population-sd statistic (confirmed by code/pipeline/realised_t.py in the companion paper repo,
    which always calls breakdown_number with np.ones(T)). Concretely: a T=12 MASE-like panel with
    plausible per-month observation-count weights has weighted k*=2 -- certifying "rival" as tied
    to the champion -- while the actual unweighted i.i.d.-studentized |t|=2.79 is comfortably
    significant at the 5% level the certificate claims to rule out. Random adversarial search
    (300,000 trials) found 909 (0.30%) such false certifications before the fix, with |t| as high
    as 3.32 while still "certified non-significant". Fixed by rejecting non-uniform weights
    outright at the one function that threads them into a Prop 2.2 certification -- the general
    prop22_certifies(k, T, z) utility itself is scope-agnostic (it has no way to know whether its
    k came from a weighted or unweighted computation), so the guard belongs here, at the caller
    that actually has the weight vector."""
    w = np.array([20, 41, 55, 34, 240, 36, 299, 261, 20, 230, 31, 260], dtype=float)
    champ = np.array([1.006, 1.019, 0.946, 0.96, 0.989, 1.093, 1.04, 1.064, 0.983,
                       1.054, 0.989, 0.999])
    rival = np.array([1.3, 1.069, 1.13, 1.153, 1.054, 1.206, 0.839, 1.361, 1.074,
                       1.169, 1.071, 0.973])
    T = 12
    L = {"champ": champ, "rival": rival}

    k, _ = breakdown_number(champ, rival, w)
    d = rival - champ
    t = abs(math.sqrt(T) * d.mean() / d.std(ddof=0))
    assert t >= 1.96, "sanity check on the fixture: the true gap must be genuinely significant"
    assert k <= _prop22_threshold_for_test(T), (
        "sanity check on the fixture: the weighted k* must actually be small enough that an "
        "unguarded certify call WOULD have falsely certified it (reproducing the pre-fix bug)"
    )

    with pytest.raises(ValueError, match=r"(?i)uniform weight"):
        certified_tied_subset(L, w=w)

    # the unweighted case must still work exactly as before -- this is not a blanket weight ban.
    subset_unweighted = certified_tied_subset(L)
    assert "rival" not in subset_unweighted, (
        f"unweighted certified_tied_subset(L) = {subset_unweighted} still includes 'rival', but "
        f"the true i.i.d. |t|={t:.3f} shows the gap IS significant -- this would mean the "
        f"UNWEIGHTED case itself is broken, not just the weighted one."
    )
    # a genuinely uniform (nonzero-constant) weight vector must still be accepted -- rescaling
    # every period by the same constant is a documented no-op, not the defect being guarded against.
    subset_uniform_w = certified_tied_subset(L, w=np.full(T, 3.0))
    assert subset_uniform_w == subset_unweighted


def _prop22_threshold_for_test(T, z=1.96):
    k = 0
    while k + 1 < T and math.sqrt(T * (k + 1) / (T - (k + 1))) < z:
        k += 1
    return k


def test_winner_stability_block_zero_is_not_silently_replaced_by_default():
    """CONFIRMED BUG (2026-09-02): `block = block or max(1, round(T**(1/3)))` treats an explicit
    block=0 as falsy and silently substitutes the default block length instead of using 0 or
    raising -- the identical bug class already fixed for decision_breakdown's `a=` parameter
    elsewhere in this file ("must test identity against None, not truthiness"). A caller who
    computes `block` programmatically and lands on 0 gets a silently different, unrequested
    bootstrap scheme with no error. Only requires that block=0 NOT be silently swapped for the
    default -- either raising ValueError or genuinely running with block=0 both satisfy it."""
    T = 5
    L = {"a": np.array([1., 2, 3, 4, 5]), "b": np.array([2., 1, 4, 3, 6])}
    s_default, _ = winner_stability(L, n_boot=3000, seed=0)
    try:
        s_block0, _ = winner_stability(L, n_boot=3000, seed=0, block=0)
    except ValueError:
        return  # raising is an acceptable fix
    assert s_block0 != s_default, (
        "winner_stability(block=0) silently fell back to the default block length "
        f"(both gave {s_default}) instead of honoring block=0 or raising."
    )


def test_winner_stability_direct_call_on_exact_tie_is_not_misleadingly_1():
    """CONFIRMED GAP (2026-09-02): fragility()'s own inline commentary calls winner_stability()==1.0
    on an exact tie "the single most misleading output this tool can produce" and guards against it
    -- but ONLY inside fragility(), via a degenerate-margin check computed before calling
    winner_stability. The public, independently-documented, independently-exported
    winner_stability() function has no equivalent guard and still returns exactly 1.0 (100% "rock
    solid") for models that are byte-identical every period."""
    L = {"a": np.full(8, 1.0), "b": np.full(8, 1.0), "c": np.full(8, 1.0)}
    stability, freq = winner_stability(L, n_boot=1000)
    assert not (stability == 1.0), (
        "winner_stability() returned exactly 1.0 for three byte-identical models -- the exact "
        "degenerate-tie case fragility.py's own commentary calls the single most misleading "
        "output this tool can produce. winner_stability() itself has no guard against it."
    )


def test_exchangeable_null_champion_is_uniform_across_models():
    """exchangeable_benchmark exists because winner_stability's null is misread as 1/K; its
    justification depends on the underlying null construction actually being exchangeable (no
    model privileged by construction). Draw many null panels exactly as exchangeable_benchmark
    does (K i.i.d.-identical models) and confirm the champion's identity is uniformly distributed
    via a chi-square goodness-of-fit test -- catches any accidental bias from tie-breaking,
    dict/array ordering, or RNG stream reuse across models."""
    from scipy.stats import chisquare

    K, T, n_rep = 6, 32, 12000
    counts = {f"m{i}": 0 for i in range(K)}
    for r in range(n_rep):
        g = np.random.default_rng(5_000_000 + r)
        L = {f"m{i}": g.normal(1.0, 0.30, T) for i in range(K)}
        counts[pooled_winner(L, np.ones(T))] += 1
    chi2, p = chisquare(list(counts.values()))
    assert p > 0.01, f"champion draw is not uniform across models: counts={counts}, chi2={chi2}, p={p}"


def test_winner_stability_large_T_runs_and_returns_valid_probability():
    """No existing test exercises winner_stability at T>=1000. Confirms it stays a well-defined
    probability in [0,1] and completes in reasonable time at panel sizes larger than any real
    series in this project's own data."""
    import time

    rng = np.random.default_rng(0)
    T = 1500
    L = {"a": rng.normal(1.0, 0.1, T), "b": rng.normal(1.03, 0.1, T), "c": rng.normal(1.05, 0.1, T)}
    t0 = time.time()
    stability, freq = winner_stability(L, n_boot=500)
    elapsed = time.time() - t0
    assert 0.0 <= stability <= 1.0
    assert abs(sum(freq.values()) - 1.0) < 1e-9
    assert elapsed < 10.0, f"winner_stability(T=1500) took {elapsed:.2f}s, too slow for interactive use"


def test_winner_stability_negative_block_behaves_as_documented_iid_case():
    """block<0 is truthy in Python, so `block or default` correctly keeps the caller's value
    (unlike block=0) and reaches the `if block <= 1` branch, the same code path as block=1
    (documented as "iid case: identical under either scheme"). Locks in that block=-1 and block=1
    give identical results, since a future refactor could easily change `block <= 1` to `block < 1`
    and silently break it."""
    # NOTE 2026-09-02: the original fixture ({"a":[1..6], "b":[2,1,5,3,6,4]}) turned out to be a
    # genuine exact pooled tie (both means are exactly 3.5) -- winner_stability() now correctly
    # returns nan for it (the degenerate-margin guard added this same session), which made this
    # test's `nan == nan` comparison spuriously fail even though both sides agreed. Swapped to a
    # panel with a real, non-degenerate winner so the comparison is meaningful again.
    T = 6
    L = {"a": np.array([1., 2, 3, 4, 5, 6]), "b": np.array([2., 1, 5, 3, 6, 10])}
    s_neg1, _ = winner_stability(L, n_boot=2000, seed=0, block=-1)
    s_pos1, _ = winner_stability(L, n_boot=2000, seed=0, block=1)
    assert np.isfinite(s_neg1) and np.isfinite(s_pos1), "fixture must not be degenerate"
    assert s_neg1 == s_pos1, (
        f"block=-1 ({s_neg1}) no longer matches block=1's documented iid behaviour ({s_pos1})"
    )


# ---- 10-agent code-review pass (2026-09-02), adversarial break-it findings ---------------------

def test_winner_stability_rejects_negative_n_boot_instead_of_fabricating_zero():
    T = 20
    rng = np.random.default_rng(0)
    L = {"a": rng.normal(1, 0.1, T), "b": rng.normal(1.1, 0.1, T)}
    with pytest.raises(ValueError):
        winner_stability(L, n_boot=-5)


@pytest.mark.parametrize("k,z", [
    (float("nan"), 1.96),
    (3, float("nan")),
    (3, -5.0),
])
def test_prop22_certifies_validates_k_and_z(k, z):
    with pytest.raises((ValueError, TypeError)):
        prop22_certifies(k, 20, z=z)


def test_exchangeable_benchmark_nonpositive_n_rep_raises_clear_error():
    """Currently crashes with a raw, low-level numpy IndexError deep inside np.percentile rather
    than a clear validation message -- inconsistent with this package's otherwise careful
    validation style. Accepts either today (documents the current crash type) so this test passes
    now and keeps passing once the message is cleaned up to a ValueError."""
    from selection_fragility.fragility import exchangeable_benchmark

    with pytest.raises((ValueError, IndexError)):
        exchangeable_benchmark(K=3, T=20, n_rep=-2, n_boot=50)


def test_winner_stability_zero_n_boot_raises_clear_error_not_bare_zerodivision():
    """n_boot=0 currently crashes with a bare, unhelpful ZeroDivisionError. Accepts either today
    (documents the current crash type) so this test passes now and keeps passing once the message
    is cleaned up to a ValueError."""
    T = 10
    L = {"a": np.arange(T, dtype=float), "b": np.arange(T, dtype=float) + 1}
    with pytest.raises((ValueError, ZeroDivisionError)):
        winner_stability(L, n_boot=0)


def test_mutating_panel_losses_in_place_after_construction_is_still_caught():
    """POSITIVE REGRESSION LOCK, confirmed SAFE (2026-09-02): validation is NOT a one-time
    construction-time check a caller can silently defeat by mutating panel.losses['model'][:] = ...
    afterward -- every public function re-validates finiteness on every call, so a NaN introduced
    post-construction is still caught. Locks in this genuinely useful safety property so a future
    'optimize away redundant validation' change doesn't quietly remove it."""
    from selection_fragility import LossPanel, report

    T = 10
    rng = np.random.default_rng(0)
    L = {"a": rng.normal(1, 0.1, T), "b": rng.normal(1.2, 0.1, T)}
    panel = LossPanel.from_losses(L)
    panel.losses["a"][:] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pooled_winner(panel.losses, panel.weights)
    with pytest.raises(ValueError, match="non-finite"):
        report(panel)


def test_winner_stability_concurrent_distinct_seeds_on_shared_panel_matches_sequential():
    """NEW concurrency scenario (2026-09-02): many threads calling winner_stability concurrently
    on the SAME shared LossPanel.losses dict, each with its OWN distinct seed -- confirms
    np.random.default_rng(seed) is genuinely locally-scoped (no hidden shared/global RNG state)
    even under real concurrent execution, not just sequential reuse."""
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(0)
    T = 25
    L = {"a": rng.normal(1, 0.15, T), "b": rng.normal(1.05, 0.15, T), "c": rng.normal(1.1, 0.15, T)}

    def run(seed):
        return winner_stability(L, n_boot=800, seed=seed)

    seeds = list(range(30)) * 4
    sequential = {s: run(s) for s in set(seeds)}
    mismatches = []

    def worker(s):
        got = run(s)
        if got[0] != sequential[s][0] or got[1] != sequential[s][1]:
            mismatches.append(s)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, seeds))

    assert mismatches == []


# ---- Round-2 10-agent review (2026-09-02) -----------------------------------------------------

def test_mcs_rejects_alpha_outside_unit_interval():
    """CONFIRMED SEVERE (2026-09-02): identify._validate_alpha was added 2026-08-16 for the
    "alpha=10 meaning 10%" practitioner slip and wired into identify.py and resolution.py -- but
    never into mcs.py, the module that actually consumes alpha. Verified pre-fix on a T=20/K=4
    panel: alpha=10 and alpha=nan each returned a POINT-IDENTIFIED singleton (p >= 10 and p >= nan
    are both always False, so every model is eliminated); alpha=0.0 and alpha=-1 never eliminated
    anything. All silent and plausible-looking."""
    import numpy as np
    import pytest
    from selection_fragility.mcs import mcs, model_confidence_set

    M = np.random.default_rng(0).normal(1, 0.2, (20, 4))
    for bad in (10, 100.0, float("nan"), -1, 0.0, 1.0):
        with pytest.raises(ValueError, match=r"(?i)alpha"):
            mcs(M, alpha=bad, B=200, block=3, seed=0)
    with pytest.raises(ValueError, match=r"(?i)alpha"):
        model_confidence_set({f"m{i}": M[:, i] for i in range(4)}, alpha=10)
    surv, _ = mcs(M, alpha=0.10, B=200, block=3, seed=0)   # a valid alpha must still work
    assert len(surv) >= 1


def test_certified_tied_subset_includes_byte_identical_models():
    """CONFIRMED SEVERE (2026-09-02): prop22_certifies returns None for k<=0 (a 'not applicable'
    sentinel), and `if prop22_certifies(...)` treated None as False -- so k*=0, the STRONGEST
    possible evidence of a tie, was the one case that failed certification. Reproduced: five
    byte-identical models returned a certified tied subset of size 1, reading as point
    identification, while model_confidence_set returned all five."""
    import numpy as np
    from selection_fragility.prop22 import certified_tied_subset
    from selection_fragility import model_confidence_set

    base = np.random.default_rng(0).normal(1, 0.2, 20)
    L = {f"m{i}": base.copy() for i in range(5)}
    assert certified_tied_subset(L) == sorted(L) == sorted(model_confidence_set(L)[0])


def test_certified_tied_subset_is_monotone_in_tie_strength():
    """The invariant the fix restores: a rival tied MORE strongly (smaller k*) must be certified
    whenever a rival tied less strongly is. Pre-fix the relation was non-monotone at exactly k*=0.

    FIXED 2026-09-05 (round-3 audit): with seed=11 and TWO-SIDED noise on r_near
    (`rng.normal(0, 1e-4, 24)`), r_near's mean loss happened to land BELOW champ/r_exact's by
    chance, so `pooled_winner` picked r_near, not champ/r_exact -- and breakdown_number(r_near,
    champ) / breakdown_number(r_near, r_exact) both came back k*=5, never 0. The `if k == 0` branch
    this test exists to exercise was never entered; it passed identically whether or not the
    certification fix under test was even present (confirmed: reverting `certified_tied_subset` to
    its pre-fix behavior does not fail this test). Using ONE-SIDED (non-negative) noise on r_near
    guarantees mean(r_near) >= mean(champ) == mean(r_exact) (equality has probability 0 under
    continuous noise), so champ/r_exact -- the exact byte-identical tie -- always wins the pooled
    vote, and whichever of the two does NOT get that name is guaranteed a true k*=0 against it,
    reliably exercising the boundary this test is meant to check."""
    import numpy as np
    from selection_fragility.fragility import breakdown_number, pooled_winner
    from selection_fragility.prop22 import certified_tied_subset

    rng = np.random.default_rng(11)
    champ = rng.normal(1.0, 0.2, 24)
    L = {"champ": champ, "r_exact": champ.copy(), "r_near": champ + np.abs(rng.normal(0, 1e-4, 24))}
    cert = set(certified_tied_subset(L))
    w = np.ones(24)
    winner = pooled_winner(L, w)
    assert winner in ("champ", "r_exact"), (
        f"one-sided noise on r_near should guarantee champ/r_exact wins the pooled vote; got {winner}")
    a = np.asarray(L[winner], float)
    ks = {m: breakdown_number(a, np.asarray(L[m], float), w)[0] for m in L if m != winner}
    exact_rival = ("champ" if winner == "r_exact" else "r_exact")
    assert ks[exact_rival] == 0, (
        f"{exact_rival} is byte-identical to the champion and must have k*=0; got {ks}")
    assert exact_rival in cert, (
        f"{exact_rival} at k*=0 (maximally tied) must be certified; got {sorted(cert)}")
    for m, k in ks.items():
        if k == 0:
            assert m in cert, f"{m} at k*=0 (maximally tied) must be certified; got {sorted(cert)}"


def test_certified_tied_subset_still_refuses_a_genuinely_separated_rival():
    """Guard that the k*=0 admission did not weaken certification: a clearly-separated rival must
    still be excluded."""
    import numpy as np
    from selection_fragility.prop22 import certified_tied_subset

    rng = np.random.default_rng(5)
    L = {"champ": rng.normal(1.0, 0.05, 40), "far": rng.normal(2.0, 0.05, 40)}
    assert certified_tied_subset(L) == ["champ"]
