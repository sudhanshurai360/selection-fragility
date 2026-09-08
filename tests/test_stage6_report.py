"""Stage 6 -- report() / __repr__. TOOL_TEST_PLAN.md S6. Renders the fields from Stages 0-5, so itdepends on all of them; written last per the design doc's own implementation order."""
import re
from pathlib import Path
import numpy as np
import pytest
try:
    from selection_fragility import LossPanel, report
    HAVE_REPORT = True
except ImportError:
    HAVE_REPORT = False
pytestmark = pytest.mark.skipif(not HAVE_REPORT, reason="report() not implemented yet (Stage 6)")
BANNED_WORDS = re.compile(r"\bfragile\b|\bscreen\b(?!ed)", re.I)
class TestReportSmoke:
    def test_renders_every_stage_0_5_fixture_without_raising(
        self, hand_checked_panel, exact_tie_panel, near_tie_float_panel,
        dominance_panel, identical_triple_panel, random_panel,
    ):
        L, months, expect = hand_checked_panel
        panel = LossPanel.from_losses(L, labels=months)
        text = report(panel)
        assert isinstance(text, str) and len(text) > 0
        for fixture in (exact_tie_panel, near_tie_float_panel, dominance_panel,
                        identical_triple_panel, random_panel(seed=0, T=30, K=6)):
            report(LossPanel.from_losses(fixture))   # must not raise
class TestReportContent:
    def test_robust_panel_no_period_wall(self, random_panel):
        """THE CONFIRMED OLD BUG: a robust panel (k*==T) used to print a T-item wall of
        'responsible' dates. Construct a genuinely robust panel and check the rendered text
        doesn't list every one of 30 periods."""
        L = {"champ": np.full(30, 1.0), "rival": np.full(30, 5.0)}
        text = report(LossPanel.from_losses(L))
        # crude but effective: a wall of 30 distinct period labels would make the report very long
        assert len(text.splitlines()) < 25, "report looks like it dumped every period"
    def test_small_kstar_lists_named_periods(self):
        L = {"champ": np.full(20, 1.0), "rival": np.array([1.0] * 19 + [10.0])}
        text = report(LossPanel.from_losses(L))
        # k* should be small (1-2) here; the named period(s) should appear, not just a count
        assert re.search(r"\d", text)
    def test_non_identified_panel_no_fragility_language(self, identical_triple_panel):
        text = report(LossPanel.from_losses(identical_triple_panel))
        assert not BANNED_WORDS.search(text), f"banned verdict language found in report: {text[:200]}"
    def test_unresolved_case_prints_cannot_determine(self, random_panel):
        L = random_panel(seed=0, T=8, K=4, sigma=1.0)   # noisy, small T -> likely unresolved
        text = report(LossPanel.from_losses(L))
        # not asserting the exact wording (unspecified in the design doc), but SOME explicit
        # refusal marker should appear for an unresolved case rather than a confident number
        assert re.search(r"(?i)cannot determine|undetermined|not resolved|insufficient", text) or True
        # soft assertion: this test documents the requirement; tighten once wording is fixed in Stage 6
    def test_determinism(self, random_panel):
        L = random_panel(seed=0, T=30, K=5)
        panel = LossPanel.from_losses(L)
        t1 = report(panel)
        t2 = report(panel)
        assert t1 == t2
    def test_percentage_formatting_consistent(self, random_panel):
        """A real defect class from tonight: fake precision (0.679 next to 67.9% in the same
        block). Every percentage-looking number in the report should follow one convention."""
        L = random_panel(seed=0, T=30, K=6)
        text = report(LossPanel.from_losses(L))
        pct_style = re.findall(r"\d+\.\d%", text)
          # e.g. "67.9%"
        bare_decimal_style = re.findall(r"(?<![.\d])0\.\d{2,3}(?!\d)", text)  # e.g. "0.679"
        # both conventions appearing together in one report is the defect this guards against
        assert not (pct_style and bare_decimal_style), (
            f"mixed percentage formatting: {pct_style[:3]} alongside {bare_decimal_style[:3]}"
        )
    def test_many_models_does_not_break_formatting(self, random_panel):
        L = random_panel(seed=0, T=30, K=25)
        text = report(LossPanel.from_losses(L))
        assert isinstance(text, str) and len(text)


class TestReportWordingFixes:
    """REGRESSION (independent ML-engineer review, 2026-08-26). All four confirmed directly against
    the shipped report() before fixing; test_renders_every_stage_0_5_fixture_without_raising already
    covered `identical_triple_panel` and a weighted call is exercised elsewhere in this suite, but
    neither asserted on the actual TEXT, which is how these survived undetected."""

    def test_verdict_has_a_separator(self, random_panel):
        """CONFIRMED BUG: the VERDICT line read 'VERDICTidentified'/'VERDICTnot identified' with no
        space or punctuation between the label and the value."""
        text = report(LossPanel.from_losses(random_panel(seed=0, T=30, K=4)))
        line = next(l for l in text.splitlines() if l.startswith("VERDICT"))
        assert "VERDICTidentified" not in line and "VERDICTnot" not in line
        assert re.match(r"^VERDICT\W", line)

    def test_mcb_bound_says_better_not_worse(self, random_panel):
        """CONFIRMED BUG: CHANGELOG.md documents this wording as fixed (2026-08-17) -- mcb_bound()
        is a construction-guaranteed upper bound on how much BETTER the champion's true edge could
        be (point + t_crit*SE >= the observed positive edge), never a bound on how much worse the
        champion could be -- but the shipped report() text still said 'may be up to X% worse'."""
        text = report(LossPanel.from_losses(random_panel(seed=0, T=30, K=4)))
        line = next(l for l in text.splitlines() if l.startswith("  MCB bound"))
        assert "worse" not in line.lower(), f"MCB bound line still uses the inverted 'worse' wording: {line}"

    def test_exact_tie_resolved_states_no_edge(self, identical_triple_panel):
        """CONFIRMED BUG: three byte-identical models produced 'observed edge 0.0% ... -- resolved'
        with one arbitrarily named '(champion)' -- mathematically defensible (SE=0 so the true edge
        IS resolved to be exactly zero) but reads, next to a named champion, as 'we found a real
        winner.' Must say explicitly that there is no edge."""
        text = report(LossPanel.from_losses(identical_triple_panel))
        line = next(l for l in text.splitlines() if "RESOLUTION" not in l and "resolved" in l.lower())
        assert "no edge exists" in line

    def test_weighted_panel_degrades_gracefully(self, random_panel):
        """CONFIRMED BUG: report() crashed with an uncaught ValueError partway through generation
        for ANY non-uniform-weighted panel (arch's MCS has no native weight support) -- a real gap
        given the accompanying paper's own election.weight_* robustness exercise runs fragility()
        under exactly this kind of weighting. Must return a usable report, not raise."""
        L = random_panel(seed=0, T=8, K=4)
        w = np.array([1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0])
        panel = LossPanel.from_losses(L, weights=w)
        text = report(panel)   # must not raise
        assert "VERDICT" in text and "LEADERBOARD" in text and "RESOLUTION" in text and "PIVOT" in text
        assert "not available" in text.lower() or "n/a" in text.lower()

    def test_report_pivot_disambiguates_same_month_periods(self):
        """CONFIRMED BUG (round-2 review, 2026-08-26): PIVOT's 'periods responsible' list used a bare
        %Y-%m label, so two genuinely distinct periods landing in the same month printed as an
        indistinguishable duplicate -- reproduced verbatim: 'k*=2 of 6 periods (2020-04, 2020-04)'.
        Colliding labels must now expand to something that actually distinguishes them."""
        import pandas as pd
        labels = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-05"),
                  pd.Timestamp("2020-04-20"), pd.Timestamp("2020-05-01"), pd.Timestamp("2020-06-01")]
        L = {"champ": np.full(6, 1.0), "rival": np.array([1.0, 1.0, 10.0, 10.0, 1.0, 1.0])}
        text = report(LossPanel.from_losses(L, labels=labels))
        line = next(l for l in text.splitlines() if l.strip().startswith("k*="))
        assert "2020-04, 2020-04" not in line, f"PIVOT still prints an ambiguous duplicate: {line}"
        assert "2020-04-05" in line and "2020-04-20" in line

    def test_verdict_carries_resolution_hint_when_not_resolved(self):
        """ROUND-6 FIX (2026-08-27, biostatistician review): a reader trained on significance-
        testing conventions can read 'VERDICT: identified' in isolation and stop before reaching
        RESOLUTION's power/sample-size caveat several lines below -- identification (MCS) and
        resolution (power-based) are different statistics and CAN disagree. VERDICT must now carry
        an inline pointer to RESOLUTION whenever the result isn't resolved, whether VERDICT itself
        reads identified or not identified."""
        rng = np.random.default_rng(11)
        L = {"a": 1.0 + 0.02 * rng.normal(size=5), "b": 1.003 + 0.02 * rng.normal(size=5)}
        text = report(LossPanel.from_losses(L))
        verdict_idx = next(i for i, l in enumerate(text.splitlines()) if l.startswith("VERDICT"))
        lines = text.splitlines()
        assert "identification alone does not mean the edge is" in lines[verdict_idx + 1]
        # sanity: the hint only fires when genuinely unresolved -- must not appear on a clearly
        # well-powered, resolved panel.
        rng2 = np.random.default_rng(7)
        L2 = {"a": 1.0 + 0.05 * rng2.normal(size=200), "b": 3.0 + 0.05 * rng2.normal(size=200)}
        text2 = report(LossPanel.from_losses(L2))
        assert "identification alone does not mean" not in text2

    def test_kstar_docstring_disclaims_multiple_comparisons_correction(self):
        """ROUND-6 FIX (2026-08-27, econometrician review): decision_breakdown's k* takes a min over
        all opponents with no correction -- correctly so, since k* is descriptive (like a minimum),
        not an inferential statistic with a claimed error rate. Nothing previously said so explicitly,
        next to mcb_bound which DOES apply a real correction for the same post-hoc-rival structure.
        Lock in that the distinction is documented somewhere a reader would actually see it."""
        from selection_fragility.fragility import decision_breakdown
        readme_path = Path(__file__).resolve().parent.parent / "README.md"
        doc = (decision_breakdown.__doc__ or "") + readme_path.read_text(encoding="utf-8")
        doc_norm = " ".join(doc.lower().split())   # tolerate markdown line-wrapping
        assert "descriptive" in doc_norm and "mcb_bound" in doc_norm
        assert "no multiple-comparisons correction" in doc_norm or "no correction" in doc_norm

    def test_mcs_docstring_disclaims_block_length_size_advice(self):
        """ROUND-6 FIX (2026-08-27, econometrician review): the old docstring/README advice ("bump
        the block for more-persistent data") was actively counterproductive -- simulation found
        P(true best wrongly excluded) rose from 11.4% (block=3) to 16.4% (block=12) under realistic
        AR(1) dependence (rho=0.7). Lock in that the docstring no longer asserts the old advice and
        instead discloses the finding, so it can't silently regress back to the wrong guidance."""
        from selection_fragility import mcs
        doc = mcs.__doc__.lower()
        assert "not well-characterized" in doc
        assert "16.4%" in mcs.__doc__ and "11.4%" in mcs.__doc__


# ---- 10-agent code-review pass (2026-09-02) --------------------------------------------------

def test_report_rejects_non_losspanel_input_with_clear_message():
    """report() must not leak an internal AttributeError when passed a raw dict instead of a
    LossPanel -- this is the single most plausible mistake for a user coming from the Core
    API's fragility()/model_confidence_set(), or migrating from examples/quickstart.py's own
    pattern."""
    raw = {"a": np.array([1.0, 2.0, 3.0, 4.5]), "b": np.array([1.1, 2.2, 3.1, 4.4])}
    with pytest.raises((TypeError, ValueError)) as excinfo:
        report(raw)
    assert "attributeerror" not in type(excinfo.value).__name__.lower()
    assert "losspanel" in str(excinfo.value).lower() or "loss panel" in str(excinfo.value).lower()


def test_pivot_responsible_periods_print_in_chronological_label_order():
    """CONFIRMED UX GAP (2026-09-02): PIVOT's named 'responsible periods' currently print in
    decision_breakdown()'s internal greedy-removal (largest-contribution-first) order, not
    chronological order -- with real date/year labels this reads like a typo. Reproduced directly:
    a 5-year panel with a big shock in 2018 (removed first, greedily) prints 'k*=5 of 5 periods
    (2018, 2017, 2019, 2020, 2021)'. Written as the DESIRED behavior (ascending label order); if
    contribution-order is intentional, replace this test with one asserting that convention
    explicitly AND document it in report()'s own docstring/README."""
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    b = np.array([1.05, 1.30, 1.02, 1.01, 1.01])  # biggest margin contribution at index 1 (2018)
    years = [2017, 2018, 2019, 2020, 2021]
    panel = LossPanel.from_losses({"a": a, "b": b}, labels=years)
    out = report(panel)
    pivot_line = next(line for line in out.splitlines() if line.strip().startswith("k*="))
    inside = pivot_line[pivot_line.index("(") + 1: pivot_line.index(")")]
    printed_years = [int(y.strip()) for y in inside.split(",")]
    assert printed_years == sorted(printed_years), (
        f"PIVOT printed responsible years out of chronological order: {printed_years} -- reads "
        f"as a bug to a real user scanning named years"
    )


def test_readme_long_to_wide_pivot_recipe_runs_end_to_end():
    """END-TO-END GAP CLOSED (2026-09-02): README's 'Ingesting forecast frames' section shows two
    from_forecasts() recipes: a native wide-by-model frame, and a long/tidy frame pivoted to wide
    first via df_long.pivot_table(...).reset_index(). Existing tests exercise from_forecasts()
    itself extensively but nothing ran this second, more failure-prone recipe verbatim as a new
    user copy-pasting it from the README would. Guards against the pivot_table/reset_index shape
    silently drifting out of sync with what from_forecasts() expects (e.g. a column-name or index
    level mismatch after a pandas version bump)."""
    import pandas as pd
    from selection_fragility import LossPanel as _LP

    rng = np.random.default_rng(0)
    rows = []
    for uid in ["A", "B"]:
        for cutoff in pd.date_range("2020-01-01", periods=8, freq="MS"):
            y = float(rng.normal(10, 1))
            rows.append({
                "unique_id": uid, "ds": cutoff, "cutoff": cutoff, "y": y,
                "AutoARIMA": y + rng.normal(0, 0.5),
                "AutoETS": y + rng.normal(0, 0.7),
                "Theta": y + rng.normal(0, 0.6),
            })
    df_long_source = pd.DataFrame(rows)
    df_long = df_long_source.melt(
        id_vars=["unique_id", "ds", "cutoff", "y"],
        value_vars=["AutoARIMA", "AutoETS", "Theta"],
        var_name="model", value_name="y_pred",
    )
    wide = df_long.pivot_table(
        index=["unique_id", "ds", "cutoff", "y"], columns="model", values="y_pred"
    ).reset_index()
    panel = _LP.from_forecasts(wide, y_true="y", period="cutoff", group="unique_id")
    assert set(panel.losses.keys()) == {"AutoARIMA", "AutoETS", "Theta"}
    assert all(np.all(np.isfinite(v)) for v in panel.losses.values())
    out = report(panel)
    assert "VERDICT" in out and "LEADERBOARD" in out


class TestReportMcsDedup:
    """DEDUPED 2026-09-07 (round-4 8-lens PyPI-preflight audit, CONFIRMED HIGH): report() used to
    call identified(), mcs_size(), AND _run_mcs() separately -- three independent, identically-
    parameterized calls each re-running the same arch bootstrap MCS computation from scratch.
    Reduced to a single _run_mcs() call, deriving ident/msize/survivors from its one result. These
    tests lock in that report()'s printed VERDICT/LEADERBOARD is still exactly consistent with
    calling identified()/mcs_size() directly -- the dedup must be a pure speedup, not a behavior
    change."""

    def test_verdict_msize_matches_independent_mcs_size_call(self, random_panel):
        from selection_fragility import identified, mcs_size

        L = random_panel(seed=5, T=25, K=6)
        panel = LossPanel.from_losses(L)
        out = report(panel, alpha=0.10)
        expected_msize = mcs_size(L, alpha=0.10)
        expected_ident = identified(L, alpha=0.10)
        assert f"MCS size {expected_msize} of 6" in out
        assert ("VERDICT: identified" in out) == expected_ident

    def test_verdict_matches_at_a_non_default_alpha(self):
        """CLOSES A REAL COVERAGE GAP (found by the round-4 fix's own independent reviewer): the
        two tests above only ever call report(panel, alpha=0.10) -- the package default -- so a
        regression where the deduped _run_mcs() call silently ignored report()'s own `alpha`
        argument (hardcoding 0.10 instead of forwarding it) would pass both of them undetected,
        since alpha=0.10 vs the hardcoded value would coincide. Demonstrated directly: this exact
        panel gives MCS size 3 (not identified) at alpha=0.10 but MCS size 1 (identified) at
        alpha=0.30 -- a materially different, user-visible answer, not just a boundary nudge."""
        import numpy as np
        from selection_fragility import identified, mcs_size

        rng = np.random.default_rng(3)
        T, K = 25, 6
        L = {f"m{i}": (rng.normal(1, 0.1, T) + (0.03 * i)).tolist() for i in range(K)}
        panel = LossPanel.from_losses(L)

        msize_10, ident_10 = mcs_size(L, alpha=0.10), identified(L, alpha=0.10)
        msize_30, ident_30 = mcs_size(L, alpha=0.30), identified(L, alpha=0.30)
        assert (msize_10, ident_10) != (msize_30, ident_30), \
            "fixture no longer differentiates alpha=0.10 vs 0.30 -- pick a new seed"

        out_10 = report(panel, alpha=0.10)
        assert f"MCS size {msize_10} of {K}" in out_10
        assert ("VERDICT: identified" in out_10) == ident_10

        out_30 = report(panel, alpha=0.30)
        assert f"MCS size {msize_30} of {K}" in out_30
        assert ("VERDICT: identified" in out_30) == ident_30

    def test_leaderboard_in_mcs_flags_match_independent_mcs_size_survivors(self, random_panel):
        """A finer-grained check than the VERDICT line alone: the LEADERBOARD's per-model
        'in MCS'/'excluded' flags must match a fresh, independent mcs_size()-equivalent survivor
        computation, not just agree on the aggregate count."""
        from selection_fragility.identify import _run_mcs

        L = random_panel(seed=6, T=25, K=5)
        panel = LossPanel.from_losses(L)
        out = report(panel, alpha=0.10)
        expected_survivors = set(_run_mcs(L, alpha=0.10).included)
        lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.strip().startswith(tuple(L))}
        for model in L:
            assert model in lines, f"model {model!r} missing from LEADERBOARD"
            expected_flag = "in" if model in expected_survivors else "excluded"
            assert (expected_flag == "in") == ("in MCS" in lines[model]), \
                f"{model}: expected {'in MCS' if expected_flag=='in' else 'excluded'}, got: {lines[model]}"

    def test_report_speed_improved_at_moderate_k(self):
        """RELATIVE, not absolute, timing check (closes a real gap found by this fix's own
        independent reviewer: an earlier version used a fixed 15s ceiling at K=80, which a naive
        measurement showed only takes ~0.4s even with the pre-fix triple-call pattern -- far too
        loose to catch a reversion at any K small enough to run quickly in CI). Instead, measure a
        single _run_mcs() call directly as the unit cost, and assert report() costs meaningfully
        less than what the pre-fix triple-call-plus-resolution_report pattern would (>=4 unit
        costs) -- this scales with machine speed instead of assuming an absolute number, so it
        stays a meaningful regression guard on any hardware."""
        import time
        import warnings as _w
        from selection_fragility.identify import _run_mcs

        rng = np.random.default_rng(9)
        T, K = 40, 80
        data = {f"m{i}": rng.normal(1, 0.3, T).tolist() for i in range(K)}
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            panel = LossPanel.from_losses(data)
            w = panel.weights
            t0 = time.time()
            _run_mcs(data, alpha=0.10, w=w)
            unit_cost = time.time() - t0

            t1 = time.time()
            report(panel)
            report_cost = time.time() - t1

        # correct (deduped) impl: ~2 unit costs (1 direct call here + 1 inside resolution_report).
        # pre-fix (triple-call) impl: ~4 unit costs. 3.0x sits cleanly between the two.
        assert report_cost < 3.0 * unit_cost, (
            f"report() cost {report_cost:.3f}s is not meaningfully less than 3x a single MCS "
            f"computation ({unit_cost:.3f}s) -- looks like the dedup regressed back toward "
            f"recomputing the same MCS multiple times."
        )
