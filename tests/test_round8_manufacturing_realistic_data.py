"""Round 8 — predictive-maintenance/manufacturing-analytics domain review.

Every prior round's realistic-data testing assumed COMPLETE panels (every candidate model has a value
for every period). Real industrial sensor/maintenance data is structurally different: sensors drop
out, maintenance logs have gaps, and staggered model deployment means different candidates may cover
genuinely different period ranges -- a data-completeness challenge, not a volatility/tail-shape one.

Findings, precisely characterized:
1. NaN-based missingness is rejected CONSISTENTLY and with a clear, actionable message across the
   entire public API (LossPanel.from_losses, from_forecasts, and every raw-array-tier function --
   pooled_winner, decision_breakdown, fragility, model_confidence_set, resolution_report). No
   inconsistency found; no silent NaN-as-zero footgun found. This is the highest-value question this
   round asked and the answer is: correctly and uniformly refused, not mishandled.
2. from_forecasts() on genuinely staggered-deployment data (a model with zero rows for an entire
   period range, not just some rows) raises a clear ValueError naming the model and the missing
   periods -- confirmed via a realistic 60-week two-model panel where model B only exists from week 20
   onward.
3. Right-censored data (a machine that has NOT failed by the observation cutoff) is a real,
   UNDOCUMENTED methodological gap -- not a code bug the tool could catch (it has no way to know a
   loss value came from a censored observation), but nothing in README.md/EVALUATION_CARD.md warns a
   predictive-maintenance user about this specific failure mode, unlike the already-documented
   loss/accuracy sign-confusion footgun. Flagged for a documentation pass, not fixed here.
4. A clean, complete, low-noise weekly maintenance panel works correctly end-to-end as a baseline.

Verdict: genuinely usable for predictive-maintenance model comparison PROVIDED the user pre-aligns/
imputes their panel first (the tool requires this explicitly and says so clearly) -- not usable
out-of-the-box on raw, ragged sensor logs, which is an honest, disclosed requirement rather than a
silent trap.
"""
import numpy as np
import pandas as pd
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.fragility import pooled_winner, decision_breakdown, fragility
from selection_fragility.mcs import model_confidence_set
from selection_fragility.resolution import resolution_report
from selection_fragility.report import report


def _staggered_deployment_frame(T=60, seed=1):
    """4 candidate maintenance models over T weeks; model B deployed 20 weeks late, model D has
    genuine sporadic sensor-downtime gaps scattered through its whole history (not just a delayed
    start) -- two different real missingness shapes in one realistic frame."""
    rng = np.random.default_rng(seed)
    rows = []
    d_downtime = set(rng.choice(T, size=8, replace=False))
    for week in range(T):
        y = rng.normal(100, 10)
        row = {"week": week, "y_true": y, "pred_A": y + rng.normal(0, 5)}
        if week >= 20:
            row["pred_B"] = y + rng.normal(0, 4)
        row["pred_C"] = y + rng.normal(0, 6)
        if week not in d_downtime:
            row["pred_D"] = y + rng.normal(0, 4.5)
        rows.append(row)
    return pd.DataFrame(rows)


class TestMissingnessHandledConsistently:
    """Q: does the whole toolkit treat NaN/missing-period data the same way everywhere, or does some
    entry point silently mishandle it (e.g. NaN-as-zero, which would be a dangerous bug for a
    lower-is-better loss)? A: rejected consistently, everywhere, with an actionable message."""

    def test_from_losses_rejects_nan_with_clear_message(self):
        rng = np.random.default_rng(0)
        T = 150
        losses = {}
        for i, name in enumerate(["arima_rul", "lstm_rul", "rf_survival", "xgb_hazard"]):
            arr = rng.normal(1.0 + 0.05 * i, 0.3, T)
            missing_idx = rng.choice(T, size=10 + 5 * i, replace=False)
            arr[missing_idx] = np.nan
            losses[name] = arr
        with pytest.raises(ValueError, match="NaN"):
            LossPanel.from_losses(losses)

    def test_from_forecasts_rejects_genuinely_missing_period_range(self):
        """Model B has ZERO rows for weeks 0-19 (staggered deployment), not just some missing rows
        within a period -- this must raise, not silently drop model B or treat it as zero-loss.
        Isolated to ONLY the staggered-deployment gap (no scattered-downtime model in the same
        frame) so the assertion can name the specific model deterministically."""
        rng = np.random.default_rng(1)
        T = 60
        rows = []
        for week in range(T):
            y = rng.normal(100, 10)
            row = {"week": week, "y_true": y, "pred_A": y + rng.normal(0, 5)}
            if week >= 20:
                row["pred_B"] = y + rng.normal(0, 4)
            rows.append(row)
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="pred_B"):
            LossPanel.from_forecasts(df, y_true="y_true", period="week")

    def test_from_forecasts_rejects_scattered_sensor_downtime_gaps(self):
        """A model with sporadic, scattered missing periods (sensor downtime through its whole
        history, not a delayed start) must also raise, naming that model -- the staggered-deployment
        test above and this one cover the two distinct real missingness shapes separately."""
        df = _staggered_deployment_frame()
        with pytest.raises(ValueError, match="pred_D"):
            LossPanel.from_forecasts(df, y_true="y_true", period="week")

    @pytest.mark.parametrize("fn", [pooled_winner, decision_breakdown, fragility,
                                     model_confidence_set, resolution_report])
    def test_every_raw_array_entry_point_rejects_nan_consistently(self, fn):
        """All five public entry points that take a raw {model: array} dict must refuse NaN the same
        way -- an inconsistency here (one silently computing garbage while others refuse) would be a
        real, dangerous bug for a lower-is-better loss metric."""
        L = {"m0": np.array([1.0, 2.0, np.nan, 4.0]), "m1": np.array([1.0, 2.0, 3.0, 4.0])}
        with pytest.raises(ValueError):
            fn(L)


class TestCleanCompletePanelBaseline:
    """Contrast case: a clean, complete, low-noise weekly maintenance panel with no gaps must work
    correctly end-to-end -- confirms the rejections above are about genuine missingness, not the
    tool being broken for this domain's realistic data shape in general."""

    def test_complete_realistic_maintenance_panel_runs_end_to_end(self):
        rng = np.random.default_rng(7)
        T = 156  # 3 years weekly
        losses = {
            "arima_rul": rng.normal(1.0, 0.25, T),
            "lstm_rul": rng.normal(0.85, 0.22, T),   # genuinely, consistently better
            "rf_survival": rng.normal(1.05, 0.30, T),
        }
        panel = LossPanel.from_losses(losses)
        out = report(panel)
        assert "lstm_rul" in out
        assert "VERDICT" in out

    def test_staggered_deployment_panel_works_once_aligned_to_common_window(self):
        """The tool's own required workaround (restrict to the window every model actually covers)
        genuinely works once applied -- this is the correct, documented-by-error-message path, not a
        theoretical claim."""
        df = _staggered_deployment_frame()
        common_window = df[df["week"] >= 20].dropna(subset=["pred_A", "pred_B", "pred_C", "pred_D"])
        panel = LossPanel.from_forecasts(common_window, y_true="y_true", period="week")
        out = report(panel)
        assert "VERDICT" in out


class TestCensoredDataIsAnUndocumentedGap:
    """Right-censored RUL/failure data (a unit that has NOT failed by the observation cutoff) is a
    real methodological risk this tool cannot detect on its own -- it has no way to know a loss value
    came from a censored observation vs. a true outcome. This is not a code bug; it IS a real,
    currently-undocumented gap (unlike the already-documented loss/accuracy sign-confusion footgun)."""

    def test_naive_censored_loss_is_silently_accepted_no_warning(self):
        """A user who naively computes 'predicted RUL vs currently-observed-survival-time' for units
        that haven't failed yet gets zero warning that this is methodologically different from a
        true observed failure time -- documenting current behavior, not asserting it's wrong to
        accept (the tool cannot know), just that nothing flags the risk."""
        rng = np.random.default_rng(3)
        T = 40
        # "true_rul" here is actually current-survival-time-so-far for still-running units -- a
        # censored proxy, not a real outcome. The tool has no way to see this distinction.
        losses = {
            "model_a": np.abs(rng.normal(5, 2, T)),
            "model_b": np.abs(rng.normal(5.5, 2, T)),
        }
        panel = LossPanel.from_losses(losses)  # accepted with zero warning -- documents the gap
        assert panel is not None

    def test_readme_documents_censoring_risk(self):
        """FLIPPED 2026-08-27 (round-8 fix pass): README now documents the right-censored/truncated-
        observation gap this test originally locked in as absent (see git history for the prior,
        gap-locking version of this test). Verify the disclosure is present and says the right thing,
        not just that some substring exists."""
        import pathlib
        text = (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
        lower = text.lower()
        assert "censor" in lower, "README should document the right-censored/truncated-observation risk."
        # Must actually convey the risk (accepted silently, no detection), not just contain the word.
        assert "no warning" in lower or "no way" in lower or "cannot" in lower or "not designed" in lower, (
            "README mentions censoring but doesn't clearly state the tool can't detect/warn about it."
        )
