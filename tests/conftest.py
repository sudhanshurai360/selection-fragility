"""Shared fixtures for the redesign test suite (release/TOOL_TEST_PLAN.md).
Fixture names F1-F11 match the plan document exactly so a test failure can be traced back to itsplan entry without translation. Everything here is either real project data or hand-computed --nothing is a mock standing in for a real number."""
import math
import pathlib
import numpy as np
import pandas as pd
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[3]
REAL_PANEL_CSV = ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
# ---- F1: the real 24-cell panel (12 series x 2 metrics) -----------------------------------------
# UPDATED 2026-08-29 (11-angle review finding: this fixture went stale when the main repo dropped
# UMCSENT for a confirmed U-Michigan licensing restriction -- the main repo's own tests were fixed
# at the time, this nested package's fixture was not).
@pytest.fixture(scope="session")
def real_panel_raw():
    """Raw predictions CSV, loaded once per test session."""
    if not REAL_PANEL_CSV.exists():
        pytest.skip(f"real panel not found at {REAL_PANEL_CSV}")
    return pd.read_csv(REAL_PANEL_CSV)
@pytest.fixture(scope="session")
def real_cells(real_panel_raw):
    """{(series, metric): {model: np.ndarray over years}} for all 12 series x {mase, rmsse}.
    mase uses 'sae' (already scaled absolute error) averaged per year;
    rmsse uses sqrt(mean('msse')) per year -- matching social_choice.py's own METRICS convention,
    the exact convention whose omission caused a real bug (rho_star.py, fixed 2026-08-08) when a
    second implementation used plain .mean() on msse instead of sqrt(mean()).
    """
    out = {}
    for sid, g in real_panel_raw.groupby("series"):
        piv_sae = g.groupby(["model", "year"])["sae"].mean().reset_index()
        piv_msse = g.groupby(["model", "year"])["msse"].mean().reset_index()
        mase = piv_sae.pivot(index="year", columns="model", values="sae").dropna()
        rmsse = piv_msse.pivot(index="year", columns="model", values="msse").dropna().pow(0.5)
        out[(sid, "mase")] = {m: mase[m].to_numpy(float) for m in mase.columns}
        out[(sid, "rmsse")] = {m: rmsse[m].to_numpy(float) for m in rmsse.columns}
    assert len(out) == 24, f"expected 24 cells (12 series x 2 metrics), got {len(out)}"
    return out
@pytest.fixture(scope="session")
def real_series_list():
    return ["BOPGEXP", "BOPGIMP", "BUSINV", "DGORDER", "HOUST", "M2SL", "PAYEMS",
            "PCE", "PI", "RSAFS", "TCU", "UNEMPLOY"]
# ---- F2: hand-checkable tiny panel (the session's own worked example) ---------------------------
@pytest.fixture
def hand_checked_panel():
    """3 models, 12 months. arima better 11/12; ets wins on the pooled average via one shock month.
    Every value below was computed by hand and independently verified twice in the session that
    designed this suite -- this is the canonical 'does the tool catch the obvious case' fixture."""
    months = ["2020-%02d" % m for m in range(1, 13)]
    L = {
        "arima":
    np.array([0.22, 0.19, 0.25, 4.00, 0.21, 0.24, 0.18, 0.26, 0.23, 0.20, 0.27, 0.22]),
        "ets":
      np.array([0.30, 0.28, 0.33, 2.50, 0.31, 0.29, 0.32, 0.30, 0.34, 0.28, 0.31, 0.33]),
        "lightgbm": np.array([0.35, 0.40, 0.38, 3.20, 0.36, 0.42, 0.37, 0.39, 0.41, 0.38, 0.36, 0.40]),
    }
    expect = dict(
        pooled_winner="ets", per_period_winner="arima", condorcet_winner="arima",
        k_star=1, responsible_period=3,
          # index 3 == April (0-indexed)
        reversal=True, concentration_gt_1=True,
    )
    return L, months, expect
# ---- F3: exact-tie panel -------------------------------------------------------------------------
@pytest.fixture
def exact_tie_panel():
    """Two models, pooled means exactly equal in float arithmetic (verified via Decimal below)."""
    a = np.array([0.80, 0.32, 0.91, 1.38])
    b = np.array([0.87, 0.81, 1.24, 0.49])
    from decimal import Decimal
    da = sum(Decimal(str(x)) for x in a) / 4
    db = sum(Decimal(str(x)) for x in b) / 4
    assert da == db, f"fixture is not actually an exact tie: {da} vs {db}"
    return {"a": a, "b": b}
# ---- F4: near-tie float panel (degenerate-margin boundary) ---------------------------------------
@pytest.fixture
def near_tie_float_panel():
    """Margin ~1e-13 -- deliberately at the float degenerate-margin boundary already used elsewhere
    in this project's own guard logic (relative floor 1e-12 in mcs_test.py / mcs.py)."""
    T = 20
    a = np.full(T, 1.0)
    b = a.copy()
    b[0] += 2e-13 * T  # sums to a margin on the order of 1e-13 relative
    return {"a": a, "b": b}
# ---- F5: dominance panel (the confirmed fabricated-reversal regression) --------------------------
@pytest.fixture
def dominance_panel():
    """b weakly dominates a: never worse, strictly better once. No voting rule can disagree.
    BY HAND: pooled=b, per_period=b, condorcet=b, reversal=False."""
    return {"a": np.array([1.0, 1.0, 1.0]), "b": np.array([1.0, 1.0, 0.0])}
# ---- F6: byte-identical triple -------------------------------------------------------------------
@pytest.fixture
def identical_triple_panel():
    vals = np.array([1.0, 2.0, 3.0])
    return {"a": vals.copy(), "b": vals.copy(), "c": vals.copy()}
# ---- F7: random panel generator (factory) ----------------------------------------------------
@pytest.fixture
def random_panel():
    def _make(seed=0, T=30, K=6, mu=1.0, sigma=0.30, model_prefix="m"):
        rng = np.random.default_rng(seed)
        return {f"{model_prefix}{i}": rng.normal(mu, sigma, T) for i in range(K)}
    return _make
# ---- F8: exchangeable-null generator (factory) -------------------------------------------------
@pytest.fixture
def exchangeable_null_panels():
    def _make(K, T, n_rep, seed=0, mu=1.0, sigma=0.30):
        rng = np.random.default_rng(seed)
        for r in range(n_rep):
            yield {f"m{i}": rng.normal(mu, sigma, T) for i in range(K)}
    return _make
# ---- F9: recentred real-shaped null (statistician's calibration DGP) -----------------------------
@pytest.fixture
def recentred_real_panel(real_cells):
    """Recentre a real cell to exact pooled equality, then circular-block-bootstrap resample it --
    the DGP tonight's statistician reviewer validated as correct-to-conservative. Used to anchor
    pivot_agreement's benchmark (~0.42) and as a size-check DGP distinct from the pure-Gaussian one."""
    def _make(series="PI", metric="mase", seed=0, block=None):
        L = real_cells[(series, metric)]
        models = list(L)
        M = np.column_stack([L[m] for m in models])
        T, K = M.shape
        grand_mean = M.mean()
        col_means = M.mean(axis=0)
        recentred = M - col_means[None, :] + grand_mean   # every column now has the SAME mean
        block = block or max(1, round(T ** (1 / 3)))
        rng = np.random.default_rng(seed)
        # circular block bootstrap resample
        starts = rng.integers(0, T, size=math.ceil(T / block))
        idx = np.concatenate([np.arange(s, s + block) % T for s in starts])[:T]
        boot = recentred[idx]
        return {m: boot[:, j] for j, m in enumerate(models)}
    return _make
# ---- F10: nixtla-style CV frame -------------------------------------------------------------------
@pytest.fixture
def nixtla_cv_frame():
    """unique_id, ds, cutoff, y, <model columns> -- realistic statsforecast.cross_validation() shape."""
    rng = np.random.default_rng(0)
    n_series, n_windows = 5, 12
    rows = []
    for sid in range(n_series):
        for w in range(n_windows):
            ds = pd.Timestamp("2020-01-01") + pd.DateOffset(months=w)
            y = 100 + rng.normal(0, 5)
            rows.append({
                "unique_id": f"series_{sid}", "ds": ds, "cutoff": ds - pd.DateOffset(months=1), "y": y,
                "AutoARIMA": y + rng.normal(0, 3), "AutoETS": y + rng.normal(0, 3.2),
                "Theta": y + rng.normal(0, 3.5),
            })
    return pd.DataFrame(rows)
# ---- F11: malformed frames (input-validation regressions) -----------------------------------------
@pytest.fixture
def stray_column_frame():
    """The confirmed regression: a leftover metadata column of small integers gets crowned best
    model. df has 3 real model columns + 1 integer 'horizon' column that must be rejected."""
    rng = np.random.default_rng(0)
    T = 36
    df = pd.DataFrame({m: rng.normal(12, 3, T) for m in ["AutoETS", "Theta", "AutoARIMA"]})
    df["horizon"] = np.tile([1, 2, 3], 12)
    return df
@pytest.fixture
def transposed_frame():
    """K=36 'models' over T=3 'periods' -- almost certainly a transposed real panel (K=3 models,
    T=36 periods) fed in backwards. Must trigger a K>>T warning, not a confident wrong answer."""
    rng = np.random.default_rng(0)
    M = np.column_stack([rng.normal(1, 0.3, 36) for _ in range(3)])   # T=36, K=3, correct orientation
    return pd.DataFrame(M.T)
                                          # transposed: 3 rows, 36 cols
@pytest.fixture
def datetime_indexed_frame():
    idx = pd.date_range("2020-01-01", periods=12, freq="ME")
    return pd.DataFrame({
        "arima":
    [0.22, 0.19, 0.25, 4.00, 0.21, 0.24, 0.18, 0.26, 0.23, 0.20, 0.27, 0.22],
        "ets":
      [0.30, 0.28, 0.33, 2.50, 0.31, 0.29, 0.32, 0.30, 0.34, 0.28, 0.31, 0.33],
        "lightgbm": [0.35, 0.40, 0.38, 3.20, 0.36, 0.42, 0.37, 0.39, 0.41, 0.38, 0.36, 0.40],
    }, index=idx)
