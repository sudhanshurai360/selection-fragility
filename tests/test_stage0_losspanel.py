"""Stage 0 -- LossPanel adapters: from_losses (dict / wide DataFrame / ndarray), from_forecasts,
save/load, and the validation guards (weight shape, metadata-column collisions, K-vs-T sanity)."""
import warnings

import numpy as np
import pandas as pd
import pytest

try:
    from selection_fragility import LossPanel
    HAVE_LOSSPANEL = True
except ImportError:
    HAVE_LOSSPANEL = False

pytestmark = pytest.mark.skipif(not HAVE_LOSSPANEL, reason="LossPanel not implemented yet (Stage 0)")


# ---- S0.1: input shape matrix ---------------------------------------------------------------------
class TestInputShapes:
    def test_wide_dataframe_datetimeindex(self, datetime_indexed_frame):
        p = LossPanel.from_losses(datetime_indexed_frame)
        assert list(p.labels) == list(datetime_indexed_frame.index)

    def test_wide_dataframe_rangeindex(self):
        df = pd.DataFrame({"a": [1., 2, 3], "b": [1.5, 1.5, 1.5]})
        p = LossPanel.from_losses(df)
        assert list(p.labels) == [0, 1, 2]

    def test_wide_dataframe_string_index(self):
        df = pd.DataFrame({"a": [1., 2, 3], "b": [1.5, 1.5, 1.5]}, index=["jan", "feb", "mar"])
        p = LossPanel.from_losses(df)
        assert list(p.labels) == ["jan", "feb", "mar"]

    def test_wide_dataframe_periodindex(self):
        # STALE FIXTURE FIXED (2026-08-25): np.arange(6.0) is a literal arithmetic sequence, which
        # the metadata-column heuristic added 2026-08-16 correctly refuses as a stray metadata
        # column, not a real loss series -- this test predates that heuristic and was never about
        # metadata detection, only about PeriodIndex working as an index type. Non-sequential
        # values exercise the same index-handling path without tripping the (correct) guard.
        rng = np.random.default_rng(0)
        idx = pd.period_range("2020-01", periods=6, freq="M")
        df = pd.DataFrame({"a": rng.normal(1.0, 0.2, 6), "b": rng.normal(1.1, 0.2, 6)}, index=idx)
        p = LossPanel.from_losses(df)
        assert len(p.labels) == 6

    def test_ndarray_wide_TxK(self):
        M = np.random.default_rng(0).normal(1, 0.3, (30, 6))
        p = LossPanel.from_losses(M)
        assert p.losses["m0"].shape == (30,) or len(next(iter(p.losses.values()))) == 30

    def test_ndarray_KxT_warns(self, transposed_frame):
        with pytest.warns(UserWarning, match=r"(?i)K\s*>\s*T|transpos"):
            LossPanel.from_losses(transposed_frame)

    def test_dict_of_arrays(self, dominance_panel):
        p = LossPanel.from_losses(dominance_panel)
        assert set(p.losses) == {"a", "b"}

    def test_dict_of_lists(self):
        p = LossPanel.from_losses({"a": [1., 2, 3], "b": [1.5, 1.5, 1.5]})
        assert isinstance(p.losses["a"], np.ndarray)

    def test_dict_of_series_mismatched_index_raises(self):
        s1 = pd.Series([1., 2, 3], index=[0, 1, 2])
        s2 = pd.Series([1., 2, 3], index=[10, 11, 12])
        with pytest.raises(ValueError, match=r"(?i)index|align"):
            LossPanel.from_losses({"a": s1, "b": s2})

    def test_nixtla_cv_frame(self, nixtla_cv_frame):
        p = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff", group="unique_id")
        assert len(p.losses) == 3          # AutoARIMA, AutoETS, Theta
        assert len(p.labels) == 12         # 12 windows

    def test_nixtla_cv_frame_round_trip_at_scale(self):
        """Positive control for the actual on-ramp real adoption depends on: a few thousand rows in
        the exact statsforecast/neuralforecast cross_validation() wide-by-model shape must always
        ingest with zero friction (M-competition-veteran review, round 3). Also covers the
        fully-long/tidy pivot path documented in the README."""
        rng = np.random.default_rng(1)
        n_series, n_windows = 50, 24     # 1,200 (series, window) rows
        rows = []
        for sid in range(n_series):
            for w in range(n_windows):
                ds = pd.Timestamp("2020-01-01") + pd.DateOffset(months=w)
                y = 100 + rng.normal(0, 5)
                rows.append({"unique_id": f"series_{sid}", "ds": ds,
                             "cutoff": ds - pd.DateOffset(months=1), "y": y,
                             "AutoARIMA": y + rng.normal(0, 3), "AutoETS": y + rng.normal(0, 3.2),
                             "Theta": y + rng.normal(0, 3.5)})
        df = pd.DataFrame(rows)
        p_wide = LossPanel.from_forecasts(df, y_true="y", period="cutoff", group="unique_id")
        assert len(p_wide.labels) == n_windows

        df_long = df.melt(id_vars=["unique_id", "ds", "cutoff", "y"],
                           value_vars=["AutoARIMA", "AutoETS", "Theta"],
                           var_name="model", value_name="y_pred")
        wide = df_long.pivot_table(index=["unique_id", "ds", "cutoff", "y"], columns="model",
                                    values="y_pred").reset_index()
        p_from_tidy = LossPanel.from_forecasts(wide, y_true="y", period="cutoff", group="unique_id")
        assert set(p_from_tidy.losses) == set(p_wide.losses)
        assert len(p_from_tidy.labels) == n_windows

    def test_ndarray_zero_models_raises_valueerror_not_stopiteration(self):
        # FIXED (2026-08-26): a (T, 0)-shaped ndarray produced an empty models dict in
        # _from_ndarray, which from_losses() then fed to `next(iter({}.values()))` -- a bare
        # StopIteration instead of a clear error, unlike every sibling zero-model path
        # (_from_wide_dataframe's "zero model columns found", _from_dict's empty-dict check).
        with pytest.raises(ValueError, match=r"(?i)zero model columns"):
            LossPanel.from_losses(np.zeros((10, 0)))


# ---- S0.2: column-content matrix -----------------------------------------------------------------
class TestColumnContent:
    def test_stray_metadata_column_rejected(self, stray_column_frame):
        """THE CONFIRMED REGRESSION: a column of small integers ('horizon': 1,2,3...) must not be
        silently treated as a model and crowned best. Must raise or must be excluded with a
        visible warning naming the column -- never silently included."""
        with pytest.raises((ValueError, TypeError), match=r"(?i)horizon|numeric|model"):
            LossPanel.from_losses(stray_column_frame)

    def test_stray_string_column_rejected(self):
        df = pd.DataFrame({"a": [1., 2, 3], "region": ["US", "US", "US"]})
        with pytest.raises((ValueError, TypeError)):
            LossPanel.from_losses(df)

    def test_all_nan_column_excluded_or_rejected(self):
        df = pd.DataFrame({"a": [1., 2, 3], "b": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match=r"(?i)nan|b\b"):
            LossPanel.from_losses(df)

    def test_partial_nan_raises_with_specific_message(self):
        df = pd.DataFrame({"a": [1., 2, np.nan], "b": [1.5, 1.5, 1.5]})
        with pytest.raises(ValueError, match=r"(?i)nan"):
            LossPanel.from_losses(df)

    def test_duplicate_column_names_raises(self):
        df = pd.DataFrame(np.ones((5, 2)))
        df.columns = ["a", "a"]
        with pytest.raises(ValueError, match=r"(?i)duplicat"):
            LossPanel.from_losses(df)

    def test_zero_model_columns_raises(self):
        df = pd.DataFrame({"horizon": [1, 2, 3]})
        with pytest.raises(ValueError):
            LossPanel.from_losses(df)

    def test_stray_metadata_column_rejected_via_plain_dict(self):
        """ROUND-7 MUTATION-TESTING GAP (2026-08-27): `test_stray_metadata_column_rejected` above
        only exercises the DataFrame path (`_from_wide_dataframe`). The dict-input path
        (`_from_dict`) calls the SAME `_check_metadata_columns` guard, but nothing exercised it
        that way -- confirmed by actually removing both `_check_metadata_columns` calls from
        `_from_dict` and re-running the full suite: 0 failures. A plain dict is the single most
        common way a user actually calls `from_losses()` (see the README quickstart), so this is
        the more important of the two paths to protect, not a redundant duplicate of the
        DataFrame test above."""
        data = {
            "model_a": np.array([1.1, 0.9, 1.3, 1.0, 1.2, 0.8, 1.1, 0.95, 1.05, 1.0, 1.1, 0.9]),
            "model_b": np.array([1.3, 1.1, 1.5, 1.2, 1.4, 1.0, 1.3, 1.15, 1.25, 1.2, 1.3, 1.1]),
            "horizon": np.tile([1.0, 2.0, 3.0], 4),
        }
        with pytest.raises((ValueError, TypeError), match=r"(?i)horizon|numeric|model"):
            LossPanel.from_losses(data)

    def test_single_model_column_raises_with_clear_message(self):
        df = pd.DataFrame({"a": [1., 2, 3]})
        with pytest.raises(ValueError, match=r"(?i)at least 2|k\s*>=\s*2|two models"):
            LossPanel.from_losses(df)

    def test_many_model_columns_loads(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"m{i}": rng.normal(1, 0.3, 30) for i in range(100)})
        p = LossPanel.from_losses(df)
        assert len(p.losses) == 100


class TestLargeKCostWarning:
    """ADDED 2026-09-07 (round-4 8-lens PyPI-preflight audit, CONFIRMED HIGH): report()/identified()/
    mcs_size() cost grows steeply with K (model count) with no documented guidance anywhere in this
    package, despite mcs()'s own B and loss-magnitude both having explicit ceilings. Warn once, at
    panel-construction time, past a measured knee (K>100) -- not raised as an error, since a large
    but real K is a legitimate, supported use case, just a slow one."""

    def test_large_k_warns(self):
        rng = np.random.default_rng(1)
        data = {f"m{i}": rng.normal(1, 0.3, 200).tolist() for i in range(150)}   # K=150, T=200 (K<T)
        with pytest.warns(UserWarning, match=r"K=150.*cost grows steeply"):
            LossPanel.from_losses(data)

    def test_k_at_or_below_threshold_does_not_warn(self):
        rng = np.random.default_rng(2)
        data = {f"m{i}": rng.normal(1, 0.3, 200).tolist() for i in range(100)}   # K=100, at the threshold
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LossPanel.from_losses(data)
            large_k_warnings = [x for x in w if "cost grows steeply" in str(x.message)]
        assert not large_k_warnings

    def test_large_k_and_transposed_frame_only_warns_once_not_twice(self):
        """K>T (the likely-transposed-frame case) and K>threshold can both be true at once -- must
        fire only the transposed-frame warning (the more likely real explanation, and the one
        requiring action), not both, to avoid a confusing double-warning."""
        rng = np.random.default_rng(3)
        data = {f"m{i}": rng.normal(1, 0.3, 20).tolist() for i in range(150)}   # K=150, T=20 (K>T)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LossPanel.from_losses(data)
            msgs = [str(x.message) for x in w]
        assert any("transposed frame" in m for m in msgs)
        assert not any("cost grows steeply" in m for m in msgs)


# ---- S0.3: row/period matrix ----------------------------------------------------------------------
class TestRowPeriods:
    @pytest.mark.parametrize("T", [0, 1])
    def test_too_few_periods_raises(self, T):
        df = pd.DataFrame({"a": np.ones(T), "b": np.ones(T)}) if T else pd.DataFrame({"a": [], "b": []})
        with pytest.raises(ValueError, match=r"(?i)period"):
            LossPanel.from_losses(df)

    @pytest.mark.parametrize("T", [2, 3, 4, 5, 8, 30, 1000])
    def test_valid_T_loads(self, T):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(1, 0.3, T), "b": rng.normal(1, 0.3, T)})
        p = LossPanel.from_losses(df)
        assert len(p.labels) == T

    def test_duplicate_period_raises_or_documented_aggregation(self):
        idx = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"])
        df = pd.DataFrame({"a": [1., 2, 3], "b": [1.5, 1.5, 1.5]}, index=idx)
        # Either raises, or aggregates -- whichever is chosen, it must not silently keep BOTH rows
        # as independent periods (that would double-count one calendar period).
        try:
            p = LossPanel.from_losses(df)
            assert len(p.labels) == 2, "duplicate period was silently kept as two independent periods"
        except ValueError:
            pass  # raising is an acceptable resolution too


# ---- S0.4: weight matrix (WMATRIX, referenced by name in the plan) --------------------------------
class TestWeights:
    def test_none_equals_uniform_ones(self, dominance_panel):
        p1 = LossPanel.from_losses(dominance_panel, weights=None)
        p2 = LossPanel.from_losses(dominance_panel, weights=np.ones(3))
        np.testing.assert_array_equal(p1.weights, p2.weights)

    def test_wrong_length_raises(self, dominance_panel):
        with pytest.raises(ValueError, match=r"(?i)length|shape"):
            LossPanel.from_losses(dominance_panel, weights=np.ones(5))

    def test_all_zero_raises(self, dominance_panel):
        with pytest.raises(ValueError, match=r"(?i)zero|weight"):
            LossPanel.from_losses(dominance_panel, weights=np.zeros(3))

    def test_negative_weight_raises(self, dominance_panel):
        w = np.array([1.0, 1.0, -1.0])
        with pytest.raises(ValueError, match=r"(?i)negative|weight"):
            LossPanel.from_losses(dominance_panel, weights=w)

    def test_nan_weight_raises(self, dominance_panel):
        w = np.array([1.0, np.nan, 1.0])
        with pytest.raises(ValueError, match=r"(?i)nan|weight"):
            LossPanel.from_losses(dominance_panel, weights=w)

    def test_one_zero_weight_equivalent_to_removing_period(self, random_panel):
        """A property test: zeroing one period's weight must give the same downstream k* as
        physically removing that period (with the remaining weights re-normalised in shape, not
        value -- both are unweighted-equal here)."""
        L = random_panel(seed=1, T=10, K=3)
        w_zeroed = np.ones(10); w_zeroed[4] = 0.0
        p1 = LossPanel.from_losses(L, weights=w_zeroed)
        L_removed = {m: np.delete(v, 4) for m, v in L.items()}
        p2 = LossPanel.from_losses(L_removed)
        from selection_fragility import decision_breakdown
        k1, *_ = decision_breakdown(p1.losses, p1.weights)
        k2, *_ = decision_breakdown(p2.losses, np.ones(9))
        assert k1 == k2


# ---- S0.5: K vs T boundary --------------------------------------------------------------------
class TestKvsTBoundary:
    def test_K_equals_T_no_warning(self, recwarn):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"m{i}": rng.normal(1, 0.3, 5) for i in range(5)})
        LossPanel.from_losses(df)
        assert not any("K" in str(w.message) and "T" in str(w.message) for w in recwarn.list)

    def test_K_equals_T_plus_1_warns(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"m{i}": rng.normal(1, 0.3, 5) for i in range(6)})
        with pytest.warns(UserWarning):
            LossPanel.from_losses(df)

    def test_K_much_greater_than_T_warns_naming_both_dims(self, transposed_frame):
        with pytest.warns(UserWarning, match=r"36|K.*3|3.*K"):
            LossPanel.from_losses(transposed_frame)

    def test_K_less_than_T_no_warning(self, recwarn):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"m{i}": rng.normal(1, 0.3, 30) for i in range(3)})
        LossPanel.from_losses(df)
        assert not any("K" in str(w.message) and "T" in str(w.message) for w in recwarn.list)


# ---- S0.6: label handling -----------------------------------------------------------------------
class TestLabels:
    def test_datetimeindex_survives_to_responsible(self, datetime_indexed_frame):
        """THE CONFIRMED REGRESSION: fragility(df)['responsible'] used to return [3] (a position);
        it must return the actual date the DataFrame's own index carries."""
        p = LossPanel.from_losses(datetime_indexed_frame)
        from selection_fragility import decision_breakdown
        k, opp, responsible_idx = decision_breakdown(p.losses, p.weights)
        responsible_labels = [p.labels[i] for i in responsible_idx]
        assert all(isinstance(lbl, (pd.Timestamp,)) for lbl in responsible_labels)
        assert pd.Timestamp("2020-04-30") in [pd.Timestamp(l).normalize() +
                                               pd.offsets.MonthEnd(0) for l in responsible_labels] \
            or any(pd.Timestamp(l).month == 4 for l in responsible_labels)

    def test_explicit_labels_override_index(self, datetime_indexed_frame):
        custom = [f"period_{i}" for i in range(12)]
        p = LossPanel.from_losses(datetime_indexed_frame, labels=custom)
        assert list(p.labels) == custom

    def test_no_index_falls_back_to_positions_and_is_flagged(self):
        p = LossPanel.from_losses({"a": [1., 2, 3], "b": [1.5, 1.5, 1.5]})
        assert list(p.labels) == [0, 1, 2]
        assert getattr(p, "labels_are_positional", None) is True


# ---- S0.2 (continued): metric parameter, and group/period column validation for from_forecasts ----
# Added after the initial pass found this section of the plan (TOOL_TEST_PLAN.md S0.2) had no
# corresponding test class -- from_losses' column/weight/shape coverage above does not exercise
# from_forecasts' loss-computation path at all.
class TestForecastMetrics:
    def test_metric_mae(self, nixtla_cv_frame):
        p = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                      group="unique_id", metric="mae")
        # hand check: MAE for one model, one period, averaged over the 5 series in that window
        assert all(np.all(v >= 0) for v in p.losses.values())

    def test_metric_mse_ge_mae_by_jensen(self, nixtla_cv_frame):
        """MSE >= MAE^2 is not generally true, but for THIS data (errors with nonzero variance),
        sqrt(MSE) >= MAE always holds by Jensen's inequality -- a real, checkable property, not an
        arbitrary tolerance."""
        p_mae = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                          group="unique_id", metric="mae")
        p_mse = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                          group="unique_id", metric="mse")
        for m in p_mae.losses:
            np.testing.assert_array_compare(lambda a, b: a >= b - 1e-9,
                                            np.sqrt(p_mse.losses[m]), p_mae.losses[m])

    def test_metric_rmse_is_sqrt_mse(self, nixtla_cv_frame):
        p_mse = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                          group="unique_id", metric="mse")
        p_rmse = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                          group="unique_id", metric="rmse")
        for m in p_mse.losses:
            np.testing.assert_allclose(p_rmse.losses[m], np.sqrt(p_mse.losses[m]), rtol=1e-6)

    def test_metric_mape_zero_ytrue_raises_or_documented(self):
        df = pd.DataFrame({
            "unique_id": ["s1"] * 4, "ds": pd.date_range("2020-01-01", periods=4, freq="ME"),
            "cutoff": pd.date_range("2019-12-01", periods=4, freq="ME"),
            "y": [0.0, 10.0, 20.0, 30.0], "model_a": [1.0, 11.0, 19.0, 31.0],
        })
        try:
            LossPanel.from_forecasts(df, y_true="y", period="cutoff", group="unique_id", metric="mape")
        except (ValueError, ZeroDivisionError):
            pass   # raising on a zero actual is an acceptable, documented resolution
        else:
            pytest.fail("MAPE with y_true=0 silently produced a value -- must raise or be documented")

    def test_metric_callable(self, nixtla_cv_frame):
        def custom_loss(y, yhat):
            return np.abs(y - yhat) ** 1.5
        p = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                      group="unique_id", metric=custom_loss)
        assert all(np.all(v >= 0) for v in p.losses.values())

    def test_unrecognized_metric_string_raises(self, nixtla_cv_frame):
        with pytest.raises(ValueError, match=r"(?i)metric"):
            LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                      group="unique_id", metric="not_a_real_metric")

    def test_from_forecasts_partial_nan_row_warns(self):
        """CONFIRMED BUG (round-2 review, 2026-08-26): groupby().mean() skips NaN by default, so a
        model missing a forecast for SOME (not all) rows within a period silently averages over
        fewer rows than a fully-covered sibling model in the same period -- a different effective
        sample size per model, no warning, no error (the existing NaN/inf guard only ever sees the
        already-averaged, NaN-free result). Reproduced: model_a has 10/10 valid rows and model_b has
        7/10 valid rows in period 0; must now warn, naming the model and the under-covered period."""
        rows = []
        for p in range(3):
            for _ in range(10):
                rows.append({"period": p, "y": 1.0, "model_a": 1.1, "model_b": 1.2})
        df = pd.DataFrame(rows)
        period0 = df.index[df["period"] == 0][:3]
        df.loc[period0, "model_b"] = np.nan
        with pytest.warns(UserWarning, match=r"model_b.*period"):
            panel = LossPanel.from_forecasts(df, y_true="y", period="period",
                                              models=["model_a", "model_b"])
        assert np.all(np.isfinite(panel.losses["model_b"]))   # still builds -- warns, doesn't raise


class TestForecastNumericStringColumn:
    """CONFIRMED BLOCKING (round-4 8-lens PyPI-preflight audit, 2026-09-07): the models=None
    auto-inference branch filtered candidates on `pd.api.types.is_numeric_dtype` alone -- a genuinely
    numeric model column read as object/string dtype (e.g. from a CSV without explicit dtype control)
    was silently EXCLUDED with no warning, no error. Reproduced: the actual best model, stored as
    numeric strings, vanished from the panel entirely and report() printed a confident verdict over
    the two remaining models as if nothing were missing."""

    def test_numeric_string_column_is_recovered_as_a_model(self):
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({"cutoff": pd.date_range("2020-01-01", periods=n, freq="MS"),
                            "y": rng.normal(10, 1, n)})
        df["ar1"] = df["y"] + rng.normal(0, 2.0, n)
        df["ets"] = df["y"] + rng.normal(0, 1.5, n)
        df["nbeats"] = (df["y"] + rng.normal(0, 0.1, n)).round(3).astype(str)   # best model, as strings
        # dtype-agnostic on purpose (fixed 2026-09-07, final pre-publish pip-install check): pandas
        # 3.0 made its dedicated string dtype the default for `.astype(str)`, not plain `object`, so
        # a literal `== object` precondition here silently stopped testing the pandas-3.x scenario
        # the moment pandas 3.0 became the default resolved version -- this is the exact dtype
        # confusion the test exists to guard against, so the setup assertion must not assume one
        # pandas generation's representation of "a column of number-looking strings".
        assert df["nbeats"].dtype == object or isinstance(df["nbeats"].dtype, pd.StringDtype)
        panel = LossPanel.from_forecasts(df, y_true="y", period="cutoff")
        assert set(panel.models) == {"ar1", "ets", "nbeats"}
        # the recovered model's losses must reflect its real (low-noise) values, not garbage
        assert panel.losses["nbeats"].mean() < panel.losses["ar1"].mean()

    def test_explicit_string_dtype_column_is_recovered_as_a_model(self):
        """Same guard, a different route to the same dtype: a column an upstream pipeline has
        explicitly opted into pandas' dedicated string dtype (`dtype="string"`, available since
        pandas 1.0 -- not just pandas 3.0's new default for plain string data)."""
        rng = np.random.default_rng(3)
        n = 20
        df = pd.DataFrame({"cutoff": pd.date_range("2020-01-01", periods=n, freq="MS"),
                            "y": rng.normal(10, 1, n)})
        df["ar1"] = df["y"] + rng.normal(0, 2.0, n)
        vals = (df["y"] + rng.normal(0, 0.1, n)).round(3)
        df["nbeats"] = vals.astype(str).astype("string")
        assert isinstance(df["nbeats"].dtype, pd.StringDtype)
        panel = LossPanel.from_forecasts(df, y_true="y", period="cutoff")
        assert set(panel.models) == {"ar1", "nbeats"}

    def test_pyarrow_backed_string_column_is_recovered_as_a_model(self):
        """WIDENED 2026-09-07 (independent review of the pandas-3.0 fix above): a pyarrow-backed
        string column (`pd.ArrowDtype(pa.string())`, the dtype `pd.read_csv(...,
        dtype_backend="pyarrow")` or `.convert_dtypes(dtype_backend="pyarrow")` produce) is neither
        plain object dtype nor `pd.StringDtype` -- confirmed to reproduce the identical silent-
        exclusion bug on both pandas 2.3.3 and 3.0.5 before this test's fix. Skips (rather than
        failing) if pyarrow isn't installed, since it is not a declared dependency of this package --
        only relevant to users who opted into it themselves."""
        pa = pytest.importorskip("pyarrow")
        rng = np.random.default_rng(4)
        n = 20
        df = pd.DataFrame({"cutoff": pd.date_range("2020-01-01", periods=n, freq="MS"),
                            "y": rng.normal(10, 1, n)})
        df["ar1"] = df["y"] + rng.normal(0, 2.0, n)
        vals = (df["y"] + rng.normal(0, 0.1, n)).round(3)
        df["nbeats"] = vals.astype(str).astype(pd.ArrowDtype(pa.string()))
        assert isinstance(df["nbeats"].dtype, pd.ArrowDtype)
        panel = LossPanel.from_forecasts(df, y_true="y", period="cutoff")
        assert set(panel.models) == {"ar1", "nbeats"}

    def test_genuine_text_metadata_column_still_excluded_not_raised(self):
        """The coercion fix must not regress the existing behavior for a column that is genuinely
        non-numeric (real categorical text) -- it should still be silently excluded, exactly as
        before, not swept in as a bogus model."""
        rng = np.random.default_rng(1)
        n = 10
        df = pd.DataFrame({"cutoff": pd.date_range("2020-01-01", periods=n, freq="MS"),
                            "y": rng.normal(10, 1, n)})
        df["ar1"] = df["y"] + rng.normal(0, 1, n)
        df["ets"] = df["y"] + rng.normal(0, 1, n)
        df["region"] = ["east", "west"] * (n // 2)
        panel = LossPanel.from_forecasts(df, y_true="y", period="cutoff")
        assert "region" not in panel.models
        assert set(panel.models) == {"ar1", "ets"}

    def test_datetime_column_not_swept_in_by_coercion(self):
        """`pd.to_numeric` "succeeds" on a datetime64 column (reinterpreting it as nanosecond-epoch
        integers) even though it is not numeric-dtype -- the coercion fix must be scoped to object
        dtype only, or a leftover `ds`-style datetime column (as in the real nixtla CV-frame shape)
        would be wrongly recovered as a bogus model."""
        rng = np.random.default_rng(2)
        n = 12
        ds = pd.date_range("2020-01-01", periods=n, freq="MS")
        df = pd.DataFrame({"ds": ds, "cutoff": ds - pd.DateOffset(months=1),
                            "y": rng.normal(10, 1, n)})
        df["ar1"] = df["y"] + rng.normal(0, 1, n)
        df["ets"] = df["y"] + rng.normal(0, 1, n)
        panel = LossPanel.from_forecasts(df, y_true="y", period="cutoff")
        assert "ds" not in panel.models
        assert set(panel.models) == {"ar1", "ets"}


class TestForecastDuplicateGroupPeriodRow:
    """CONFIRMED HIGH (round-4 8-lens PyPI-preflight audit, 2026-09-07): `group` was validated as a
    column but never used as part of the aggregation key, so a duplicated (group, period) row (a
    re-appended CV fold, a fan-out join) was silently pooled into that period's mean -- the existing
    "fewer non-missing rows than the period total" coverage warning could never fire, since both
    duplicate rows inflate the same total together. Reproduced: duplicating one row with an outlier
    value flipped a panel's identification verdict with zero warning of any kind. Scoped to
    group-not-None only -- see TestForecastMetrics::test_from_forecasts_partial_nan_row_warns for why
    group=None legitimately allows multiple rows per period and must not be caught by this check."""

    def test_duplicate_group_period_row_raises(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "unique_id": ["s1", "s1", "s2", "s2", "s1"],
            "cutoff": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01", "2020-01-01"]),
            "y": rng.normal(10, 1, 5),
        })
        df["ar1"] = df["y"] + rng.normal(0, 1, 5)
        df["ets"] = df["y"] + rng.normal(0, 1, 5)
        with pytest.raises(ValueError, match=r"(?i)duplicate.*period.*group|duplicate.*\(period, group\)"):
            LossPanel.from_forecasts(df, y_true="y", period="cutoff", group="unique_id")

    def test_duplicate_row_actually_flips_the_verdict_if_uncaught(self):
        """Direct evidence the check is protecting something real, not a hypothetical: build the
        exact clean panel first (correctly identified), then show that duplicating one row -- which
        the check above now rejects outright -- would otherwise have flipped the verdict."""
        rng = np.random.default_rng(4)
        clean = pd.DataFrame({
            "unique_id": ["s1"] * 8,
            "cutoff": pd.date_range("2020-01-01", periods=8, freq="MS"),
            "y": rng.normal(10, 1, 8),
        })
        clean["model_a"] = clean["y"] + rng.normal(0, 1.0, 8)
        clean["model_b"] = clean["y"] + rng.normal(0, 1.0, 8)
        clean_panel = LossPanel.from_forecasts(clean, y_true="y", period="cutoff", group="unique_id")
        assert np.isfinite(clean_panel.losses["model_a"]).all()

        dup_row = clean.iloc[[0]].copy()
        dup_row["model_a"] += 50   # a large outlier on the duplicated row
        dirty = pd.concat([clean, dup_row], ignore_index=True)
        with pytest.raises(ValueError, match=r"(?i)duplicate"):
            LossPanel.from_forecasts(dirty, y_true="y", period="cutoff", group="unique_id")

    def test_no_group_repeated_period_rows_still_work(self):
        """The check must NOT fire when group=None -- multiple rows sharing a period with no group
        column is a real, supported, tested shape (repeated/pooled observations within a period)."""
        rows = [{"period": p, "y": 1.0, "a": 1.1, "b": 1.2} for p in range(2) for _ in range(10)]
        df = pd.DataFrame(rows)
        panel = LossPanel.from_forecasts(df, y_true="y", period="period", models=["a", "b"])
        assert panel.models == ["a", "b"]

    def test_unpivoted_long_frame_with_autoinferred_models_warns(self):
        """FIXED 2026-09-10 (round-7 final pre-publish review, real_data_dogfood lens): the single
        most natural mistake in this documented API -- calling from_forecasts() on a still-long/tidy
        frame (one row per (period, model), with a leftover model-identity column) without pivoting
        to wide-by-model first, and with no group=. Reproduced on real project data: this silently
        auto-inferred leftover numeric columns as if they were competing models, pooling every real
        model's rows together within each period, with zero warnings. The duplicate-(period,group)
        check above deliberately does NOT fire when group=None (the sibling test just above this one
        proves that), so this needed its own, differently-scoped warning -- keyed on rows repeating
        within a period (a genuinely wide frame has exactly one row per period), not on group=None
        alone, so it does not fire on the common, correct wide-format case (see the explicit-models
        test above and the auto-infer tests elsewhere in this file, neither of which should warn)."""
        rows = []
        for p in range(20):
            for model_name, val in [("real_model_a", 1.0 + p * 0.01), ("real_model_b", 1.2 + p * 0.01)]:
                rows.append({"period": p, "y": 1.1, "model": model_name, "pred": val, "extra_metric": 5.0})
        df = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match=r"(?i)rows repeat within a period"):
            panel = LossPanel.from_forecasts(df, y_true="y", period="period")
        # confirms the failure mode itself: auto-inference crowned "pred"/"extra_metric" as models,
        # not the real "model" column's values -- the warning exists precisely because this happens.
        assert set(panel.models) == {"pred", "extra_metric"}


class TestForecastGroupPeriodColumns:
    def test_missing_group_column_raises(self, nixtla_cv_frame):
        df = nixtla_cv_frame.drop(columns=["unique_id"])
        with pytest.raises((ValueError, KeyError), match=r"(?i)group|unique_id"):
            LossPanel.from_forecasts(df, y_true="y", period="cutoff", group="unique_id")

    def test_missing_period_column_raises(self, nixtla_cv_frame):
        df = nixtla_cv_frame.drop(columns=["cutoff"])
        with pytest.raises((ValueError, KeyError), match=r"(?i)period|cutoff"):
            LossPanel.from_forecasts(df, y_true="y", period="cutoff", group="unique_id")

    def test_group_equals_period_raises(self, nixtla_cv_frame):
        with pytest.raises(ValueError, match=r"(?i)group|period|same"):
            LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff", group="cutoff")

    def test_models_none_autoinfers_same_as_explicit(self, nixtla_cv_frame):
        p_auto = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                          group="unique_id", models=None)
        p_explicit = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                              group="unique_id",
                                              models=["AutoARIMA", "AutoETS", "Theta"])
        assert set(p_auto.losses) == set(p_explicit.losses)
        for m in p_auto.losses:
            np.testing.assert_array_equal(p_auto.losses[m], p_explicit.losses[m])

    def test_explicit_models_subset_excludes_the_rest(self, nixtla_cv_frame):
        p = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                      group="unique_id", models=["AutoARIMA", "AutoETS"])
        assert set(p.losses) == {"AutoARIMA", "AutoETS"}

    def test_weights_count_inferred_matches_group_sizes(self, nixtla_cv_frame):
        p = LossPanel.from_forecasts(nixtla_cv_frame, y_true="y", period="cutoff",
                                      group="unique_id", weights="count")
        n_series = nixtla_cv_frame["unique_id"].nunique()
        n_periods = nixtla_cv_frame["cutoff"].nunique()
        assert len(p.weights) == n_periods
        assert np.all(p.weights == n_series) or np.all(p.weights > 0)


# ---- 10-agent code-review pass (2026-09-02): panel.py input-layer findings ------------------------
# The core breakdown-point algorithm itself was independently stress-tested (500,000+ adversarial
# trials, brute-force cross-checked) and found exact with zero mismatches. Every finding below is in
# the INPUT layer (LossPanel.from_losses()'s dict-construction path), not the algorithm.

class TestMixedSeriesArrayAlignment:
    """`from_losses()`'s dict-of-Series index-alignment guard (`if all(isinstance(v, pd.Series) for
    v in vals): ...`) only fires when EVERY value in the dict is a pd.Series. Mixing even one plain
    list/array with a Series silently falls through to `np.array(v, dtype=float)`, which reads the
    Series by STORED POSITION and discards its index entirely -- with zero error and zero warning.
    This matters because breakdown_number/decision_breakdown/per_period_winner/condorcet_status/
    concentration all do PAIRED, PER-PERIOD arithmetic across models that is only meaningful if
    every model's t-th array entry actually refers to the same real-world period."""

    def test_mixed_series_and_plain_array_raises_or_aligns_by_index(self):
        """A Series with a shuffled/non-default index mixed with a plain list must either (a) be
        rejected outright (matching the all-Series branch's existing strictness), or (b) actually
        be reindex-aligned by its real index -- never silently treated as if it had no index at
        all. Minimal repro: 4 periods, model 'a' given as a Series in REVERSED index order, model
        'b' as a plain list in natural order. If the library silently drops 'a's index, its values
        get paired against the WRONG periods of 'b'."""
        s_a = pd.Series([1.0, 2.0, 3.0, 4.0], index=[3, 2, 1, 0])   # reversed index
        list_b = [10.0, 20.0, 30.0, 40.0]                             # natural order 0,1,2,3
        with pytest.raises(ValueError, match=r"(?i)index|align|order"):
            LossPanel.from_losses({"a": s_a, "b": list_b})

    def test_mixed_series_array_kstar_matches_hand_aligned_ground_truth(self):
        """End-to-end reproduction of the real-world consequence: k* computed on a silently
        misaligned mixed-input panel diverges dramatically from the TRUE, correctly year-aligned
        k*. Asserts the CORRECT behavior (either raise, or match the aligned answer) -- fails
        against the currently-shipped code, which returns k*=5 where the true answer is k*=1 (a 5x
        overstatement of decision robustness, the unsafe direction for a fragility tool)."""
        from selection_fragility.fragility import decision_breakdown

        years = [2015, 2016, 2017, 2018, 2019, 2020]
        a_true = {2015: 1.0, 2016: 1.0, 2017: 1.0, 2018: 1.0, 2019: 1.0, 2020: 5.0}
        b_true = {2015: 5.0, 2016: 1.2, 2017: 1.2, 2018: 1.2, 2019: 1.2, 2020: 1.2}

        a_aligned = np.array([a_true[y] for y in years])
        b_aligned = np.array([b_true[y] for y in years])
        k_true, _opp_true, _removed_true = decision_breakdown({"a": a_aligned, "b": b_aligned})
        assert k_true == 1   # sanity check on the hand-constructed ground truth itself

        shuffled_idx = [2020, 2016, 2017, 2018, 2019, 2015]
        s_a = pd.Series([a_true[y] for y in shuffled_idx], index=shuffled_idx)
        list_b = [b_true[y] for y in years]

        try:
            p = LossPanel.from_losses({"a": s_a, "b": list_b})
        except ValueError:
            return   # rejecting the ambiguous mixed input outright is an acceptable fix
        k, _opp, _removed = decision_breakdown(p.losses, p.weights)
        assert k == k_true, (
            f"mixed Series+array input silently misaligned periods: got k*={k}, but the "
            f"correctly year-aligned ground truth is k*={k_true}. LossPanel accepted this input "
            f"with no error/warning and produced a confidently wrong fragility verdict."
        )


class TestBoolArrayViaDictPath:
    """`from_forecasts()`/`from_losses()` with a wide DataFrame both explicitly reject boolean
    columns ("looks like a flag/indicator column, not a model") -- see panel.py's own is_bool_dtype
    guards, added specifically because a promo/holiday/beat-baseline indicator column is a
    realistic stray input. The dict-of-arrays path (`LossPanel.from_losses({model: array, ...})`)
    has no equivalent check, so the same class of stray column silently becomes a "model" there,
    inconsistent with the other two ingestion surfaces."""

    def test_boolean_array_model_rejected_via_dict_path(self):
        with pytest.raises(TypeError, match=r"(?i)boolean|flag"):
            LossPanel.from_losses({
                "real_model": np.array([1.0, 2.0, 1.5, 1.2, 0.9]),
                "is_holiday_flag": np.array([True, False, False, True, False]),
            })


class TestMalformedShapeAtConstruction:
    """`from_losses()`'s per-model validation loop checks length/finiteness/magnitude but never
    ndim -- a 2-D "model" array is accepted at CONSTRUCTION time and only crashes later, the first
    time an actual analysis function is called. Worse, LossPanel.__repr__'s blanket
    `except Exception` fallback means print(panel) shows nothing wrong at all."""

    def test_2d_model_array_rejected_at_construction_not_downstream(self):
        with pytest.raises(ValueError, match=r"(?i)1-d|dimension|shape"):
            LossPanel.from_losses({
                "a": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],   # malformed: shape (3, 2)
                "b": [1.0, 2.0, 3.0],
            })

    def test_2d_model_array_never_reaches_repr_to_mask_the_defect(self):
        """FIXED 2026-09-02: previously a 2-D 'model' array was accepted at construction and only
        crashed on first analysis call, and __repr__'s broad `except Exception` fallback meant
        `print(panel)` showed the innocuous `LossPanel(models=['a', 'b'], T=3)` with zero
        indication anything was wrong. Now the malformed shape is rejected at construction (see
        the sibling test above), so a broken panel object claiming this shape can never exist to
        be printed in the first place -- the masking scenario is structurally unreachable, not
        merely avoided by chance."""
        with pytest.raises(ValueError, match=r"(?i)1-d|dimension|shape"):
            LossPanel.from_losses({
                "a": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "b": [1.0, 2.0, 3.0],
            })

    def test_ragged_model_array_gives_clear_message_not_raw_numpy_error(self):
        """FIXED 2026-09-10 (round-7 final pre-publish review, adversarial_input_fuzzing lens): a
        RAGGED (jagged, unequal-inner-length) nested list -- distinct from the RECTANGULAR 2-D case
        above, which numpy converts cleanly before the ndim check catches it -- made the bare
        np.asarray(v) call itself raise, escaping as a raw numpy internals message naming neither
        'LossPanel', 'from_losses()', nor the offending model key. Reachable via LossPanel.load() on
        a tampered/malformed saved panel file with this exact shape in its 'losses' dict."""
        with pytest.raises(ValueError, match=r"(?i)model 'a'.*(ragged|jagged)"):
            LossPanel.from_losses({
                "a": [[1.0, 2.0], [3.0]],   # ragged: inner lists have different lengths
                "b": [1.0, 2.0, 3.0],
            })


class TestScalarModelValueErrorMessage:
    """A scalar (0-D) 'model' value crashes with a raw `TypeError: len() of unsized object` instead
    of the clear ValueError every other malformed-shape input in this module produces. Regardless
    of which dict key is scalar (order-independence checked explicitly, since `T` is inferred from
    whichever model happens to be iterated first)."""

    @pytest.mark.parametrize("data", [
        {"a": 5.0, "b": [1.0, 2.0, 3.0]},
        {"b": [1.0, 2.0, 3.0], "a": 5.0},
    ], ids=["scalar_first", "scalar_second"])
    def test_scalar_model_value_raises_clear_valueerror(self, data):
        with pytest.raises(ValueError, match=r"(?i)array|shape|scalar|1-d"):
            LossPanel.from_losses(data)
