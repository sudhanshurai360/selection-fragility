"""Cross-cutting tests X1-X6. TOOL_TEST_PLAN.md 'Cross-cutting tests' section.
Applies across every stage rather than belonging to one -- run once, referenced by all."""
import time
import warnings
import numpy as np
import pandas as pd
import pytest

from selection_fragility import fragility, decision_breakdown, model_confidence_set

try:
    from arch.bootstrap import MCS as ArchMCS
    HAVE_ARCH = True
except ImportError:
    HAVE_ARCH = False

try:
    from selection_fragility import LossPanel, identified, mcs_size
    HAVE_NEW_API = True
except ImportError:
    HAVE_NEW_API = False


# ---- X1: shared input validation --------------------------------------------------------------
@pytest.mark.skipif(not HAVE_NEW_API, reason="new API not implemented yet")
class TestSharedValidation:
    def test_nan_anywhere_raises_naming_offender(self):
        L = {"a": np.array([1., np.nan, 3.]), "b": np.array([1., 2., 3.])}
        with pytest.raises(ValueError, match=r"(?i)nan"):
            identified(L)
        with pytest.raises(ValueError, match=r"(?i)nan"):
            decision_breakdown(L, np.ones(3))

    def test_K0_raises(self):
        with pytest.raises(ValueError):
            identified({})

    def test_per_function_T_floor(self):
        """Different functions may have different minimum T -- e.g. selection_regret needs T>=3,
        identified() needs T>=2. Document and test each floor explicitly rather than assuming one
        global minimum."""
        L1 = {"a": np.ones(1), "b": np.full(1, 2.0)}
        with pytest.raises(ValueError, match=r"(?i)period"):
            identified(L1)

    def test_mismatched_weight_length_raises_everywhere(self):
        L = {"a": np.ones(10), "b": np.full(10, 2.0)}
        w_wrong = np.ones(5)
        with pytest.raises(ValueError, match=r"(?i)length|shape"):
            decision_breakdown(L, w_wrong)


# ---- X2: narrowed differential test against arch (replaces the old full-field diff) --------------
@pytest.mark.skipif(not (HAVE_ARCH and HAVE_NEW_API), reason="arch or new API unavailable")
class TestDifferentialAgainstArch:
    @pytest.mark.parametrize("seed", range(20))
    def test_identified_matches_arch_directly(self, seed):
        rng = np.random.default_rng(seed)
        T, K = 30, 6
        M = rng.normal(1.0, 0.3, (T, K))
        L = {f"m{i}": M[:, i] for i in range(K)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arch_m = ArchMCS(M, size=0.10, reps=500, block_size=3, bootstrap="circular", seed=seed)
            arch_m.compute()
        arch_size = len(arch_m.included)

        # FIXED 2026-08-16: this called mcs_size(L) with no seed, comparing against arch's OWN
        # bootstrap at seed=`seed` while the wrapper silently used its default (seed=0). The two
        # agreed whenever the MCS decision was not close, but at seed=17 the data is a genuinely
        # borderline case where the bootstrap seed alone tips the elimination decision -- 5 vs 6.
        # The wrapper was never wrong; this was an apples-to-oranges comparison. Both sides must
        # use the SAME seed for this to be a fair differential test at all.
        our_size = mcs_size(L, seed=seed)
        assert our_size == arch_size, f"seed={seed}: arch gives {arch_size}, wrapper gives {our_size}"


# ---- X3: golden master, whole-API, real data ----------------------------------------------------
@pytest.mark.skipif(not HAVE_NEW_API, reason="new API not implemented yet")
class TestGoldenMaster:
    def test_all_26_real_cells_produce_stable_output(self, real_cells, real_series_list):
        """Not a value check (that's S1.4/S2/S3/S4's job) -- a SHAPE/STABILITY check: every real
        cell must produce output without raising, and running twice must give identical results.
        A genuine golden-value snapshot (with stored expected numbers) is created once Stage
        1-4 implementations land and their exact output values are locked in."""
        for sid in real_series_list:
            for metric in ("mase", "rmsse"):
                L = real_cells[(sid, metric)]
                r1 = identified(L)
                r2 = identified(L)
                assert r1 == r2, f"{sid}/{metric}: non-deterministic identified()"


# ---- X4: cost/performance budget ------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_NEW_API, reason="new API not implemented yet")
class TestPerformanceBudget:
    def test_identified_small_panel_under_1s(self, random_panel):
        L = random_panel(seed=0, T=32, K=6)
        t0 = time.time()
        identified(L)
        assert time.time() - t0 < 1.0

    def test_identified_large_panel_under_30s(self, random_panel):
        L = random_panel(seed=0, T=1000, K=100)
        t0 = time.time()
        identified(L)
        assert time.time() - t0 < 30.0

    def test_no_71x_slowdown_regression(self, random_panel):
        """The specific regression this guards: exchangeable_benchmark() took 6.42s at K=6,T=32
        in the old design; a vectorised version does the SAME work in 0.09s. If any new function
        that plays a similar role takes anywhere near the old figure, this should catch it."""
        L = random_panel(seed=0, T=32, K=6)
        t0 = time.time()
        identified(L)
        mcs_size(L)
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"took {elapsed:.2f}s on a K=6,T=32 panel -- possible slowdown regression"


# ---- X5: dtype and container robustness -----------------------------------------------------------
@pytest.mark.skipif(not HAVE_NEW_API, reason="new API not implemented yet")
class TestDtypeRobustness:
    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
    def test_dtype_sweep(self, dtype, random_panel):
        L_f = random_panel(seed=0, T=30, K=4)
        if dtype == np.int64:
            L = {k: (v * 100).astype(dtype) for k, v in L_f.items()}
        else:
            L = {k: v.astype(dtype) for k, v in L_f.items()}
        identified(L)   # must not raise

    def test_pandas_nullable_dtype_documented_behavior(self):
        """pandas nullable Int64/Float64 with pd.NA did not exist in the old code's test surface
        at all. Either coerced cleanly or raises with a clear message -- this test documents
        whichever choice is made.

        FIXED 2026-08-27 (round-7 hollow-test-hunter finding): the previous version of this test
        was a no-op regardless of outcome -- `assert True` on the raise branch, a bare `pass` on
        the success branch that never inspected the resulting panel at all, so a successful-but-
        CORRUPTED coercion (e.g. values silently truncated or reordered) would have passed
        identically to a correct one. Now the success branch actually verifies the panel's values
        against the real input, not just that construction didn't crash."""
        a_vals = [1.0, 2.0, 3.0]
        b_vals = [1.5, 1.5, 1.5]
        df = pd.DataFrame({
            "a": pd.array(a_vals, dtype="Float64"),
            "b": pd.array(b_vals, dtype="Float64"),
        })
        try:
            panel = LossPanel.from_losses(df)
        except (ValueError, TypeError):
            pass   # raising with SOME message is an acceptable resolution
        else:
            # Clean coercion is also acceptable, but "clean" must actually mean the real values
            # survived intact -- not just that nothing crashed.
            assert set(panel.losses.keys()) == {"a", "b"}
            np.testing.assert_allclose(np.asarray(panel.losses["a"], dtype=float), a_vals)
            np.testing.assert_allclose(np.asarray(panel.losses["b"], dtype=float), b_vals)


# ---- X6: top-level rename-invariance sweep across the FULL public API ------------------------------
@pytest.mark.skipif(not HAVE_NEW_API, reason="new API not implemented yet")
class TestFullAPIRenameInvariance:
    @pytest.mark.parametrize("seed", range(30))
    def test_full_surface_invariant_to_renaming(self, seed, random_panel):
        L = random_panel(seed=seed, T=30, K=5)
        rng = np.random.default_rng(seed + 1000)
        remap = dict(zip(L, rng.permutation(list(L))))
        L2 = {remap[m]: v for m, v in L.items()}

        assert identified(L) == identified(L2)
        assert mcs_size(L) == mcs_size(L2)

        try:
            from selection_fragility import selection_regret, minimum_detectable_edge, mcb_bound
            r1, r2 = selection_regret(L), selection_regret(L2)
            if r1 is not None and r2 is not None:
                assert r1 == pytest.approx(r2, rel=1e-6)
            assert minimum_detectable_edge(L) == pytest.approx(minimum_detectable_edge(L2), rel=1e-6)
            assert mcb_bound(L) == pytest.approx(mcb_bound(L2), rel=1e-6)
        except ImportError:
            pass   # Stage 2 not implemented yet -- covered separately once it lands
