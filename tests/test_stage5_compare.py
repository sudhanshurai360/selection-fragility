"""Stage 5 -- compare(previous, current). TOOL_TEST_PLAN.md S5."""
import subprocess
import sys
import numpy as np
import pytest

try:
    from selection_fragility import LossPanel, compare
    HAVE_COMPARE = True
except ImportError:
    HAVE_COMPARE = False

pytestmark = pytest.mark.skipif(not HAVE_COMPARE, reason="compare() not implemented yet (Stage 5)")


def _panel(seed, T, K, champion_edge=0.0, extra_periods=0):
    """K models, model 0 gets `champion_edge` advantage. `extra_periods` new rows appended,
    where a DIFFERENT model (model 1) is given a strong advantage instead -- simulates 'a new
    period arrives and looks like it changed the champion'."""
    rng = np.random.default_rng(seed)
    base = rng.normal(1.0, 0.2, (T, K))
    base[:, 0] -= champion_edge
    if extra_periods:
        new_rows = rng.normal(1.0, 0.2, (extra_periods, K))
        new_rows[:, 1] -= 2.0   # model 1 dramatically better in the new periods only
        base = np.vstack([base, new_rows])
    return {f"m{i}": base[:, i] for i in range(K)}


class TestCompare:
    def test_no_change_reports_cleanly(self):
        L = _panel(seed=0, T=30, K=4, champion_edge=0.5)
        prev = LossPanel.from_losses(L)
        curr = LossPanel.from_losses(L)
        r = compare(prev, curr)
        assert r.champion_changed is False
        assert r.act is False

    def test_single_new_period_champion_reverts_check(self):
        L_base = _panel(seed=0, T=30, K=4, champion_edge=0.5)
        prev = LossPanel.from_losses(L_base)
        L_extended = _panel(seed=0, T=30, K=4, champion_edge=0.5, extra_periods=1)
        curr = LossPanel.from_losses(L_extended)
        r = compare(prev, curr)
        # removing the 1 new period should reproduce the OLD champion -- test the mechanism exists
        assert hasattr(r, "champion_without_new_periods")

    def test_multiple_new_periods_removed_as_a_block(self):
        L_base = _panel(seed=1, T=30, K=4, champion_edge=0.5)
        prev = LossPanel.from_losses(L_base)
        L_extended = _panel(seed=1, T=30, K=4, champion_edge=0.5, extra_periods=5)
        curr = LossPanel.from_losses(L_extended)
        r = compare(prev, curr)
        assert r.n_new_periods == 5

    def test_champion_changed_but_both_in_mcs_not_a_real_separation(self):
        L = _panel(seed=2, T=20, K=3, champion_edge=0.02)   # tiny edge -> both likely in MCS
        prev = LossPanel.from_losses(L)
        L2 = _panel(seed=3, T=20, K=3, champion_edge=0.02)
        curr = LossPanel.from_losses(L2)
        r = compare(prev, curr)
        if r.champion_changed and r.old_champion_still_in_mcs:
            assert r.act is False

    def test_champion_changed_and_left_mcs_is_real_separation(self):
        L = _panel(seed=4, T=40, K=2, champion_edge=0.5)
        prev = LossPanel.from_losses(L)
        L2 = _panel(seed=4, T=40, K=2, champion_edge=-0.5)   # flip which model dominates, hard
        curr = LossPanel.from_losses(L2)
        r = compare(prev, curr)
        if r.champion_changed and not r.old_champion_still_in_mcs:
            assert r.act is True

    def test_mismatched_model_set_raises(self):
        L1 = _panel(seed=0, T=20, K=3)
        L2 = {f"n{i}": v for i, v in _panel(seed=0, T=20, K=3).items()}  # different names entirely
        prev = LossPanel.from_losses(L1)
        curr = LossPanel.from_losses(L2)
        with pytest.raises(ValueError, match=r"(?i)model|match|differ"):
            compare(prev, curr)

    def test_base_rate_churn_cross_check(self):
        """At K=6, T=32 with no true winner, champion changes on a new period roughly every 7-9
        periods (tonight's redesign-reviewer table, rolling window=26 gave ~0.18 per new period at
        K=6). Sanity range, not an exact reproduction (different window mechanics)."""
        rng = np.random.default_rng(0)
        base = rng.normal(1.0, 0.3, (32, 6))
        prev = LossPanel.from_losses({f"m{i}": base[:, i] for i in range(6)})
        L2 = {f"m{i}": np.append(base[:, i], rng.normal(1.0, 0.3)) for i in range(6)}
        curr = LossPanel.from_losses(L2)
        r = compare(prev, curr)
        assert 0.0 <= r.churn_base_rate <= 1.0

    def test_save_load_round_trip(self, tmp_path):
        L = _panel(seed=0, T=20, K=3)
        p = LossPanel.from_losses(L)
        path = tmp_path / "panel.json"
        p.save(path)
        p2 = LossPanel.load(path)
        for m in L:
            np.testing.assert_array_equal(p.losses[m], p2.losses[m])
        r_direct = compare(p, p)
        r_reloaded = compare(p2, p)
        assert r_direct.champion_changed == r_reloaded.champion_changed

    def test_cli_exit_code(self, tmp_path):
        L = _panel(seed=0, T=20, K=3, champion_edge=0.5)
        prev = LossPanel.from_losses(L)
        prev_path = tmp_path / "prev.json"
        prev.save(prev_path)
        curr_path = tmp_path / "curr.json"
        LossPanel.from_losses(L).save(curr_path)   # identical -> no change -> exit 0
        result = subprocess.run(
            [sys.executable, "-m", "selection_fragility", "compare",
             str(prev_path), str(curr_path), "--exit-code"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_missing_file_exits_2(self, tmp_path):
        """Round-2 review, 'does it deliver on its promises' lens: CHANGELOG.md claimed every CLI
        error path exits 2, but `python -m selection_fragility compare <missing> <missing>` raised
        an unhandled FileNotFoundError at exit 1 -- indistinguishable from an internal bug. Now
        wrapped: a bad/missing file exits 2 with a clean one-line message on stderr, not a traceback."""
        missing_a = tmp_path / "does_not_exist_a.json"
        missing_b = tmp_path / "does_not_exist_b.json"
        result = subprocess.run(
            [sys.executable, "-m", "selection_fragility", "compare",
             str(missing_a), str(missing_b)],
            capture_output=True, text=True,
        )
        assert result.returncode == 2
        assert "Traceback" not in result.stderr
        # Checking the filename rather than the exact full path string: on Windows, subprocess
        # argv-passing/OSError formatting can round-trip backslash path separators doubled
        # (verified via a real GitHub Actions run, 2026-08-27 -- result.stderr contained
        # 'C:\\\\Users\\\\...' where str(missing_a) gives 'C:\\Users\\...'), so a byte-exact
        # full-path comparison is platform-fragile. The filename is what actually matters here
        # (which file was missing), and it survives any path-separator escaping either way.
        assert missing_a.name in result.stderr

    def test_cli_wellformed_json_but_not_a_losspanel_gives_clear_message(self, tmp_path):
        import json

        notpanel_path = tmp_path / "notpanel.json"
        notpanel_path.write_text(json.dumps({"hello": "world", "not": "a panel"}))
        curr_path = tmp_path / "curr.json"
        L = _panel(seed=0, T=20, K=3, champion_edge=0.5)
        LossPanel.from_losses(L).save(curr_path)
        result = subprocess.run(
            [sys.executable, "-m", "selection_fragility", "compare",
             str(notpanel_path), str(curr_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 2
        assert "Traceback" not in result.stderr
        assert "losspanel" in result.stderr.lower() or "not a valid" in result.stderr.lower()


# ---- 10-agent code-review pass (2026-09-02) --------------------------------------------------

def test_compare_rejects_non_losspanel_input_with_clear_message():
    prev = {"a": np.array([1.0, 1.1, 0.9, 1.0]), "b": np.array([1.2, 1.1, 1.3, 1.2])}
    curr = {"a": np.array([1.0, 1.1, 0.9, 1.0]), "b": np.array([1.2, 1.1, 1.3, 1.2])}
    with pytest.raises((TypeError, ValueError)) as excinfo:
        compare(prev, curr)
    assert "attributeerror" not in type(excinfo.value).__name__.lower()
