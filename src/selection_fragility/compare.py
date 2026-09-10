"""
selection_fragility.compare — compare(previous, current), the run-over-run champion-change diagnostic.
The single most-requested missing feature from the practitioner review: "the champion changed this
week -- is that signal or noise?"
"""
import numpy as np

from .fragility import _validate_losses, pooled_winner, decision_breakdown
from .identify import _run_mcs, _validate_alpha


class ChangeReport:
    """The result of `compare(previous, current)`. `.act` is the single bool a CI/pipeline promotion
    step should gate on: True only when the champion changed AND the old champion has genuinely left
    the Model Confidence Set (a real separation, not two models still statistically tied).

    ROUND-8 FIX (2026-08-27, fresh-eyes full-codebase review): on a genuinely non-uniform-weight
    panel, `arch.bootstrap.MCS` has no native support for the weights -- the same refusal
    `report()`/`resolution_report()` already surface cleanly (round 6/7 fixes) instead of computing
    an MCS on a silently different (unweighted) question than the champion itself was determined
    with. `compare()` previously let that ValueError propagate raw and uncaught. Now caught here:
    `old_champion_still_in_mcs` is `None` (undetermined, not a silent False) and `mcs_error` carries
    the reason. `.act` is conservatively forced to `False` in this case -- a CI/pipeline promotion
    step must never treat "MCS couldn't be computed" as "the old champion left the MCS."
    """

    def __init__(self, *, previous_champion, current_champion, champion_changed,
                 champion_without_new_periods, n_new_periods, old_champion_still_in_mcs,
                 k_star, churn_base_rate, act, mcs_error=None):
        self.previous_champion = previous_champion
        self.current_champion = current_champion
        self.champion_changed = champion_changed
        self.champion_without_new_periods = champion_without_new_periods
        self.n_new_periods = n_new_periods
        self.old_champion_still_in_mcs = old_champion_still_in_mcs
        self.k_star = k_star
        self.churn_base_rate = churn_base_rate
        self.act = act
        self.mcs_error = mcs_error

    def __repr__(self):
        if not self.champion_changed:
            head = f"no champion change ({self.current_champion}); act=False"
        elif self.champion_without_new_periods is None:
            head = (f"champion changed {self.previous_champion} -> {self.current_champion}; current "
                     f"shares no periods with previous -- cannot attribute the change; act={self.act}")
        elif self.champion_without_new_periods == self.previous_champion:
            head = (f"champion changed {self.previous_champion} -> {self.current_champion}, entirely "
                     f"due to the {self.n_new_periods} new period(s) (removing them reverts to the old "
                     f"champion); act={self.act}")
        else:
            head = f"champion changed {self.previous_champion} -> {self.current_champion}; act={self.act}"
        if self.mcs_error is not None:
            mcs = f"MCS undetermined ({self.mcs_error})"
        else:
            mcs = "still in the MCS" if self.old_champion_still_in_mcs else "left the MCS"
        return (f"ChangeReport({head}, old champion {mcs}, k*={self.k_star}, "
                f"churn_base_rate={self.churn_base_rate:.3f})")


def _churn_base_rate(L, w, n_perm=400, seed=0):
    """How often would the champion appear to change, RUN-OVER-RUN, out of pure noise -- i.e. if ONE
    new period arrives that behaves like a typical historical period in shape but carries no
    consistent model identity (a within-period label permutation applied to a single, randomly-drawn
    EXISTING period, then appended), how often does that alone flip the pooled champion away from the
    one observed on the T periods actually in hand? Distribution-free, built entirely from the
    caller's own data.

    REDESIGNED 2026-08-16 (independent 'wild' review): the original version independently permuted
    EVERY period's model-assignment on EVERY trial -- a full reshuffle of the whole T-period panel,
    not a single new period arriving. Verified directly this made the field measure almost exactly
    (K-1)/K regardless of the actual data (K=6 gave 0.83 whether the panel had no true skill
    difference OR one model with a massive true edge -- literally the wrong direction, since more
    real signal should make the champion MORE robust to a single new period, not equally fragile).
    The root cause: a full reshuffle destroys ALL cross-period structure, so every model's post-
    permutation pooled mean becomes an independent draw from the same pool and the "champion" is
    essentially a uniform pick among K labels. The fix keeps the T periods actually observed
    UNCHANGED (preserving whatever real, if fragile, lead the data has) and only randomizes the
    identity assignment of ONE new incoming period, drawn from the shape of a real historical period
    -- matching what a single real `compare()` call actually represents. Verified against
    tests/test_stage5_compare.py::test_base_rate_churn_cross_check's own independently-validated
    target (a TRUE null panel, one new null period added, ~0.08-0.35 with target ~0.18): the redesign
    lands at ~0.13, inside that range; the OLD version gave ~0.83, roughly 5x outside it, and the
    cross-check test only ever validated `champion_changed` from real `compare()` calls, never this
    internal field directly, which is how the gap went undetected across 4 prior review rounds.

    CONFIRMED REGRESSION (determinism/statelessness audit, 2026-08-17): `models = list(L)` used the
    caller's dict insertion order to build M's column order, so the SAME logical panel (same model
    names, same values) produced a DIFFERENT churn_base_rate depending purely on how the caller's
    dict happened to be constructed -- every other stochastic field in this package (winner_stability,
    pivot_agreement) avoids this since they re-dispatch through pooled_winner/decision_breakdown,
    which sort internally; this was the one function still shuffling at raw array-position level.
    Verified directly: same seed, same 6-model/30-period panel, reversed dict key order moved
    churn_base_rate from 0.42 to 0.465 -- a headline user-facing diagnostic swinging ~0.05 absolute
    purely from dict construction order, against a documented ~0.08-0.35 target band. Fixed the same
    way as identify.py::_to_frame and mcs.py::model_confidence_set's earlier order-dependence fixes:
    sort models by name so M's column order is a function of model identity, not insertion order."""
    # FIXED 2026-09-02 (10-agent code-review pass, CONFIRMED BUG): n_perm is a PUBLIC, documented
    # kwarg of compare() (e.g. a caller computing it from a config value that can go negative by
    # mistake). A negative n_perm made `range(n_perm)` empty (the tally stayed 0) so
    # `changes/n_perm` silently returned -0.0 -- indistinguishable in comparison semantics from a
    # genuine "zero churn expected by chance" reading -- with no resampling ever actually
    # performed and no error raised.
    if n_perm <= 0:
        raise ValueError(f"n_perm must be a positive integer; got {n_perm}.")
    # FIXED 2026-09-09 (round-2 stress-review, tool_source_audit lens, CONFIRMED with measured
    # impact): both champion picks below used to be a bare np.argmin(np.average(...)), with NO
    # degenerate-tie floor -- unlike pooled_winner() (fragility.py), which every OTHER champion
    # determination in this package goes through, including compare()'s own curr_champ/prev_champ
    # that this function's result is supposed to describe. On a near-tied panel (means differing by
    # ~5e-15, well inside pooled_winner's _MARGIN_REL_FLOOR=1e-12), the raw argmin can pick a
    # DIFFERENT model than pooled_winner's name-sorted tie-break -- reproduced directly: compare()
    # reported current_champion='a' while this function's un-floored argmin internally tracked
    # champion 'z', and churn_base_rate came out 0.029 (reads as very stable) when the rate computed
    # against the ACTUALLY-reported champion 'a' was 0.97 (the opposite conclusion), same call, same
    # data. Now routes both picks through pooled_winner() so this function's internal champion
    # notion always agrees with the one compare() reports to the caller.
    models = sorted(L)
    M = np.column_stack([np.asarray(L[m], float) for m in models])   # T x K, REAL unmodified data
    obs_champ_idx = models.index(pooled_winner({m: M[:, i] for i, m in enumerate(models)}, w))
    rng = np.random.default_rng(seed)
    T = M.shape[0]
    mean_w = float(np.mean(w))
    changes = 0
    for _ in range(n_perm):
        t_star = int(rng.integers(0, T))
        new_period = rng.permutation(M[t_star])   # same values as a real period, random model assignment
        Mp = np.vstack([M, new_period])
        wp = np.append(w, mean_w)
        perm_champ_idx = models.index(pooled_winner({m: Mp[:, i] for i, m in enumerate(models)}, wp))
        if perm_champ_idx != obs_champ_idx:
            changes += 1
    return changes / n_perm


def compare(previous, current, *, alpha=0.10, n_perm=400, seed=0):
    """Compare two LossPanels from consecutive runs of the same evaluation and report whether the
    pooled champion changed, and whether that change looks like signal or noise.

    Computes, all cheap: (1) did the pooled champion change; (2) recompute the champion on `current`
    restricted to the periods it shares with `previous` (by LABEL, not position) -- if that reverts to
    the OLD champion, the change is entirely the new data, exact and free; (3) is the old champion
    still in the current Model Confidence Set (if both remain tied, there is no real separation to act
    on); (4) k* of the new champion; (5) the within-period-permutation churn base rate (§2.6) -- how
    often the champion would appear to change out of pure label noise on data this size.

    A period in `current` counts as "new" iff its label does not appear anywhere in `previous.labels`
    -- this is a set-membership test, not a positional slice, so it is correct whether the new
    period(s) land at the end (the common append case), get inserted mid-panel by a merge/sort, or a
    rolling window drops an old period while adding a new one at the same total T (previously
    invisible to this diagnostic, since `n_new` was computed purely from `len(current) - len(previous)`
    and came out 0). If `current` shares NO periods with `previous` at all (e.g. two genuinely
    unrelated panels), there is nothing to restrict to -- `champion_without_new_periods` is reported as
    None rather than silently attributing the change to data it never actually removed.

    Raises ValueError if `previous` and `current` have different model sets -- comparing mismatched
    columns silently would be worse than refusing."""
    # FIXED 2026-09-02 (10-agent code-review pass, CONFIRMED GAP): compare() used to access
    # previous.models/.losses directly with no type check, so raw dicts (an easy mistake for a
    # caller who built L={'a':...} for the raw-array tier and assumes it also works here) raised a
    # bare AttributeError instead of a clear message. The CI/pipeline-gating use case this function
    # is built for makes an uncaught internal AttributeError especially bad: a promotion script
    # catching a narrow exception type would not catch this and would crash the pipeline rather
    # than fail cleanly.
    for _name, _p in (("previous", previous), ("current", current)):
        if not (hasattr(_p, "losses") and hasattr(_p, "models") and hasattr(_p, "weights")
                and hasattr(_p, "labels")):
            raise TypeError(
                f"compare() expects two LossPanel objects, got {type(_p).__name__} for {_name!r}. "
                f"Build one first with LossPanel.from_losses(...) or LossPanel.from_forecasts(...)."
            )
    # FIXED 2026-09-10 (round-6 stress-review, cli_end_to_end lens): an invalid `alpha` (e.g. the
    # "meant 10%" slip alpha=10, or a negative/nan value) used to reach _run_mcs() -> _validate_alpha
    # further down inside the try/except below, which exists ONLY to catch a different, legitimate
    # case (a non-uniform curr_w with no native MCS support). That broad `except ValueError` caught
    # this ValueError identically, silently classified it as "MCS undetermined", and forced
    # act=False -- so a real champion change combined with a malformed --alpha still exited 0 under
    # `--exit-code`, the flag whose entire purpose is failing CI on a real change. Validating alpha
    # here, before that try/except, lets it raise immediately and reach __main__.py's existing
    # ValueError -> exit 2 handling, exactly like every other bad-input path.
    _validate_alpha(alpha)
    if set(previous.models) != set(current.models):
        raise ValueError(
            f"previous and current panels have different model sets -- cannot compare mismatched "
            f"models: {sorted(previous.models)} vs {sorted(current.models)}."
        )
    prev_L, prev_w = previous.losses, previous.weights
    curr_L, curr_w = current.losses, current.weights
    _validate_losses(curr_L, curr_w)

    prev_champ = pooled_winner(prev_L, prev_w)
    curr_champ = pooled_winner(curr_L, curr_w)
    champion_changed = prev_champ != curr_champ

    curr_labels = list(current.labels)
    T_curr = len(curr_labels)
    # POSITIONAL-LABEL FALLBACK, FIXED 2026-08-17 (independent 'wild' review, cross-fix interaction
    # lens): the label-based new-period detection above assumes labels carry real cross-panel period
    # IDENTITY. `LossPanel.from_losses()`'s default construction (no explicit `labels=`, the most
    # common real usage -- a bare dict of arrays or an ndarray) assigns synthetic positional labels
    # `0..T-1` and sets `labels_are_positional=True`. Two ENTIRELY UNRELATED panels built this way
    # collide on labels purely by list-index coincidence, defeating this function's own "unrelated
    # panels -> None, don't guess" safety net -- verified directly: two panels built from independent
    # random draws (no shared periods in any real sense) still reported the champion change as
    # "entirely due to new periods" or wrongly claimed 10 shared periods, purely because both used
    # the default 0..9 labels. Positional labels carry NO real identity information at all outside
    # the one panel that assigned them, so label-set comparison is meaningless here -- fall back to
    # the position/count-based detection (assume the newest periods are the trailing rows) whenever
    # EITHER side lacks real labels, which is the best available signal absent any identity
    # information the caller chose not to provide. Real, non-positional labels (the case this
    # function's label-based redesign was actually built for) are unaffected.
    if previous.labels_are_positional or current.labels_are_positional:
        T_prev = len(previous.labels)
        n_new = max(0, T_curr - T_prev)
        if n_new == 0:
            # CONFIRMED REGRESSION (control-flow/static-logic audit, 2026-08-17): this function's own
            # docstring promises "if `current` shares NO periods with `previous` at all... reported
            # as None rather than silently attributing the change to data it never actually removed"
            # -- the label-based branch below honors that (n_new==T_curr -> None), but this positional
            # branch did not: `n_new = max(0, T_curr - T_prev)` is forced to 0 whenever T_curr <= T_prev,
            # regardless of whether the two panels share anything real, so it fell straight through to
            # a confident `curr_champ` instead. Positional labels carry NO identity information at
            # all (both panels always trivially have labels 0..T-1), so unlike the label-based branch,
            # there is no way to tell "the literal same panel, nothing new" apart from "an entirely
            # unrelated panel of the same or smaller size" here -- verified directly: two panels built
            # from independent random draws, both length 10, positional labels, reported a confident
            # champion_without_new_periods instead of the honest None this diagnostic exists to give
            # when it cannot actually attribute the change to specific removed data. T_curr < T_prev
            # is even less determinable (nothing to even align positionally), so None applies there
            # too, not just at T_curr == T_prev.
            champion_without_new_periods = None
        else:
            keep = list(range(T_curr - n_new))
            L_trunc = {m: np.asarray(v, float)[keep] for m, v in curr_L.items()}
            w_trunc = np.asarray(curr_w, float)[keep]
            champion_without_new_periods = pooled_winner(L_trunc, w_trunc)
    else:
        # STR-NORMALIZED, FIXED 2026-08-17 (caught in self-review before shipping): `LossPanel.save()`
        # stringifies every label (`[str(l) for l in self.labels]`), so a panel round-tripped through
        # save()/load() has str labels while a freshly-built panel with the SAME periods can have
        # Timestamp (or other) labels -- exactly the real `compare()` workflow (load last run's saved
        # baseline, compare against a freshly computed current run). Comparing raw label objects made
        # EVERY period in `current` look "new" whenever only one side had been through a save/load
        # round trip (Timestamp('2020-01-31') != '2020-01-31 00:00:00' by identity, even though they
        # name the same period) -- reproduced directly: n_new_periods came out as T instead of 0 on two
        # otherwise-identical panels. Comparing str(label) on both sides matches what save() already
        # does, so a loaded and a freshly-built panel over the same periods agree regardless of which
        # side (if either) went through a round trip.
        prev_label_set = {str(lbl) for lbl in previous.labels}
        is_new = [str(lbl) not in prev_label_set for lbl in curr_labels]
        n_new = sum(is_new)
        if n_new == 0:
            champion_without_new_periods = curr_champ
        elif n_new == T_curr:
            champion_without_new_periods = None
        else:
            keep = [i for i, new in enumerate(is_new) if not new]
            L_trunc = {m: np.asarray(v, float)[keep] for m, v in curr_L.items()}
            w_trunc = np.asarray(curr_w, float)[keep]
            champion_without_new_periods = pooled_winner(L_trunc, w_trunc)

    # w THREADED THROUGH, FIXED 2026-08-16 (independent 'wild' review): matches the same fix in
    # report()/resolution_report() -- the MCS must see the same weights the champion is computed
    # with, or `old_champion_still_in_mcs` can silently answer a different, unweighted question.
    #
    # ROUND-8 FIX (2026-08-27): a genuinely non-uniform curr_w has no native MCS support in
    # arch.bootstrap.MCS -- report()/resolution_report() already refuse cleanly rather than compute
    # the MCS on a silently different (unweighted) question; this call previously let that
    # ValueError propagate raw and uncaught. Caught here the same way, undetermined not silently
    # False, and `.act` is forced False so a CI/pipeline promotion step never treats "couldn't
    # compute" as "genuinely left the MCS."
    # FIXED 2026-09-09 (round-2 stress-review, tool_source_audit lens): this block used to call
    # mcs_size(curr_L, ...) as a bare, discarded statement immediately before _run_mcs(curr_L, ...)
    # with IDENTICAL arguments -- mcs_size() is itself just len(_run_mcs(...).included)
    # (identify.py), so that ran the full arch.bootstrap.MCS elimination twice per compare() call
    # for no reason (confirmed by timing: ~2x a single _run_mcs() call). The ValueError this try/
    # except exists to catch (a non-uniform curr_w with no native MCS support) is raised identically
    # by _run_mcs() alone, so removing the redundant call changes no behavior.
    mcs_error = None
    try:
        m = _run_mcs(curr_L, alpha=alpha, seed=seed, w=curr_w)
        old_champion_still_in_mcs = prev_champ in set(m.included)
    except ValueError as e:
        old_champion_still_in_mcs = None
        mcs_error = str(e)

    k_curr, _opp_curr, _removed_curr = decision_breakdown(curr_L, curr_w, a=curr_champ)

    churn = _churn_base_rate(curr_L, curr_w, n_perm=n_perm, seed=seed)

    act = bool(champion_changed and old_champion_still_in_mcs is False)

    return ChangeReport(
        previous_champion=prev_champ,
        current_champion=curr_champ,
        champion_changed=champion_changed,
        champion_without_new_periods=champion_without_new_periods,
        n_new_periods=n_new,
        old_champion_still_in_mcs=old_champion_still_in_mcs,
        k_star=k_curr,
        churn_base_rate=churn,
        act=act,
        mcs_error=mcs_error,
    )
