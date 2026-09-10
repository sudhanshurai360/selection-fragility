"""
selection_fragility.panel — LossPanel, the single validated entry point for every public function in
this package.

Every other module in this package consumes a LossPanel (or its raw `.losses`/`.weights` dict+array,
which are the SAME validated data). The bare constructor is not public -- go through `.from_losses()`
or `.from_forecasts()` so a caller cannot skip validation by accident.
"""
import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd

from .fragility import _MAX_ABS_LOSS


def _is_pandas(obj):
    return hasattr(obj, "columns") and hasattr(obj, "index")


def _validate_weights(w, T):
    # copy=True, FIXED 2026-08-17 (independent 'wild' review, mutation/aliasing lens): `np.asarray`
    # only copies when a dtype conversion is actually needed -- a caller-supplied float64 array (the
    # common case; the standard output of any numpy/pandas computation) sailed through unchanged, so
    # the "validated" LossPanel silently ALIASED the caller's own weights array. A caller mutating
    # their own array after building the panel -- routine, unrelated code that never touches the
    # panel object -- then silently changed the "already-validated" panel's weights underneath it.
    # LossPanel's own docstring calls itself "the single validated entry point"; that guarantee
    # requires the data to actually be copied in, not merely dtype-coerced.
    # FIXED 2026-09-10 (round-6 stress-review, security_resource_exhaustion/error_message_quality
    # lenses): a bare, un-guarded `np.array(w, dtype=float, copy=True)` let a realistic weights
    # mistake (a dict, a list with one bad string) escape as a raw numpy/Python error naming neither
    # "weights" nor this function -- a striking gap given every OTHER input in this "fanatically
    # validated" package gets a clear domain message. A bare string is called out specially: the
    # only string weights accepts anywhere in the package is from_forecasts()'s "count" shorthand,
    # which from_losses() (routed through this same function) does not support -- a caller who
    # learned the shorthand from one constructor got no hint it doesn't apply to the other.
    if isinstance(w, str):
        raise TypeError(
            f"weights must be an array-like of {T} numeric values (one per period), not a string; "
            f"got {w!r}. The 'count' shorthand is only accepted by LossPanel.from_forecasts() "
            f"(weight each period by its row count) -- from_losses() has no string shorthand."
        )
    try:
        w = np.array(w, dtype=float, copy=True)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"weights must be an array-like of {T} numeric values (one per period); got "
            f"{type(w).__name__} that could not be converted to a numeric array ({e})."
        ) from e
    if w.size != T:
        raise ValueError(f"weights must have length {T} (one per period); got length {w.size}.")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must not contain NaN/inf values.")
    if np.any(w < 0):
        raise ValueError("weights must not contain negative values -- weights are observation counts, "
                          "not signed contributions.")
    if np.all(w == 0):
        raise ValueError("weights are all zero; at least one period must carry positive weight.")
    return w


def _looks_like_metadata_column(arr, T):
    """An integer-valued column that is either an exact ARITHMETIC SEQUENCE (constant step -- a row
    index or step counter, e.g. 0,1,2,...,T-1) or a short REPEATING/PERIODIC pattern (e.g. a horizon
    index 1,2,3,1,2,3,...) looks like structural metadata, not a per-period loss -- real loss data is
    essentially never either. FIXED 2026-08-16 (independent review): an earlier version used
    cardinality alone (`nunique <= some floor`) as the signal, which is wrong in BOTH directions --
    verified directly: it wrongly REJECTS a genuine low-cardinality integer loss column (e.g. rounded-
    dollar losses with only 6 distinct values over T=30) and wrongly ACCEPTS a high-cardinality
    step-counter column (a distinct value every period, so cardinality alone never flags it). Neither
    failure mode applies to a periodicity/arithmetic-sequence check, which is about the SHAPE of the
    sequence, not how many distinct values it has -- verified against both counterexamples plus the
    original `horizon` regression fixture."""
    arr = np.asarray(arr)
    if T < 4:
        return False
    # CONFIRMED REGRESSION (independent review, 2026-08-16), found while testing the fix that removed
    # this function's `is_integer_dtype` caller-side gate: a genuinely CONSTANT loss column (a real,
    # if boring, signal -- e.g. a baseline that scored identically every period) was being wrongly
    # flagged by the periodic-tiling check below, since a constant array trivially "tiles" at every
    # period length. A constant value is degenerate real data, not an index/step pattern -- exclude it
    # explicitly before the shape checks, rather than let it fall through to the periodicity loop.
    if np.all(arr == arr[0]):
        return False
    diffs = np.diff(arr)
    if diffs.size > 0 and np.all(diffs == diffs[0]) and diffs[0] != 0:
        return True
    for p in range(2, min(10, T // 2) + 1):
        # CONFIRMED REGRESSION (control-flow/static-logic audit, 2026-08-17): `reps = T // p` FLOORS,
        # so when `p` doesn't evenly divide `T` the comparison only covered the truncated prefix
        # `arr[:reps*p]` -- the trailing `T - reps*p` elements were never checked against the pattern
        # at all, i.e. a genuine real loss array whose first `reps*p` values happen to prefix-tile a
        # short pattern but whose FINAL values clearly break it was still flagged as periodic
        # metadata and silently rejected. Verified directly: [1,2,3,1,2,3,1,2,3,999] (period-3 for 9
        # of 10 values, then a clear outlier) was misclassified as metadata and its model rejected
        # from the panel -- exactly the "wrongly REJECTS a genuine ... loss column" failure mode this
        # function's own docstring says the periodicity check exists to avoid. Fixed by tiling out to
        # (and comparing against) the FULL length-T array via ceiling division, so a trailing partial
        # period must also continue the pattern, not just the part that happened to divide evenly.
        reps_needed = -(-T // p)   # ceiling division: enough full tiles of length p to reach >= T
        if T // p >= 2 and np.array_equal(arr, np.tile(arr[:p], reps_needed)[:T]):
            return True
    return False


def _check_metadata_columns(models, T):
    # CONFIRMED GAP (claims-vs-implementation traceability audit, 2026-08-17): `_from_wide_dataframe`
    # runs every column through `_looks_like_metadata_column` before accepting it as a model, but the
    # dict-input path (`_from_dict`) built its model arrays directly and never called the heuristic at
    # all -- a dict key named 'horizon' holding an exact arithmetic sequence (or any other stray
    # metadata column) was silently crowned a model. Verified directly:
    # LossPanel.from_losses({'horizon': np.arange(1,11,dtype=float), 'model_a': ...}) previously
    # accepted 'horizon' as a model with zero warning. Shares the one heuristic with the DataFrame
    # path so both surfaces stay in lockstep rather than drifting into two independently-maintained
    # copies of the same check.
    if T <= 0:
        return
    for name, arr in models.items():
        if _looks_like_metadata_column(arr, T):
            raise ValueError(
                f"column '{name}' looks like a categorical/metadata column (an exact arithmetic "
                f"sequence or a short repeating pattern across {T} periods), not a per-period "
                f"loss -- a stray metadata column must never be silently crowned a model. Drop "
                f"it, or build the panel from an explicit dict of the real model columns only."
            )


def _check_no_str_collision(raw_labels):
    # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-16): every model dict built by this
    # module is keyed by `str(label)`, but the duplicate-label guards upstream (e.g.
    # `df.columns.duplicated()`) check the RAW labels, not their stringified form. Two distinct, legal,
    # non-duplicate raw labels that collide only after str() -- e.g. an int column `1` and a str column
    # `'1'` -- silently merge into one model dict entry, discarding one model's data with zero warning.
    # Checked wherever raw labels are turned into str-keyed model dicts.
    seen = {}
    for lbl in raw_labels:
        seen.setdefault(str(lbl), []).append(lbl)
    collided = {s: ls for s, ls in seen.items() if len(ls) > 1}
    if collided:
        detail = ", ".join(f"{ls!r} -> '{s}'" for s, ls in collided.items())
        raise ValueError(f"distinct column/model labels collide once stringified ({detail}) -- "
                          f"LossPanel keys models by str(label), so these would silently merge into "
                          f"one model, discarding the others; rename before building a LossPanel.")


_LARGE_K_WARN_THRESHOLD = 100   # see the warning message below for the measured cost curve this is based on


def _check_K_vs_T(K, T):
    if K > T:
        warnings.warn(
            f"K={K} models but T={T} periods (K > T) -- this looks like it might be a transposed frame "
            f"(models and periods swapped); did you mean to pass the data transposed?",
            UserWarning, stacklevel=3,
        )
    # LARGE-K COST WARNING ADDED 2026-09-07 (round-4 8-lens PyPI-preflight audit, CONFIRMED HIGH):
    # report()/identified()/mcs_size() delegate to arch.bootstrap.MCS, an iterative elimination
    # procedure with no documented cost guidance anywhere in this package despite mcs()'s own B and
    # loss-magnitude both having explicit, enforced ceilings. Measured directly (T=50, this
    # package's own report()): K=100 -> 0.9s, K=200 -> ~3s (post-dedup; ~9s before report() stopped
    # recomputing the same MCS four times), K=500 did not finish within 100s. This is a genuine
    # scaling cliff a moderately-sized model sweep (e.g. an AutoML/hyperparameter search) can hit
    # with no error and no progress indicator -- warn once, at panel-construction time (before any
    # expensive computation has actually started), rather than leaving a user to just wait. Not
    # raised as an error (unlike mcs()'s B/loss-magnitude guards): unlike those, a large-but-real K
    # is a legitimate, supported use case, just a slow one -- unlike a K>T transposed frame, which
    # is a near-certain user mistake, so that warning stays separate (elif, not additionally, to
    # avoid double-warning the same likely-mistake case).
    elif K > _LARGE_K_WARN_THRESHOLD:
        warnings.warn(
            f"K={K} models -- report()/identified()/mcs_size() cost grows steeply with model count "
            f"(measured on this package's own benchmark: ~1s at K=100, ~3s at K=200, potentially "
            f"minutes at K=500+). This is not an error -- large model sweeps are supported -- but "
            f"expect it to be slow; if you don't need per-model MCS membership, consider narrowing "
            f"to a candidate shortlist first. NOTE: this warning is keyed on K only -- report()'s "
            f"PIVOT section can also get expensive at large T once a panel resolves, even at a "
            f"modest K well under this threshold; see README.md's 'A note on cost' for measured "
            f"numbers.",
            UserWarning, stacklevel=3,
        )


class LossPanel:
    """A validated {model: per-period loss array} panel, plus period labels and per-period weights.

    Construction always goes through `.from_losses()` (wide DataFrame / ndarray / dict of per-model
    arrays) or `.from_forecasts()` (long forecast-vs-actual frame, losses computed here). The bare
    constructor raises -- it exists only so the two classmethods and `.load()` can build an instance
    after they have already done the validation.
    """

    def __init__(self, losses, weights, labels, labels_are_positional, _validated=False):
        if not _validated:
            raise RuntimeError("LossPanel() is not a public constructor; use .from_losses() or "
                                ".from_forecasts() so input validation cannot be bypassed.")
        self.losses = losses
        self.weights = weights
        self.labels = list(labels)
        self.labels_are_positional = labels_are_positional
        self.models = list(losses)

    def __repr__(self):
        try:
            from .report import report
            return report(self)
        except Exception:
            return f"LossPanel(models={self.models}, T={len(self.labels)})"

    # ------------------------------------------------------------------ from_losses ----
    @classmethod
    def from_losses(cls, data, weights=None, labels=None):
        if _is_pandas(data):
            loss_dict, inferred_labels, labels_are_positional = cls._from_wide_dataframe(data)
        elif isinstance(data, dict):
            loss_dict, inferred_labels, labels_are_positional = cls._from_dict(data)
        elif isinstance(data, np.ndarray):
            loss_dict, inferred_labels, labels_are_positional = cls._from_ndarray(data)
        else:
            raise TypeError(f"from_losses() does not know how to read {type(data).__name__}; pass a "
                             f"wide DataFrame, a dict of {{model: array}}, or a 2-D ndarray (periods x models).")

        T = len(next(iter(loss_dict.values())))
        if T == 0:
            raise ValueError("panel has zero periods.")
        if T == 1:
            raise ValueError("panel has only 1 period; need at least 2 periods to compare models.")
        if len(loss_dict) < 2:
            raise ValueError(f"need at least 2 models to compare a selection decision; got {len(loss_dict)}.")

        for m, v in loss_dict.items():
            # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-17, security/hostile-input
            # lens): T above is taken from the FIRST model's length only -- a dict of plain arrays
            # (not pandas Series, which already get their own index-equality check) with mismatched
            # per-model lengths passed straight through with no error, silently building a LossPanel
            # on ragged data that only crashed later, downstream, on first real analysis call, with
            # an error message that never mentioned this as the actual source.
            if len(v) != T:
                raise ValueError(f"all models must share the same number of periods; '{m}' has "
                                  f"{len(v)}, expected {T} (from the first model).")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"model '{m}' has NaN/inf loss values; drop or impute those periods "
                                  f"before building a LossPanel -- a single NaN silently poisons the "
                                  f"pooled mean and can crown a spurious winner.")
            # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-17, numerical-extremes lens):
            # loss magnitudes beyond ~1e150 silently overflow float64 arithmetic downstream (squared
            # differentials, summed margins), producing self-contradictory reports rather than a
            # clear error -- see fragility.py's `_MAX_ABS_LOSS` for the full story. This one check,
            # on the FINAL loss_dict shared by every from_losses() input path (DataFrame/dict/
            # ndarray), covers all of them uniformly. No real per-period loss metric approaches
            # this magnitude; 1e100 is a huge margin below where the overflow actually starts.
            if np.any(np.abs(np.asarray(v, float)) > _MAX_ABS_LOSS):
                raise ValueError(f"model '{m}' has a loss magnitude above {_MAX_ABS_LOSS:.0e}, far beyond any real "
                                  f"per-period loss metric -- arithmetic at this scale silently overflows "
                                  f"float64 downstream and produces meaningless results rather than a "
                                  f"clear error. Check for a units/scaling bug upstream.")

        _check_K_vs_T(len(loss_dict), T)

        if labels is not None:
            final_labels = list(labels)
            # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-17, security/hostile-input
            # lens): an explicit `labels=` of the wrong length was never checked against T. Too
            # SHORT crashed downstream with a bare, unhelpful IndexError; too LONG produced NO
            # crash at all -- report() silently printed a wrong period count ("k*=1 of 10 periods"
            # on 5 actual periods of data), confidently-wrong output rather than an error.
            if len(final_labels) != T:
                raise ValueError(f"labels has length {len(final_labels)}, expected {T} (one per "
                                  f"period, matching the loss arrays).")
            # CONFIRMED REGRESSION (blind re-implementation audit flagged the gap, confirmed and
            # traced to a real silently-wrong compare() answer, 2026-08-17): `_from_wide_dataframe`
            # has enforced `df.index.is_unique` since day one, but this explicit `labels=` path (the
            # one dict/ndarray callers actually use) only ever checked LENGTH, never uniqueness --
            # duplicate label text was silently accepted. compare()'s new-period detection is a
            # set-membership test on str(label) (see compare.py's own docstring: "a period in
            # `current` counts as new iff its label does not appear anywhere in `previous.labels`"),
            # so a genuinely NEW period whose label happens to text-collide with an EXISTING one
            # (plausible from any upstream relabeling bug or reused date string) is silently treated
            # as old. Reproduced directly: previous=10 periods (true champion 'a'), current=previous
            # plus one new period with a dramatic edge for 'b' labeled to collide with an existing
            # label -- compare() reported n_new_periods=0 and champion_without_new_periods='b', the
            # OPPOSITE of the correct answer ('a', the true champion restricted to the actual 10 old
            # periods) -- exactly the "entirely due to new data" signal this diagnostic exists to
            # give, silently backwards. Enforcing uniqueness at construction, matching the DataFrame
            # path, closes this at the source rather than needing per-consumer defenses.
            if len(set(str(l) for l in final_labels)) != T:
                dupes = sorted({str(l) for l in final_labels if
                                 sum(str(x) == str(l) for x in final_labels) > 1})
                raise ValueError(f"labels contains duplicate value(s) {dupes[:5]}{'...' if len(dupes) > 5 else ''} "
                                  f"-- labels must uniquely identify each period, since compare() relies on "
                                  f"label identity (as text) to tell a new period from an old one; a duplicate "
                                  f"silently lets a genuinely new period masquerade as an old one.")
            labels_are_positional = False
        else:
            final_labels = inferred_labels

        w = _validate_weights(weights, T) if weights is not None else np.ones(T)
        return cls(loss_dict, w, final_labels, labels_are_positional, _validated=True)

    @staticmethod
    def _from_wide_dataframe(df):
        if df.columns.duplicated().any():
            dups = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(f"duplicate column names {dups} in the frame -- rename before building "
                              f"a LossPanel; pandas silently keeps both under the hood.")
        if not df.index.is_unique:
            raise ValueError("duplicate period values in the index -- cannot silently keep both rows "
                              "as independent periods; deduplicate or aggregate first.")
        _check_no_str_collision(df.columns)

        T = len(df)
        models = {}
        for col in df.columns:
            s = df[col]
            if not pd.api.types.is_numeric_dtype(s):
                raise TypeError(f"column '{col}' is not numeric (dtype {s.dtype}); a LossPanel model "
                                 f"column must be a per-period numeric loss. Drop it, or build the panel "
                                 f"from an explicit dict of the model columns only.")
            if pd.api.types.is_bool_dtype(s):
                # pandas bool is numeric-but-not-integer (is_integer_dtype(bool)==False), so a
                # True/False flag column (a promo/holiday/beat-baseline indicator -- extremely common
                # in real eval frames) passed the is_numeric_dtype check above and, unless it happened
                # to tile perfectly, could reach the periodicity check undetected. A boolean value can
                # never be a legitimate per-period loss -- reject it outright rather than rely on the
                # heuristic catching it only by luck. CONFIRMED (independent 'wild' review, 2026-08-16).
                raise TypeError(f"column '{col}' is boolean, not a per-period numeric loss -- looks "
                                 f"like a flag/indicator column, not a model. Drop it, or build the "
                                 f"panel from an explicit dict of the model columns only.")
            # copy=True -- see _validate_weights's comment; `to_numpy(dtype=float)` alone only copies
            # when a dtype conversion is actually needed, aliasing the DataFrame's own backing array
            # for the (common) already-float64 case.
            arr = s.to_numpy(dtype=float, copy=True)
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                raise ValueError(f"model '{col}' has {int(nan_mask.sum())} NaN value(s) out of {T} "
                                  f"periods; drop or impute those periods before building a LossPanel.")
            # CONFIRMED REGRESSION (independent review, 2026-08-16): this guard was gated on
            # `is_integer_dtype`, but the periodicity/arithmetic-sequence check itself is dtype-blind
            # -- it only looks at the VALUES. A column with the identical horizon/step-index pattern
            # stored as float64 (extremely plausible after any merge, groupby().mean(), or CSV read
            # without explicit dtype control -- all of which silently upcast) sailed straight through
            # and was crowned champion with a confident "identified"/"resolved" report. Applied to
            # ANY numeric column now, regardless of dtype -- real continuous loss data essentially
            # never forms an exact arithmetic sequence or short repeating cycle whether stored as int
            # or float, so this is not expected to create new false positives.
            if T > 0 and _looks_like_metadata_column(arr, T):
                raise ValueError(
                    f"column '{col}' looks like a categorical/metadata column (an exact arithmetic "
                    f"sequence or a short repeating pattern across {T} periods), not a per-period "
                    f"loss -- a stray metadata column must never be silently crowned a model. Drop "
                    f"it, or build the panel from an explicit dict of the real model columns only."
                )
            models[str(col)] = arr

        if len(models) == 0:
            raise ValueError("zero model columns found in the frame after validation.")

        labels = list(df.index)
        return models, labels, False

    @staticmethod
    def _from_dict(data):
        if len(data) == 0:
            raise ValueError("empty dict passed to from_losses(); need at least 2 models.")
        _check_no_str_collision(data.keys())
        vals = list(data.values())
        # FIXED 2026-09-02 (10-agent code-review pass, CONFIRMED SEVERE finding, independently
        # reproduced by two separate reviews): this used to be `all(isinstance(v, pd.Series) ...)`
        # -- the index-alignment check below only fired when EVERY value was a Series. Mixing even
        # ONE plain list/array with a Series (a realistic shape: one model's losses came from a
        # groupby/merge that wasn't re-sorted, the other from a plain np.array the caller assumes
        # is already period-ordered) silently fell through to the `else` branch, which does
        # `np.array(v, dtype=float)` on EVERY value -- discarding the Series' own index entirely
        # and reading it by raw storage position. Demonstrated: a 5-year panel with one model's
        # Series index reversed relative to a sibling plain array produced k*=5 where the true,
        # correctly-aligned answer is k*=1 (a 5x overstatement of decision robustness), and
        # separately a complete plurality-winner reversal. Now checks ANY value being a Series
        # (not just ALL), and rejects the mix outright rather than guessing how to align a plain
        # array that carries no index of its own -- ambiguous input should raise, not be silently
        # resolved by assumption.
        any_series = any(isinstance(v, pd.Series) for v in vals)
        all_series = all(isinstance(v, pd.Series) for v in vals)
        if any_series and not all_series:
            non_series = [k for k, v in data.items() if not isinstance(v, pd.Series)]
            raise ValueError(
                f"mixing pandas Series with plain arrays/lists in the same from_losses() dict is "
                f"ambiguous and unsafe: {non_series} have no index to align against the Series' "
                f"own index, and silently assuming they share the Series' period order risks "
                f"misaligning periods across models with no error. Pass ALL models as Series "
                f"sharing a common index, or ALL as plain arrays/lists in a already-aligned "
                f"shared order, not a mix of both."
            )
        # FIXED 2026-09-02 (10-agent code-review pass, three CONFIRMED gaps, all reachable only
        # via this dict-of-arrays path -- _from_wide_dataframe already guards all three):
        #  (a) a boolean array (a promo/holiday/beat-baseline flag -- exactly the stray-column
        #      shape _from_wide_dataframe's own is_bool_dtype guard exists for) was silently
        #      crowned a model;
        #  (b) a scalar (0-D) value crashed later with a raw, unhelpful `TypeError: len() of
        #      unsized object` from the T-inference line below, instead of a clear ValueError
        #      naming the actual cause;
        #  (c) a >1-D "model" array (e.g. an accidentally-nested list) was accepted at
        #      CONSTRUCTION time and only crashed on first analysis call, and even then
        #      __repr__'s broad `except Exception` fallback masked the malformed shape from
        #      `print(panel)`, the very sanity check callers reach for.
        # Checked once here, before either branch below builds `models`, so the message is
        # attributed to the actual offending key regardless of which branch handles it.
        for k, v in data.items():
            if np.isscalar(v) or (hasattr(v, "ndim") and getattr(v, "ndim", 1) == 0):
                raise ValueError(
                    f"model '{k}' is a scalar, not a per-period array -- from_losses() needs each "
                    f"model's value to be a 1-D sequence of per-period losses."
                )
            # FIXED 2026-09-10 (round-7 final pre-publish review, adversarial_input_fuzzing lens): a
            # RAGGED (jagged, unequal-inner-length) nested list -- e.g. a tampered/malformed saved
            # LossPanel JSON file's "losses" dict -- made the bare np.asarray(v) below raise a raw
            # numpy internals message ("setting an array element with a sequence...") naming neither
            # "LossPanel", "from_losses", nor the offending model key, unlike every sibling shape/
            # dtype check in this same loop. The sibling weights path (_validate_weights, below) was
            # already protected against the identical ragged-list case; this closes the same gap
            # here.
            try:
                arr_for_shape = v.to_numpy() if isinstance(v, pd.Series) else np.asarray(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"model '{k}'s value could not be converted to an array -- from_losses() "
                                  f"needs each model's value to be a rectangular 1-D sequence of per-period "
                                  f"losses, not a ragged/jagged nested sequence ({e}).") from e
            if pd.api.types.is_bool_dtype(getattr(v, "dtype", arr_for_shape.dtype)):
                raise TypeError(
                    f"model '{k}' is boolean, not a per-period numeric loss -- looks like a "
                    f"flag/indicator column, not a model. Drop it, or build the panel from an "
                    f"explicit dict of the model columns only."
                )
            if arr_for_shape.ndim != 1:
                raise ValueError(
                    f"model '{k}' has shape {arr_for_shape.shape}, not 1-D -- from_losses() needs "
                    f"each model's value to be a flat per-period array, not a nested/multi-"
                    f"dimensional one."
                )
        if all_series:
            first_idx = vals[0].index
            for k, v in data.items():
                if not v.index.equals(first_idx):
                    raise ValueError(
                        "dict of pandas Series has mismatched indices across models -- cannot silently "
                        "align/reindex, since that would fabricate a shared period axis that was never "
                        "actually there. Pass a common index, or plain arrays/lists instead."
                    )
            labels = list(first_idx)
            # copy=True on both branches below -- see _validate_weights's comment. A dict of
            # already-float64 Series/arrays (the common case) sailed through `np.asarray` unchanged,
            # aliasing the caller's own arrays; the panel's `.losses` then silently changed underneath
            # it whenever the caller mutated their own dict/Series, with no error or warning.
            models = {str(k): np.array(v.to_numpy(), dtype=float, copy=True) for k, v in data.items()}
            _check_metadata_columns(models, len(labels))
            return models, labels, False
        models = {str(k): np.array(v, dtype=float, copy=True) for k, v in data.items()}
        T = len(next(iter(models.values()))) if models else 0
        _check_metadata_columns(models, T)
        return models, list(range(T)), True

    @staticmethod
    def _from_ndarray(arr):
        if arr.ndim != 2:
            raise ValueError(f"ndarray input to from_losses() must be 2-D (periods x models); got "
                              f"shape {arr.shape}.")
        T, K = arr.shape
        # CONFIRMED REGRESSION (2026-08-26, reproduced directly): a (T, 0)-shaped ndarray (zero model
        # columns) produced an empty `models` dict here, which `from_losses()` then fed to
        # `T = len(next(iter(loss_dict.values())))` -- `next(iter({}.values()))` raises a bare
        # StopIteration instead of a clear ValueError, unlike every sibling zero-model path
        # (`_from_wide_dataframe`'s "zero model columns found" check, `_from_dict`'s empty-dict check).
        # Guarded here at the source, matching those siblings, so the error is raised with the actual
        # cause instead of a confusing StopIteration surfacing from unrelated code two frames away.
        if K == 0:
            raise ValueError(f"ndarray has zero model columns (shape {arr.shape}); need at least 2 "
                              f"models to compare a selection decision.")
        models = {f"m{i}": arr[:, i].astype(float) for i in range(K)}
        # _looks_like_metadata_column keys only on VALUES, not names -- a positional column has no
        # user-supplied name for the heuristic to skip, but a stray metadata column (e.g. a horizon
        # index) smuggled in as one of the array's columns is exactly as detectable by value here as
        # it is in the DataFrame/dict paths. Applied for the same reason and via the same shared check.
        _check_metadata_columns(models, T)
        return models, list(range(T)), True

    # --------------------------------------------------------------- from_forecasts ----
    @classmethod
    def from_forecasts(cls, df, y_true, period, group=None, models=None, metric="mae", weights=None):
        if not _is_pandas(df):
            raise TypeError("from_forecasts() expects a pandas DataFrame.")
        # FIXED 2026-09-10 (round-6 stress-review, error_message_quality lens): these three
        # "named column not found" cases raised KeyError while the two sibling cases below (models=,
        # weights=) already raise ValueError for the identical situation -- an inconsistency with no
        # reason behind it, and KeyError's own __str__ wraps the message in an extra pair of quotes
        # (a "not found in the frame."" artifact a caller printing str(e) would see). Standardized to
        # ValueError, matching the majority precedent already set by models=/weights=.
        if y_true not in df.columns:
            raise ValueError(f"y_true column '{y_true}' not found in the frame.")
        if period not in df.columns:
            raise ValueError(f"period column '{period}' not found in the frame.")
        if group is not None:
            if group not in df.columns:
                raise ValueError(f"group column '{group}' not found in the frame.")
            if group == period:
                raise ValueError(f"group and period must be different columns; got the same column "
                                  f"'{group}' passed for both.")

        structural = {y_true, period} | ({group} if group else set())
        coerced = {}   # column name -> numeric-coerced Series, for any candidate found via dtype coercion below
        if models is None:
            # AUTO-INFERENCE STRAY-COLUMN GUARD ADDED 2026-08-16 (confirmed regression, independent
            # review): this branch previously accepted ANY numeric, non-structural column as a model
            # -- unlike from_losses()'s wide-DataFrame path, it had ZERO protection against a stray
            # metadata column (e.g. a 'horizon' step index) despite an earlier design note claiming
            # this exact bug class was closed. Verified directly: a nixtla-style CV frame with a
            # `horizon` column cycling 1,2,3 was silently inferred as a fourth "model" and ranked
            # (mean loss ~98) alongside the three real models (mean loss ~2-3). Only applied when
            # inferring automatically -- an explicit `models=[...]` list is trusted as-is.
            # Applied to ANY numeric column regardless of dtype, not just integer -- CONFIRMED
            # REGRESSION (independent review, 2026-08-16): the original int-only gate let an
            # identical horizon/step-index pattern stored as float64 (e.g. after a groupby().mean()
            # or a CSV read without explicit dtype control, both of which silently upcast) sail
            # straight through and get crowned champion. The shape check itself never looked at
            # dtype; only the caller-side gate did.
            # NUMERIC-STRING COERCION ADDED 2026-09-07 (round-4 8-lens PyPI-preflight audit,
            # CONFIRMED BLOCKING). This branch used to filter candidates on
            # `pd.api.types.is_numeric_dtype(df[c])` alone, which is a DTYPE check, not a content
            # check -- a genuinely numeric model column read from a CSV without explicit dtype
            # control (or produced by a merge/reindex that upcasts to object) commonly ends up as
            # object/string dtype despite every value being a clean number, and was silently
            # EXCLUDED from model_cols with no warning, no error, nothing. Reproduced directly: a
            # 3-model frame where the genuinely best model's column was stored as numeric strings
            # dropped that model entirely, and report() printed a fully confident VERDICT over the
            # remaining two models as if nothing were missing. Now every non-structural column is
            # numeric-COERCED (`pd.to_numeric(..., errors="coerce")`) before the dtype filter,
            # so a numeric-string column is correctly recovered as a candidate; a column that is
            # genuinely non-numeric (real text metadata) still coerces to all-NaN and is excluded
            # exactly as before -- this is a pure widening of what's RECOGNIZED as numeric, not a
            # loosening of what's accepted as a model.
            candidates = []
            for c in df.columns:
                if c in structural:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    candidates.append(c)
                    continue
                # Coercion is scoped to columns pandas itself calls string-like -- a datetime64/
                # timedelta64 column is not, but `pd.to_numeric` "succeeds" on it anyway (silently
                # reinterpreting it as nanosecond-epoch integers), which is never a real forecasting
                # model and must not be swept in here. Confirmed: this guard is what actually
                # excludes a leftover `ds`-style datetime column in the nixtla CV-frame shape;
                # dtype-based exclusion alone (the pre-fix behavior) already handled it correctly,
                # this fix must not regress that.
                # WIDENED 2026-09-07 (final pre-publish pip-install check, CONFIRMED BLOCKING, two
                # passes). Originally scoped to plain OBJECT dtype only -- pandas 3.0 (the version a
                # plain `pip install` resolves to today, since this package pins no upper bound) made
                # its dedicated string dtype the DEFAULT for any string data, including a plain
                # `pd.Series(["1.5", "2.3"])`, for which `is_object_dtype` is False; the object-only
                # guard silently excluded EVERY numeric-string model column under pandas 3.0,
                # reproducing byte-for-byte the exact original bug (a numeric model column read as
                # strings vanishes from model_cols with no warning) this whole coercion branch exists
                # to fix. A first fix added `isinstance(..., pd.StringDtype)` alongside the object
                # check; independent review then found that still misses a THIRD, increasingly common
                # shape -- a pyarrow-backed string column (`pd.ArrowDtype(pa.string())`, e.g. from
                # `pd.read_csv(..., dtype_backend="pyarrow")`), which is neither object dtype nor
                # `pd.StringDtype`. `pd.api.types.is_string_dtype` -- confirmed directly, both
                # generations -- returns True for all three shapes (plain object containing strings,
                # `pd.StringDtype` under either storage backend, and `pd.ArrowDtype(pa.string())`) and
                # False for datetime64, whether native or object-boxed, so one call replaces the
                # two-part check above and closes all three gaps without reopening the datetime one.
                if not pd.api.types.is_string_dtype(df[c]):
                    continue
                as_num = pd.to_numeric(df[c], errors="coerce")
                if as_num.notna().any():
                    coerced[c] = as_num
                    candidates.append(c)
            model_cols = []
            for c in candidates:
                col = coerced.get(c, df[c])
                if pd.api.types.is_bool_dtype(col):
                    # Same reasoning as _from_wide_dataframe: a bool flag is numeric-but-not-integer,
                    # so it isn't caught by an integer-dtype gate and can only be caught by luck via
                    # the periodicity heuristic. A boolean value can never be a legitimate forecasting
                    # model's loss -- reject outright. CONFIRMED (independent 'wild' review, 2026-08-16).
                    raise ValueError(
                        f"column '{c}' is boolean, not a per-period numeric loss -- looks like a "
                        f"flag/indicator column, not a forecasting model. Pass models=[...] "
                        f"explicitly to override."
                    )
                if _looks_like_metadata_column(col.to_numpy(), len(df)):
                    raise ValueError(
                        f"column '{c}' looks like a categorical/metadata column (an exact "
                        f"arithmetic sequence or a short repeating pattern), not a forecasting "
                        f"model -- a stray metadata column must never be silently crowned a model. "
                        f"Pass models=[...] explicitly to override."
                    )
                model_cols.append(c)
            # UNPIVOTED-LONG-FRAME WARNING. ADDED 2026-09-10 (round-7 final pre-publish review,
            # real_data_dogfood lens, CONFIRMED against real project data). The single most natural
            # mistake in this documented API: calling from_forecasts() on a still-long/tidy frame
            # (one row per (period, model), with a leftover model-identity column) without first
            # pivoting to wide-by-model, and with no group= (the common case when there is no
            # separate series/group axis). Reproduced directly on
            # multiseries/results/breadth_distinct/breadth_distinct_predictions.csv: this silently
            # auto-inferred leftover numeric columns (a calendar-year column, the raw prediction
            # column, precomputed per-row error/scale columns) as "6 competing models," pooling every
            # real model's rows together within each period, and report() printed a fully confident
            # VERDICT/LEADERBOARD over nonsense columns -- zero warnings, zero errors. The existing
            # duplicate-(period,group) guard below is deliberately scoped to fire only when group is
            # NOT None (repeated period rows are a legitimate, intentional shape when there is no
            # group axis), so it structurally cannot catch this. The distinguishing signal: a
            # genuinely wide-format frame has exactly one row per period; a not-yet-pivoted long
            # frame has one row per (period, model), so rows repeat within a period. Keyed on that,
            # not on "group is None" alone, to avoid warning on the common, correct wide-format case.
            if group is None and len(df) > df[period].nunique():
                warnings.warn(
                    f"from_forecasts(): {len(df)} rows but only {df[period].nunique()} distinct "
                    f"'{period}' values -- rows repeat within a period. If your data has one row per "
                    f"(period, model) with a model-identifier column (a long/tidy frame), it needs to "
                    f"be pivoted to wide-by-model first (see README's 'Ingesting forecast frames'), "
                    f"or the auto-inferred model columns {model_cols!r} may not be the real models -- "
                    f"they could be leftover metadata/error columns silently pooled across models. "
                    f"Pass models=[...] explicitly once you've confirmed the right columns, or set "
                    f"group= if this is intentionally a multi-series/multi-group panel.",
                    UserWarning, stacklevel=3,
                )
        else:
            missing = [m for m in models if m not in df.columns]
            if missing:
                raise ValueError(f"models {missing} not found in the frame.")
            model_cols = list(models)
        if len(model_cols) == 0:
            raise ValueError("zero model columns found (after excluding y_true/period/group and any "
                              "non-numeric columns); pass models=[...] explicitly.")
        _check_no_str_collision(model_cols)

        if isinstance(metric, str):
            if metric not in ("mae", "mse", "rmse", "mape"):
                raise ValueError(f"unrecognized metric '{metric}'; use one of 'mae', 'mse', 'rmse', "
                                  f"'mape', or pass a callable(y_true, y_pred).")
        elif not callable(metric):
            raise ValueError(f"metric must be a string ('mae'/'mse'/'rmse'/'mape') or a callable "
                              f"(y_true, y_pred) -> per-row loss; got {type(metric).__name__}.")

        y = df[y_true].to_numpy(float)
        if metric == "mape" and np.any(y == 0):
            raise ValueError("MAPE is undefined when y_true is 0 for one or more rows; use a different "
                              "metric, or drop/impute those rows first.")

        # DUPLICATE (group, period) ROW CHECK ADDED 2026-09-07 (round-4 8-lens PyPI-preflight audit,
        # CONFIRMED HIGH; NARROWED after an existing test caught an over-broad first version). README
        # documents the group-based ingestion contract explicitly as "one row per (series, period)"
        # -- when `group` is given, a (group, period) combination appearing more than once is not a
        # supported shape, it's a data-quality bug (a duplicated CV fold, a fan-out join). Previously
        # this was invisible: `group` is validated as a column but never used as part of the
        # aggregation key, so a duplicate row was silently POOLED into its period's mean alongside
        # the real row, and the existing "fewer non-missing rows than the period total" coverage
        # warning could never fire (both rows inflate the same total together, so nothing ever looks
        # under-covered). Reproduced directly: duplicating one row with an outlier value flipped a
        # panel from "identified" to "not identified" with zero warning of any kind.
        #
        # SCOPED TO group-not-None ONLY: an earlier version of this check also fired when group=None,
        # duplicating on period alone -- but TestForecastMetrics::test_from_forecasts_partial_nan_row_
        # warns (and the function's own design) legitimately builds MULTIPLE rows per period with NO
        # group column at all (repeated/pooled observations within a period is a real, tested,
        # supported shape when there is no group axis to key on) -- that test's fixture rows are even
        # byte-identical by construction, which rules out an exact-row-duplicate heuristic too. There
        # is no way to distinguish "an accidental duplicate" from "an intentional repeated
        # observation" by row content alone when group=None, so this check only applies where the
        # README's own documented contract actually establishes (group, period) uniqueness.
        if group:
            dup_mask = df.duplicated(subset=[period, group], keep=False)
            if dup_mask.any():
                dup_keys_seen = df.loc[dup_mask, [period, group]].drop_duplicates()
                raise ValueError(
                    f"from_forecasts(): {int(dup_mask.sum())} row(s) share a duplicate (period, "
                    f"group) value -> {dup_keys_seen.to_dict('records')[:5]}"
                    f"{' (+more)' if len(dup_keys_seen) > 5 else ''} -- this function's documented "
                    f"input contract (when group= is given) is one row per (series, period). A "
                    f"duplicate is almost always a data bug (a re-appended CV fold, a fan-out join) "
                    f"that would otherwise be silently pooled into that period's mean with no "
                    f"warning. Drop the duplicate(s) or fix the upstream join before building a "
                    f"LossPanel."
                )

        period_order = None
        period_total_rows = df.groupby(df[period].to_numpy()).size()
        losses = {}
        for m in model_cols:
            yhat = coerced[m].to_numpy(float) if m in coerced else df[m].to_numpy(float)
            if metric == "mae":
                e = np.abs(y - yhat)
            elif metric == "mse":
                e = (y - yhat) ** 2
            elif metric == "rmse":
                e = (y - yhat) ** 2                # sqrt applied after period aggregation, below
            elif metric == "mape":
                e = np.abs(y - yhat) / np.abs(y)
            else:                                   # callable
                e = np.asarray(metric(y, yhat), float)

            tmp = pd.DataFrame({"_period": df[period].to_numpy(), "_e": e})
            # PARTIAL-NaN COVERAGE WARNING ADDED 2026-08-26 (round-2 review, "does it deliver on its
            # promises" lens): a `.groupby().mean()` skips NaN rows by default (skipna=True), so a
            # model missing a forecast for SOME rows within a period (while other rows in that same
            # period are present) never produces a NaN loss value -- it just silently averages over
            # fewer rows than a fully-covered model in the same period, i.e. a different effective
            # sample size per model per period, with nothing downstream able to tell. Verified
            # directly: model_a with 10/10 valid rows and model_b with 7/10 valid rows in the same
            # period built a "clean" panel with no error and no warning, since the NaN/inf check below
            # only ever sees the already-averaged (NaN-free) per-period means. Compare each model's
            # non-null row count against the period's TOTAL row count in the frame (not against other
            # models', so this fires even for a single affected model) and warn, naming the model and
            # the specific under-covered periods, rather than silently changing what "this model's
            # period mean" means without telling the caller.
            per_period_n = tmp.groupby("_period")["_e"].count()
            short = per_period_n[per_period_n < period_total_rows.reindex(per_period_n.index)]
            if len(short):
                warnings.warn(
                    f"from_forecasts(): model '{m}' has fewer non-missing rows than the period total "
                    f"at period(s) {list(short.index)} -- its period mean there is being averaged "
                    f"over only {list(short.astype(int))} of {list(period_total_rows.reindex(short.index).astype(int))} "
                    f"row(s), a smaller effective sample than a fully-covered model in the same "
                    f"period gets. This usually means '{m}' is missing a forecast for some rows; "
                    f"consider dropping or imputing them before building a LossPanel."
                )
            per_period = tmp.groupby("_period")["_e"].mean()
            if metric == "rmse":
                per_period = np.sqrt(per_period)
            if period_order is None:
                period_order = per_period.index
            else:
                per_period = per_period.reindex(period_order)
            losses[str(m)] = per_period.to_numpy(float)

        labels = list(period_order)
        T = len(labels)
        # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-16): from_losses() raises
        # immediately on any NaN/inf loss value ("a single NaN silently poisons the pooled mean and
        # can crown a spurious winner"), but from_forecasts() had NO equivalent check -- a model
        # missing a forecast for one period, or a metric like MAPE dividing by a near-zero actual,
        # produces a NaN loss here (e.g. via the reindex-to-period_order step for a model whose
        # groupby doesn't cover every period) that sailed straight into a "validated" LossPanel.
        # `__repr__`'s blanket except-Exception fallback meant `print(panel)` showed nothing wrong;
        # the failure only surfaced later, downstream, with a misleading "before computing a Model
        # Confidence Set" message that never mentioned from_forecasts() as the actual source.
        for m, arr in losses.items():
            if not np.all(np.isfinite(arr)):
                bad = [str(labels[i]) for i in range(T) if not np.isfinite(arr[i])]
                raise ValueError(f"model '{m}' has NaN/inf loss values at period(s) {bad}; this "
                                  f"usually means it is missing a forecast for some periods, or the "
                                  f"metric divided by ~0 (e.g. MAPE with a near-zero actual) -- drop "
                                  f"or impute those periods before building a LossPanel.")
            # Same magnitude sanity check as from_losses() -- see its comment for the full story.
            # A computed loss (e.g. MSE on an unscaled raw-dollar y_true/y_pred pair) can reach this
            # magnitude even when neither NaN nor inf individually.
            if np.any(np.abs(arr) > _MAX_ABS_LOSS):
                raise ValueError(f"model '{m}' has a computed loss magnitude above {_MAX_ABS_LOSS:.0e}, far beyond "
                                  f"any real per-period loss metric -- arithmetic at this scale silently "
                                  f"overflows float64 downstream. Check for a units/scaling bug in "
                                  f"y_true/y_pred, or use a scale-invariant metric.")
        _check_K_vs_T(len(losses), T)

        if weights is None:
            w = np.ones(T)
        elif isinstance(weights, str) and weights == "count":
            # ROUTED THROUGH _validate_weights, FIXED 2026-08-17 (own-review pass, fresh read): this
            # was the only weight-construction branch in the module that skipped `_validate_weights`.
            # Safe TODAY only by an implicit invariant nothing enforces -- `period_order` is built
            # from the same groupby(period) as `cnt`, so every count is provably >=1 and never NaN --
            # but a future change to how `period_order`/`cnt` are computed could silently break that
            # guarantee with no check left to catch it. Costs nothing to validate explicitly instead
            # of relying on today's implementation happening to be safe.
            cnt = df.groupby(period).size().reindex(period_order)
            w = _validate_weights(cnt.to_numpy(float), T)
        elif isinstance(weights, str):
            if weights not in df.columns:
                raise ValueError(f"weights column '{weights}' not found in the frame.")
            wcol = df.groupby(period)[weights].mean().reindex(period_order)
            w = _validate_weights(wcol.to_numpy(float), T)
        else:
            w = _validate_weights(weights, T)

        return cls(losses, w, labels, False, _validated=True)

    # ------------------------------------------------------------------- save/load ----
    def save(self, path):
        """Serialize to JSON. Labels are stringified (`str(label)`) since JSON has no native
        Timestamp/date type -- a panel built with Timestamp labels loads back with str labels of the
        same text, not the original type. `compare()` accounts for this by comparing str(label) on
        both sides; anything else comparing raw label objects across a save/load boundary must do
        the same.

        ATOMIC WRITE, FIXED 2026-08-27 (round-8 concurrency-safety review): writing directly to
        `path` left a window where a concurrent reader could open a partially-written file and hit
        a raw JSONDecodeError -- confirmed reproducibly (299 of ~800 concurrent reads corrupted in
        a stress test with 3 writer + 5 reader threads on the same path). Fixed with the standard
        write-to-temp-then-`os.replace` pattern: the temp file lives in the SAME directory as
        `path` (so the replace is on the same filesystem, required for atomicity, and works
        identically on POSIX and Windows -- relevant now that this package runs real cross-platform
        CI) and is only renamed into place once fully written and flushed. A concurrent reader
        therefore always sees either the complete old file or the complete new one, never a partial
        write."""
        payload = {
            "losses": {m: list(map(float, v)) for m, v in self.losses.items()},
            "weights": list(map(float, self.weights)),
            "labels": [str(l) for l in self.labels],
            "labels_are_positional": bool(self.labels_are_positional),
        }
        directory = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".losspanel-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path):
        """Deserialize from JSON written by `.save()`. Labels come back as `str` regardless of what
        type they were saved from (see `.save()`'s docstring) -- a round trip is not type-preserving
        for labels, only value-preserving as text."""
        # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-17, security/hostile-input
        # lens): `load()` used to call the bare `cls(...)` constructor directly with `_validated=True`,
        # bypassing EVERY check `.from_losses()` enforces -- negative weights (verified directly: a
        # strictly-dominated model won pooled_winner() outright, its weighted mean going NEGATIVE, an
        # impossible value for a real loss), all-zero weights (loaded silently, crashed downstream
        # with an uncaught ZeroDivisionError instead of a clear error at load time), mismatched
        # labels length (too short crashed with a bare IndexError; too long silently printed a WRONG
        # period count with no error at all), and bare JSON `NaN`/`Infinity` tokens (Python's json
        # module accepts these non-standard literals by default; `__repr__`'s blanket except-Exception
        # then hid the corruption behind an innocuous-looking printout). A file this permissive to
        # load is exactly the kind of "already validated" object the rest of the package trusts
        # everywhere downstream with no re-checking. Routing through `.from_losses()` closes all of
        # this at once by construction, since a save()/load() round trip is just another way of
        # constructing a panel and deserves the exact same guarantees as building one fresh.
        with open(path, encoding="utf-8") as f:
            def _reject_json_extensions(token):
                raise ValueError(f"malformed LossPanel file: bare '{token}' is not valid JSON (Python's "
                                  f"json module accepts it as a non-standard extension, but a NaN/Infinity "
                                  f"loss or weight must never load silently) -- the file at '{path}' is "
                                  f"either corrupted or was not produced by LossPanel.save().")
            payload = json.load(f, parse_constant=_reject_json_extensions)
        # FIXED 2026-09-02 (10-agent code-review pass, CONFIRMED UX GAP): a well-formed JSON file
        # that simply isn't a LossPanel (wrong file, export from something else) used to raise a
        # bare KeyError('losses') here -- str(e) is just "'losses'", giving no indication of what
        # is wrong or why. Every other malformed-input case above already gets a message naming
        # LossPanel; this closes the last gap in that sweep.
        missing = [k for k in ("losses", "weights", "labels") if k not in payload]
        if missing:
            raise ValueError(f"'{path}' is not a valid LossPanel file: missing required key(s) "
                              f"{missing} (a LossPanel JSON file must have been produced by "
                              f".save() -- got top-level keys {sorted(payload.keys())} instead).")
        panel = cls.from_losses(payload["losses"], weights=payload["weights"], labels=payload["labels"])
        panel.labels_are_positional = bool(payload.get("labels_are_positional", False))
        return panel
