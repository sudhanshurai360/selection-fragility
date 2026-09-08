"""
selection_fragility.identify — `identified` / `mcs_size`, delegating to `arch.bootstrap.MCS`.

Replaces the hand-rolled elimination loop in `mcs.py` for the new v1.0 surface: `arch` is faster at
matched replication counts (no fixed multiplier stated here -- a timing comparison is environment-
and version-dependent and will drift; benchmark it yourself if the number matters, don't trust a
number written in this docstring) and removes a standing cross-validation obligation
(this project maintaining its own re-implementation of a well-established published procedure).
Verified against arch directly on both guards `mcs.py` was written for -- deterministic dominance and
T near the block length (see tests/test_stage1_identified.py::TestArchGuardsDirect).

`mcs.py` / `model_confidence_set()` is kept as-is, not removed in this pass -- it is exercised by
`tests/test_selection_fragility.py`'s pre-existing, still-passing suite and by
tests/test_stage1_identified.py's structural-property tests (nesting, permutation-invariance, ...),
which serve as PRE-REGISTERED ground truth for this wrapper. `identified`/`mcs_size` are the new,
recommended entry point; `model_confidence_set` remains available for anyone who needs the raw
survivor list plus stopping p-value.
"""
import hashlib
import warnings

import numpy as np
import pandas as pd
from arch.bootstrap import MCS as _ArchMCS

from .fragility import _MAX_ABS_LOSS, _unwrap_panel


def _validate_alpha(alpha):
    # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-16): a very plausible practitioner
    # slip -- passing alpha=10 meaning "10%" instead of 0.10 -- silently produced mcs_size=0 (an
    # impossible result: a Model Confidence Set can never be empty by definition) and a report()
    # that printed a self-contradicting explanation ("MCS size > 1" stated as the reason for
    # 'not identified' while the actual size was 0). Nowhere in this otherwise fanatically-validated
    # package (weights, K vs T, NaN, falsy `a`) was alpha ever checked to be in (0, 1).
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1 (it is a significance level, e.g. "
                          f"0.05 or 0.10, not a percentage); got {alpha}.")


def _to_frame(L):
    if len(L) < 2:
        raise ValueError(f"need at least 2 models to compute a Model Confidence Set; got {len(L)}.")
    # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-16): the internal tie-breaking jitter
    # below is drawn from a FIXED, position-seeded RNG and applied positionally to whatever column
    # order `list(L)` happens to be in (plain dict insertion order). For two models tied to within
    # ~1e-13 relative, the exact SAME {model: array} mapping -- same names, same values, only
    # insertion order changed -- produced a different MCS survivor set (91.5% disagreement rate at
    # K=6, T=24 with one manufactured near-tie). Sorting columns by name here makes the frame handed
    # to arch (and therefore the jitter draw) depend only on model IDENTITY, never incidental
    # insertion order.
    models = sorted(L)
    T = None
    cols = {}
    for m in models:
        a = np.asarray(L[m], float)
        if a.ndim != 1 or a.size == 0:
            raise ValueError(f"model '{m}' loss must be a non-empty 1-D array; got shape {a.shape}.")
        if T is None:
            T = a.size
            if T < 2:
                raise ValueError(f"need at least 2 periods to compute a Model Confidence Set; got {T}.")
        elif a.size != T:
            raise ValueError(f"all models must share the same number of periods; "
                              f"'{m}' has {a.size}, expected {T}.")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"model '{m}' has NaN/inf loss values; drop or impute those periods "
                              f"before computing a Model Confidence Set.")
        # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-17, numerical-extremes lens):
        # loss magnitudes beyond ~1e150-1e200 make `arch.bootstrap.MCS` internally overflow/underflow
        # (squaring, dividing) and raise the SAME IndexError this module's own degenerate-tie
        # fallback below is written to catch -- conflating a genuine exact tie with an arithmetic
        # overflow, and reporting "everyone survives" for data that may not be tied at all. See
        # fragility.py's `_MAX_ABS_LOSS` for the full story; same 1e100 floor, applied here since
        # this path (direct `identified()`/`mcs_size()` calls) validates independently of
        # fragility.py's `_validate_losses` and panel.py's LossPanel checks.
        if np.any(np.abs(a) > _MAX_ABS_LOSS):
            raise ValueError(f"model '{m}' has a loss magnitude above {_MAX_ABS_LOSS:.0e}, far beyond any real "
                              f"per-period loss metric -- arithmetic at this scale silently overflows "
                              f"the bootstrap variance calculation. Check for a units/scaling bug upstream.")
        cols[m] = a
    return pd.DataFrame(cols)


class _AllIncluded:
    """Degenerate-tie result: every model included, none excluded."""
    def __init__(self, models):
        self.included = list(models)


def _validate_mcs_weights(w):
    # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-16): `_run_mcs`/`mcs_size`/`identified`
    # accepted no `w` at all, so callers that themselves receive weights (`report()`, `resolution_report()`,
    # `compare()`) silently computed the Model Confidence Set UNWEIGHTED while the panel's pooled champion
    # (`pooled_winner(L, w)`) WAS computed with them -- observed directly producing a self-contradictory
    # report(): a model shown as "(champion)" on the weighted leaderboard while simultaneously "excluded"
    # from a supposedly-identified MCS that never saw the weights that made it champion.
    # `arch.bootstrap.MCS` has no native period-weight parameter, and there is no quick, statistically
    # sound way to bolt one on tonight (naive row-repetition would corrupt the circular block bootstrap's
    # inclusion-probability guarantees documented in fragility.py's `_block_resample_idx`). Per this
    # package's own R1/R2 discipline ("refuse rather than print a confidently-looking number when the
    # data cannot support it"), a genuinely non-uniform weight vector is refused outright rather than
    # silently dropped -- uniform weights (or None) are unaffected and behave exactly as before.
    if w is None:
        return
    wa = np.asarray(w, float)
    if wa.size and not np.allclose(wa, wa[0]):
        raise ValueError(
            "arch.bootstrap.MCS has no native support for non-uniform period weights, so the Model "
            "Confidence Set cannot be silently computed unweighted while the panel's pooled champion "
            "is computed WITH these weights -- that mismatch is exactly what previously produced a "
            "model reported as both '(champion)' and 'excluded from the MCS' in the same report. Pass "
            "uniform weights (or w=None) to compute an MCS, or drop weighting for this call."
        )


def _col_seed(a: np.ndarray) -> int:
    """CONFIRMED REGRESSION (Phase-2 holistic review, finding A7, 2026-08-27). Deterministic (NOT
    affected by PYTHONHASHSEED -- Python's builtin hash() randomizes bytes/str per process, sha256
    does not) seed derived from a column's OWN data content, used below so the per-cell jitter
    draw depends only on "what values does this specific model's loss array contain" -- never on
    that model's name or its position in the (name-sorted) column order.

    Two bugs this closes, both verified by direct repro before this fix:
      1. A byte-identical PAIR inside a larger, otherwise-distinct panel could get OPPOSITE MCS
         verdicts (one included, one excluded) -- reproduced directly: 16/200 trials on a 5-model
         panel with one byte-identical pair. Root cause: the old jitter was ONE shared
         `rng.normal(size=M.shape)` draw filled positionally across the whole matrix, so two
         IDENTICAL columns still received two DIFFERENT (independent) noise draws -- enough for
         arch's bootstrap to manufacture a fake significant difference between models that are, by
         construction, computationally indistinguishable. Seeding each column's jitter from that
         column's own data means byte-identical columns now get the IDENTICAL jitter draw, so they
         stay byte-identical after jittering and arch correctly treats them as tied with EACH
         OTHER (both included or both excluded together, never split).
      2. Rename-dependence on near-tied panels, DESPITE `_to_frame`'s existing `sorted(L)` fix --
         that fix only guarantees INSERTION-order invariance for a FIXED set of names; renaming a
         model changes its ALPHABETICAL rank, which changes its POSITION in the sorted column
         order, which changed which positional slice of the old shared jitter draw it received.
         Reproduced directly: 186/200 trials disagreed on a 6-model, T=24 panel with one
         manufactured ~1e-13-relative near-tie, renaming a single model (same data, new name).
         Seeding by data content instead of position/name makes the draw genuinely
         identity-invariant: a renamed model with unchanged data gets the unchanged jitter, and
         therefore the unchanged verdict.

    Byte-identical, not value-identical (third-agent review, 2026-08-27): +0.0 and -0.0 compare
    equal but have different byte patterns, so a column containing an exact-zero loss with a
    different internal sign bit than an otherwise-identical column would get a different seed.
    Exotic (needs an exact-zero loss value with differing float sign, e.g. from different
    computation paths) and does not affect any of the repro cases above; noted, not fixed.
    """
    digest = hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).digest()
    return int.from_bytes(digest[:8], "big")


def _run_mcs(L, alpha=0.10, reps=500, block_size=3, seed=0, w=None):
    _validate_alpha(alpha)
    _validate_mcs_weights(w)
    if int(reps) < 1:
        raise ValueError(f"_run_mcs(): reps must be a positive integer; got {reps}. A non-positive "
                          f"reps ran zero bootstrap replications and returned an all-included set "
                          f"with no resampling ever performed.")
    if int(block_size) < 1:
        raise ValueError(f"_run_mcs(): block_size must be a positive integer; got {block_size}.")
    df = _to_frame(L)
    models = list(df.columns)
    # arch's MCS (both method='R', the default, and method='max') divides each pairwise mean-loss
    # difference by its OWN bootstrap variance; when every model's per-period losses are BYTE-IDENTICAL
    # that variance is exactly zero for every off-diagonal pair (arch's `+= eye(k)` guard only touches
    # the diagonal), giving a 0/0 = NaN test statistic. Verified directly against arch on this
    # package's byte-identical-triple fixture: 'R' crashes with an IndexError inside its elimination
    # loop, 'max' hangs. No model differs from any other in this case, so there is no basis on which
    # elimination could exclude anyone -- short-circuit rather than feed arch degenerate input.
    M = df.to_numpy()
    if np.all(M == M[:, [0]]):
        return _AllIncluded(models)
    # A SECOND, DIFFERENT arch degenerate-tie bug (the byte-identical guard above covers only the
    # first): ANY pair of models with an EXACTLY EQUAL weighted mean -- not just byte-identical
    # arrays -- crashes arch's own `_format_pvalues` with "arrays used as indices must be of integer
    # or boolean type" (verified directly against `exact_tie_panel`, means equal at 0.8525, arrays
    # otherwise different). An EARLIER version of this function caught that IndexError and fell back
    # to "everyone survives" -- correct only when the tie spans every model still under comparison.
    # CONFIRMED WRONG (independent review, 2026-08-16) for a PARTIAL tie: K=4 with m0/m1 exactly tied
    # and m2/m3 decisively, un-ambiguously worse (means 0.99/0.99/2.06/6.00) still crashes arch, and
    # the blanket fallback returned all 4 models as "included" when the true MCS is {m0, m1} -- a
    # silent, large overstatement of the tied set. Fixed by breaking any exact tie with an
    # infinitesimal jitter BEFORE handing the data to arch, so arch's own elimination logic runs to
    # completion and correctly excludes the genuinely worse models instead of crashing.
    #
    # THE JITTER MUST BE PER-CELL RANDOM NOISE, NOT A PER-COLUMN CONSTANT OFFSET -- caught by testing
    # this fix against its own repro case: a first version added a deterministic constant per column
    # (`np.arange(K) * eps`), which for two BYTE-IDENTICAL tied columns turns their difference into an
    # exactly-constant, zero-variance series -- precisely arch's "deterministic dominance -> certain
    # rejection" case, so the jitter itself manufactured a fake significant difference out of an
    # infinitesimal, meaningless offset (verified: this wrongly excluded one of the two genuinely tied
    # models). Independent per-cell noise breaks the exact tie in floating-point terms while leaving
    # the DIFFERENCE between formerly-tied columns as a small-variance random series (not a constant),
    # so arch's bootstrap correctly measures a near-zero t-statistic and keeps both. Seeded internally
    # and fixed (not the caller's `seed`) so results stay reproducible across calls.
    #
    # SCALE TIGHTENED 2026-08-16 (independent review): this was originally 1e-9 relative, three orders
    # of magnitude ABOVE `fragility.py`'s own `_MARGIN_REL_FLOOR = 1e-12` -- the package's own stated
    # boundary for "this margin is real, not floating-point noise". Verified directly: at 1e-9 the
    # jitter swamped a constructed deterministic edge of exactly 1e-10 relative (which `fragility.py`'s
    # own floor treats as real, since 1e-10 > 1e-12), flipping `identified()` from the correct True to
    # a wrong False. Tightened to 1e-13 -- verified to still reliably break arch's exact-tie crash on
    # both the byte-identical-pair and partial-tie repro cases -- so the jitter is now safely BELOW,
    # not above, the package's own real-margin floor, resolving the inconsistency. Still 3+ orders of
    # magnitude above float64 machine epsilon (~2.2e-16), so it remains numerically well-behaved.
    #
    # SCALE MADE PER-COLUMN, NOT PANEL-WIDE (CONFIRMED REGRESSION, 2026-08-26, reproduced directly):
    # a single `float(np.mean(np.abs(M)))` over the WHOLE panel let one model with a legitimately
    # large-but-valid magnitude (e.g. ~1e10-1e15, far under the 1e100 `_MAX_ABS_LOSS` floor -- a
    # plausible mixed-units mistake, raw-dollar loss alongside a normalized metric) inflate the jitter
    # std applied to EVERY column, including two unrelated, small-scale models with a real, decisive,
    # resolvable edge between them. Verified directly: with one such large-magnitude model in the
    # panel, the injected per-cell jitter std (tens, from a panel mean in the 1e14-1e15 range) swamped
    # a genuine ~3e-5 edge between two ~1.0-scale models (whose own noise std was ~1e-6) so completely
    # that the reported MCS survivor was decided by the FIXED internal jitter draw, not by the actual
    # data -- swapping which of the two small models was truly better (while holding the jitter seed
    # fixed, as this code always does) did NOT change the reported survivor. This is not a narrow
    # tie-break corruption; every MCS call on a panel spanning multiple magnitude scales was affected,
    # since the jitter above is applied unconditionally to every call, not only on an arch crash.
    # Fixed by computing jitter std PER COLUMN, from that column's own mean magnitude -- so a model at
    # one scale can no longer contaminate the tie-break precision of models at a different scale
    # sharing the same panel. Each column's own jitter remains well below its own real-margin floor,
    # exactly as verified above, now per-model rather than panel-wide.
    col_scale = np.abs(M).mean(axis=0)
    col_scale = np.where(col_scale == 0, 1.0, col_scale)
    # FIXED 2026-08-27 (finding A7): was ONE shared `rng.normal(size=M.shape)` draw, filled
    # positionally -- see _col_seed's docstring for the two bugs that caused (byte-identical-pair
    # split verdicts; rename-dependence surviving the earlier sort-by-name fix). Each column now
    # draws its own jitter from an RNG seeded by that column's own data content, so identical
    # columns get identical jitter (stay tied with each other) and a renamed-but-data-unchanged
    # model gets the unchanged jitter (stays rename-invariant).
    jitter = np.column_stack([
        np.random.default_rng(_col_seed(M[:, j])).normal(0.0, 1.0, size=M.shape[0])
        for j in range(M.shape[1])
    ]) * (col_scale[np.newaxis, :] * 1e-13)
    dfj = pd.DataFrame(M + jitter, columns=df.columns)
    # BLOCK-VS-T GUARD. FIXED 2026-09-02 (round-2 10-agent review, CONFIRMED SEVERE). `mcs.py:161`
    # shrinks the block when `T <= block` ("so the bootstrap retains power") and
    # `fragility.winner_stability` RAISES on `block >= T` (a block that long "only ever produces a
    # rotation of the same T periods"). `_run_mcs` -- the recommended engine, and the one `report()`
    # uses -- had neither, and `block_size` is an unvalidated public kwarg. Measured false
    # "point-identified" rate on PURE NOISE (K=3 iid, alpha=0.10, 120 trials/cell): T=3/block=3
    # (THE DEFAULT) 0.650; T=4/block=6 0.908; T=6/block=6 0.983 -- against 0.03-0.17 at block=1.
    # So `report()` on a legal T=3 panel printed "VERDICT: identified" on noise about two thirds of
    # the time. Same remedy as mcs.py's, so the two engines now agree.
    T_eff = M.shape[0]
    if T_eff <= block_size:
        block_eff = max(1, T_eff // 2)
        warnings.warn(f"identify: T={T_eff} <= block_size={block_size}; shrinking block to "
                      f"{block_eff} so the bootstrap retains power (a block >= T only ever "
                      f"produces a rotation of the same T periods, which manufactures spurious "
                      f"point-identification).")
        block_size = block_eff
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # arch's own internal convergence chatter, not ours to surface
        m = _ArchMCS(dfj, size=alpha, reps=reps, block_size=block_size, bootstrap="circular", seed=seed)
        try:
            m.compute()
        except IndexError:
            # Even after jittering, arch failed -- e.g. several models simultaneously tied for the
            # SAME value so the jitter alone didn't fully resolve every comparison. The honest last
            # resort is "everyone survives"; this should now be rare (essentially only the
            # all-tied case, already caught above, or higher-order degeneracies).
            return _AllIncluded(models)
    return m


def mcs_size(L, alpha=0.10, reps=500, block_size=3, seed=0, w=None):
    """Size of the Model Confidence Set at level `alpha`: the count of models statistically
    indistinguishable from the best (Hansen, Lunde & Nason 2011), via `arch.bootstrap.MCS` with
    `bootstrap='circular'` passed explicitly -- arch's own default is 'stationary', and comparing
    against the wrong bootstrap silently understates the cross-check (the exact defect fixed
    pipeline-side, task #64). |MCS| == 1 => point-identified; > 1 => not identified.

    `w`, if given, must be uniform (or None) -- see `_validate_mcs_weights` for why a genuinely
    non-uniform weight vector is refused rather than silently ignored.

    TWO INDEPENDENT MCS ENGINES, NOT ONE TUNED TWO WAYS (round-6 foundation-model-researcher
    review, 2026-08-27). This function (and `report()`, which calls it internally) delegates to
    `arch.bootstrap.MCS` -- a genuinely different implementation from this package's own native
    `mcs.model_confidence_set()`, kept deliberately separate as an independent cross-check, not
    a second copy of the same algorithm. Their default bootstrap replication counts differ
    (`reps=500` here vs. `B=2000` there) because they're different engines, not an oversight --
    aligning them would undermine the point of having an independent check and risks moving
    already-verified numbers. The two DO agree once reps/B are matched explicitly (confirmed
    directly on a highly-correlated panel: default-vs-default disagreed on |MCS|, reps=500/
    B=500 agreed exactly). If you call both on the SAME panel and expect the SAME identification
    verdict, pass matching reps=/B= to each -- don't compare their un-matched defaults and treat
    a difference as a disagreement about the data.

    Accepts a `LossPanel` directly, matching every other stage of the README's staged API (FIXED
    2026-09-07, round-4 8-lens PyPI-preflight audit, CONFIRMED HIGH): following the README's own
    Stage 0 -> Stage 1 example literally (`panel = LossPanel.from_losses(data)` then
    `identified(panel)`) used to raise a raw internal `TypeError: object of type 'LossPanel' has no
    len()` three stack frames deep inside `_to_frame`, with no guidance -- every other stage's
    function already accepted a `LossPanel` (this exact bug class was already found and fixed in
    `resolution_report()` on 2026-08-26; it just never got applied to these two, the actual Stage 1
    functions themselves). Uses the same shared `_unwrap_panel` helper `report()`/`compare()`/
    `resolution_report()`/the raw-array tier already use, not a new, separately-maintained check."""
    L, w = _unwrap_panel(L, w)
    m = _run_mcs(L, alpha=alpha, reps=reps, block_size=block_size, seed=seed, w=w)
    return len(m.included)


def identified(L, alpha=0.10, reps=500, block_size=3, seed=0, w=None):
    """True iff the Model Confidence Set at level `alpha` contains exactly one model -- a FACT about
    the data at this (alpha, T), not a threshold call to be tuned. See `mcs_size` for the survivor
    count when |MCS| > 1, and for the `w` contract."""
    return bool(mcs_size(L, alpha=alpha, reps=reps, block_size=block_size, seed=seed, w=w) == 1)
