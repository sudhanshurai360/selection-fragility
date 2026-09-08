"""selection_fragility.mcs — the Model Confidence Set (Hansen, Lunde & Nason 2011), the identification companion to k*.

Given a T x K matrix of per-period losses (T periods, K models), the MCS returns the set of models that are
statistically INDISTINGUISHABLE from the best at level alpha. |MCS| = 1 means the best model is point-identified;
|MCS| > 1 means selection is not identified -- a set of models is tied, and which one you "crown" is not
supported by the data. k* then measures how fragile a crowned winner is; the MCS says whether a unique winner
exists at all.

Range statistic T_R = max_{i,j} |dbar_i - dbar_j| / se_ij, with se from a CIRCULAR block bootstrap over periods
(Politis-Romano; see _block_idx); iteratively eliminate the worst-mean-loss model until the equal-predictive-
ability null is not rejected at alpha. The Monte-Carlo p-value uses the standard (b+1)/(B+1) form (Davison &
Hinkley 1997).

Caveats
--------
LIMITED POWER AT SMALL EDGES. Under the DEFAULT studentized elimination rule, the equal-predictive-ability
test detected true edges of 2%, 5% and 10% at rates of 0.19, 0.50 and 0.68, and a 30% edge 77% of the time --
on 13 shock-prone macro series with 25-36 evaluation periods each (median 30). False discovery on a genuinely
null panel is 0.007. Part of the small-edge shortfall is arithmetic rather than resolution: an edge handed to a
model already well behind does not make it best, and restricted to trials where the edge does make the focal
model the pooled best, detection at 10% is 0.87. The rest is shock-year variance, not panel length.

THESE FIGURES REPLACE an earlier docstring reporting 0.00 at every edge up to 10%, which came from a superseded
design that re-centred all models to an equal pooled mean and made one high-variance benchmark uneliminable.
That docstring also quoted raw-rule detection rates of 19%/35%/45%; no producer in this project computes them
and they must not be relied on.

At small T the MCS will frequently return |MCS| > 1, so a large tied set can reflect LOW POWER as much as a
genuine tie. Do NOT read |MCS| > 1 as a well-powered "these models are equivalent" -- read it as "cannot be
distinguished at this sample size", and report an MCS block-length sweep rather than a single number.

- The elimination step defaults to Hansen-Lunde-Nason's published e_max statistic (studentized deviation from
  the survivor-set average), not the raw-mean simplification used before 2026-07-27. Switching between the two
  rules changes NOTHING on the accompanying study's 13 curated series -- both give identical tied sets on every
  series -- so the simplification was never load-bearing there, but pass `elimination="raw"` to recover the old
  rule if you need it. Either way, cardinality is cross-checked against an independent MCS implementation
  (`arch`): the identified/non-identified verdict agrees on all 13 series in the accompanying study; exact
  tied-set cardinality agrees on 11 of 13. Cross-check against an independent MCS implementation for your own
  headline claims too."""
import numpy as np

from .fragility import _unwrap_panel

# Below this many evaluation periods, a zero bootstrap standard error is a small-sample artifact
# rather than evidence of a deterministic gap. See the se<=0 branch below.
MIN_T_FOR_ZERO_SE = 8
# Relative floor below which a bootstrap standard error is arithmetic noise rather than a measurement.
# See the long note at its use site. Mirrored in the pipeline copy, code/pipeline/mcs_test.py.
_SE_REL_FLOOR = 1e-12
# Elimination rule: "studentized" is Hansen-Lunde-Nason's published e_max; "raw" is the pre-2026-07-27
# simplification. Module-level so it is visible and overridable without threading a new kwarg everywhere.
ELIMINATION = "studentized"
# Magnitude cap, matching LossPanel.from_losses()'s own validation threshold. Without this, a loss value near or
# beyond this scale overflows float64 arithmetic INSIDE the bootstrap se computation (squaring a ~1e200 deviation
# produces 1e400, which overflows to `inf`, not `nan`). An infinite se is NOT caught by the existing degenerate-se
# guard below (that guard only fires when se collapses toward zero) -- so `t_obs = obs_diff / se` becomes a huge
# real gap divided by infinity, i.e. exactly 0.0: the pathological model is silently read as statistically
# INDISTINGUISHABLE from everyone else, and whether it then survives in the returned MCS depends on elimination
# order (column order / dict insertion order), not on the data. Confirmed by direct reproduction: the same 3-model
# panel (one deliberately astronomical loss column) returns a different survivor set under 3 different column
# orderings, including the astronomical model surviving in 2 of 3. Rejecting the input outright, at the same
# threshold LossPanel already enforces, is cheaper and more honest than trying to make the arithmetic robust to
# arbitrary magnitude.
_MAX_ABS_LOSS = 1e100
# Bootstrap resample count ceiling. mcs()'s cost scales with T*B; nothing previously bounded B, so a
# small, innocuous-looking config value (B=1_000_000) could run unbounded for tens of seconds or more
# even on a small panel -- found in round-5's adversarial security review. 100,000 is 50x the
# documented default (2000) and comfortably above any legitimate precision need (Monte-Carlo p-value
# resolution beyond B=100,000 is far finer than the alpha thresholds this package operates at), while
# still rejecting the clearly-pathological end of the range.
_MAX_BOOTSTRAP_B = 100_000
def _block_idx(T, block, rng, circular=True):
    """Block-bootstrap indices over evaluation periods, CIRCULAR by default (Politis & Romano 1992).
    Why circular rather than moving-block. The moving-block scheme draws a start uniformly from 0..T-block, so an
    observation near either END of the sample is reachable by only the few blocks that overlap it, while an
    interior observation belongs to `block` distinct windows. Inclusion probabilities are therefore markedly
    non-uniform: measured at T=38, block=3, the first and last observations are resampled at ~0.34x the rate of an
    interior one -- a 3.2x spread, widening to 5.2x at block=5. The circular scheme wraps blocks past the end, so
    every observation lies in exactly `block` windows and inclusion is uniform (measured ratio 1.02-1.04, i.e.
    Monte-Carlo noise). It also makes the estimate far less sensitive to the block length, which is a tuning
    parameter no user should have to reason about: on one reference cell, winner_stability spans 0.168 across
    block lengths 1-8 under moving-block versus 0.039 under circular, and the moving-block curve shows an
    unjustifiable 0.157 jump between block=1 and block=2 that is purely the edge effect switching on.
    The cost is joining the end of the sample to its beginning, which is legitimate under the stationarity already
    assumed by the Model Confidence Set apparatus; the canonical MCS implementations use the closely related
    stationary bootstrap, which also wraps. Pass circular=False to recover the previous moving-block behaviour.
    """
    if block <= 1:
        return rng.integers(0, T, size=T)
                     # iid case: identical under either scheme
    if not circular:
        out = []
        while len(out) < T:
            start = int(rng.integers(0, max(1, T - block + 1)))
            out.extend(range(start, min(start + block, T)))
        return np.asarray(out[:T])
    starts = rng.integers(0, T, size=int(np.ceil(T / block)))
    return np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]
def mcs(L, alpha=0.10, B=2000, block=3, seed=0, elimination=None):
    """Model Confidence Set on a T x K loss matrix (lower loss = better).
    Returns (surviving_indices, p_at_stop): the column indices of the models in the MCS, and the p-value at which
    elimination stopped. |surviving| == 1 => point-identified; > 1 => non-identified (a tied set).
    Notes
    -----
    - A pair whose per-period loss differential is CONSTANT and non-zero (one model beats the other every single
      period) has zero bootstrap variance -- treated as a certain rejection (t -> inf), not skipped. A differential
      that is exactly zero is a genuine tie and is uninformative (skipped).
    - If T <= block the bootstrap has no resampling variability; the block is shrunk with a warning so the test
      retains power.
    - BLOCK LENGTH'S EFFECT ON SIZE UNDER REAL SERIAL DEPENDENCE IS NOT WELL-CHARACTERIZED BY THIS PACKAGE. A round-6
      review (2026-08-27) simulated K=5, T=30, alpha=0.10, a genuine true-best model, and AR(1) loss-differential
      dependence (rho=0.7, realistic, not extreme): P(true best wrongly excluded) was 11.4% at block=3 (already
      above the nominal 10%) and 16.4% at block=12 -- a LARGER block made size WORSE, not better, contradicting the
      "bump the block for more-persistent data" advice this docstring used to give. Independently re-verified with a
      different DGP (varying the true edge from 0 to 0.1): block=12 was worse than block=3 in every configuration
      tested. Do NOT assume a bigger block improves coverage on dependent data -- validate block-length choice
      against your own data via simulation (the same discipline EVALUATION_CARD.md's power table already follows),
      and report an MCS block-length sweep rather than a single number regardless of which block you pick.
    - `elimination`: "studentized" (default, HLN's e_max) or "raw" (the pre-2026-07-27 simplification). Pass
      `elimination=None` (the default) to read the module-level `ELIMINATION` flag AT CALL TIME rather than baking
      a value in as a Python default argument -- a plain default argument is evaluated once at function-definition
      time, so setting `selection_fragility.mcs.
ELIMINATION = "raw"` afterwards would silently have no effect.
    """
    if elimination is None:
        elimination = ELIMINATION
    if elimination not in ("studentized", "raw"):
        raise ValueError(f"elimination must be 'studentized' or 'raw'; got {elimination!r}")
    rng = np.random.default_rng(seed)
    L = np.asarray(L, float)
    # ALPHA VALIDATION. FIXED 2026-09-02 (round-2 10-agent review, CONFIRMED SEVERE). `identify.py`'s
    # `_validate_alpha` was added 2026-08-16 for exactly the practitioner slip below and wired into
    # identify.py and resolution.py -- but never into mcs.py, the one module where `alpha` is actually
    # consumed (`if p >= alpha` below). Verified on a T=20, K=4 panel: alpha=10 (meaning "10%") and
    # alpha=nan both returned a POINT-IDENTIFIED singleton, because `p >= 10` and `p >= nan` are both
    # always False so every model is eliminated; alpha=0.0 and alpha=-1 never eliminate anything.
    # All four are silent, plausible-looking, and wrong. Checked inline rather than imported from
    # identify.py on purpose: identify.py imports `arch`, and mcs.py is deliberately numpy-only.
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"mcs(): alpha must be strictly between 0 and 1 (it is a significance level, "
                          f"e.g. 0.05 or 0.10, not a percentage); got {alpha}.")
    if L.ndim != 2 or L.shape[1] < 2:
        raise ValueError(f"mcs() needs a 2-D T x K loss matrix with K>=2 models; got shape {L.shape}.")
    if not np.all(np.isfinite(L)):
        raise ValueError("mcs(): loss matrix has non-finite values (NaN/inf); drop or impute those periods first -- "
                         "a NaN silently poisons a model's mean and can crown a spurious point-identified winner.")
    if np.any(np.abs(L) > _MAX_ABS_LOSS):
        bad = float(np.abs(L).max())
        raise ValueError(f"mcs(): loss magnitude {bad:.3g} exceeds {_MAX_ABS_LOSS:.0e}; this is almost certainly a "
                          f"units/scale mistake (e.g. raw-dollar loss mixed with a normalized metric), not a real "
                          f"per-period loss. Float64 silently overflows the bootstrap se computation well before "
                          f"this scale, which can crown the astronomical model a spurious MCS survivor depending "
                          f"on column order -- see the _MAX_ABS_LOSS note above. Rescale first.")
    if B > _MAX_BOOTSTRAP_B:
        raise ValueError(f"mcs(): B={B} exceeds the {_MAX_BOOTSTRAP_B} bootstrap-resample ceiling. Cost scales "
                          f"with T*B, so a large B on even a small panel can run for a very long time; "
                          f"{_MAX_BOOTSTRAP_B} is already 50x the documented default (2000) and far finer "
                          f"Monte-Carlo resolution than this test's alpha thresholds need. If you have a genuine "
                          f"reason to exceed it, call the bootstrap loop directly rather than through this guard.")
    if B < 1:
        raise ValueError(f"mcs(): B must be a positive integer; got {B}.")
    T, K = L.shape
    if T <= block:
        block_eff = max(1, T // 2)
        import warnings
        warnings.warn(f"mcs(): T={T} <= block={block}; shrinking block to {block_eff} so the bootstrap retains power.")
        block = block_eff
    boot = [_block_idx(T, block, rng) for _ in range(B)]
    surv = list(range(K))
    p_stop = 1.0
    while len(surv) > 1:
        sub = L[:, surv]; k = len(surv)
        dbar = sub.mean(0)
        bmeans = np.array([sub[idx].mean(0) for idx in boot])
          # B x k
        pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        TRb = np.zeros(B); TR = 0.0
        for (a, b) in pairs:
            diff = bmeans[:, a] - bmeans[:, b]
            se = diff.std()
            obs_diff = dbar[a] - dbar[b]
            # DEGENERATE-se TEST, scale-relative -- NOT `se <= 0`. A bootstrap differential that is constant in
            # exact arithmetic does not reliably compute to std()==0: summing and dividing the block means leaves
            # fp residue whose size tracks the LOSS LEVEL. Measured on the T=1 identical-models case: at loss
            # level 1.0 se came out exactly 0.0, but at loss level 0.0 it came out 2.6e-23 -- so the exact-zero
            # test fired for one and missed the other, and the miss produced t_obs = 3.8e15, p = 0.0005 and a
            # "point-identified" verdict from a single period. The guard was magnitude-dependent dead code.
            # Comparing against a relative floor makes it scale-free: se this small relative to the losses being
            # compared carries no information, it is arithmetic noise. Real bootstrap se is O(sd/sqrt(T)), many
            # orders of magnitude above this floor, so no genuine test is caught by it.
            scale = max(abs(dbar[a]), abs(dbar[b]), abs(obs_diff))
            if se <= _SE_REL_FLOOR * scale:
                # A degenerate se means the bootstrap saw NO real variation in this pair's mean difference. That is only
                # evidence of a real gap when there is enough data for the absence of variation to mean
                # something. At tiny T it is an artifact -- with T=1 every resample is the same observation, so
                # se is 0 by construction and ANY non-zero difference, however microscopic, yielded
                # TR = inf => "certain rejection" => |MCS| = 1. A tool whose central claim is that small
                # samples cannot identify a winner must not report point-identification from one period.
                # Below MIN_T_FOR_ZERO_SE we decline to reject on a degenerate standard error.
                if abs(obs_diff) > 0 and T >= MIN_T_FOR_ZERO_SE:
                    TR = np.inf
                                        # deterministic non-zero gap: certain rejection
                continue
                                               # obs_diff == 0: genuine exact tie, skip
            t_obs = abs(obs_diff) / se
            TR = max(TR, t_obs)
            TRb = np.maximum(TRb, np.abs(diff - obs_diff) / se)
        # standard Monte-Carlo p-value (b+1)/(B+1); TR=inf (deterministic dominance) -> p=0 (certain rejection).
        p = float((np.sum(TRb >= TR) + 1) / (B + 1)) if np.isfinite(TR) else 0.0
        if p >= alpha:
            p_stop = p; break
        # ELIMINATION RULE -- see docstring. "studentized" is HLN's e_max: drop argmax of the model's mean-loss
        # deviation from the survivor-set average, studentized by the same bootstrap already drawn.
        if elimination == "studentized":
            d_dot = dbar - dbar.mean()
            b_dot = bmeans - bmeans.mean(axis=1, keepdims=True)
            se_dot = b_dot.std(axis=0)
            scale = np.maximum(np.abs(dbar).max(), 1e-300)
            safe = se_dot > _SE_REL_FLOOR * scale
            t_dot = np.where(safe, d_dot / np.where(safe, se_dot, 1.0), np.where(d_dot > 0, np.inf, -np.inf))
            drop = int(np.argmax(t_dot))
        else:
            drop = int(np.argmax(dbar))
                                 # legacy raw-mean rule
        surv.pop(drop)
        p_stop = p
    return surv, p_stop
def model_confidence_set(L: dict, alpha=0.10, B=2000, block=3, seed=0, elimination=None):
    """Convenience wrapper over `mcs` that takes a dict {model_name: array(T)} and returns (surviving_model_names,
    p_at_stop). |surviving| == 1 => the best model is point-identified; > 1 => selection is not identified.
    Note: the returned `p` is the p-value at which elimination STOPPED, not a calibrated confidence in the set; the
    headline quantity is the surviving set (its size), not `p`. See the module caveats on low power at small T.

    TWO INDEPENDENT MCS ENGINES, NOT ONE TUNED TWO WAYS (round-6 foundation-model-researcher
    review, 2026-08-27). This is this package's own native MCS -- `identify.identified()`/
    `mcs_size()` (and `report()`, which calls those) delegate to `arch.bootstrap.MCS` instead, a
    genuinely different implementation kept deliberately separate as an independent cross-check.
    Default bootstrap counts differ on purpose (`B=2000` here vs. `reps=500` there) -- they agree
    once B/reps are matched explicitly. Don't compare their un-matched defaults on the same panel
    and read a difference as a disagreement about the data; pass matching B=/reps= to each if you
    need the same verdict from both.

    Accepts a `LossPanel` directly (FIXED 2026-09-07, round-4 8-lens PyPI-preflight audit): used to
    raise `TypeError: 'LossPanel' object is not iterable` at `list(L)` below -- the same bug class
    already fixed for `identify.identified()`/`mcs_size()` and for `fragility.py`'s
    per_period_winner/condorcet_winner/condorcet_status the same day. `mcs()` itself (the raw T x K
    matrix engine this function wraps) is unaffected and unchanged -- it was never meant to accept a
    LossPanel, only this convenience wrapper is."""
    L, _ = _unwrap_panel(L, None)
    models = list(L)
    if len(models) < 2:
        raise ValueError(f"model_confidence_set needs >=2 models; got {len(models)}.")
    lengths = {int(np.asarray(L[m], float).size) for m in models}
    if len(lengths) != 1:
        raise ValueError(f"all models must share the same number of periods; got lengths {sorted(lengths)}.")
    M = np.column_stack([np.asarray(L[m], float) for m in models])
    surv_idx, p = mcs(M, alpha=alpha, B=B, block=block, seed=seed, elimination=elimination)
    return [models[i] for i in surv_idx], p
