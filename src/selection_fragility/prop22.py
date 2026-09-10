"""selection_fragility.prop22 — Proposition 2.2, inverted into a k* certificate, plus the
certified-tied subset.

k* itself is untouched from `fragility.py` -- exact, brute-force-verified, the real hook. What's
added here is a reframing: `|t| <= sqrt(Tk/(T-k))` inverts to `k >= T*z^2/(T+z^2)`, a k*<=threshold
certificate of non-significance at the given z, with ZERO simulation. For any T>=14 this threshold is
flat at 3. This is a DESCRIPTIVE INFLUENCE MEASURE (like Cook's distance), never a test in the
hypothesis-testing sense -- it only ever certifies non-significance, never certifies significance.

WHY `z` HERE, NOT `alpha` LIKE THE REST OF THE PACKAGE (NOTED 2026-09-10, round-6 stress-review,
api_consistency lens): every other significance-level parameter in this package is `alpha` (a
bootstrap-based p-value threshold, e.g. `mcs()`/`resolution_report()`). This module's certificate is
a CLOSED-FORM Gaussian critical value with no bootstrap involved, so `z` (e.g. 1.96 for two-sided 5%)
is the natural parameter, not a p-value to compare against one. `z = scipy.stats.norm.ppf(1 -
alpha/2)` converts a two-sided alpha to the equivalent z if you want to think in the rest of the
package's vocabulary; this module does not do that conversion for you since it would add a scipy
dependency this module deliberately avoids (numpy-only, matching mcs.py's own design note)."""
import math
import numpy as np
from .fragility import _as_loss_dict, _validate_losses, pooled_winner, breakdown_number, _unwrap_panel
def _prop22_threshold(T, z=1.96):
    """Largest k' such that sqrt(T*k'/(T-k')) < z -- any k <= this threshold certifies non-significance
    of the champion-vs-rival gap at the level implied by z (default z=1.96 <-> two-sided 5%)."""
    k = 0
    while k + 1 < T and math.sqrt(T * (k + 1) / (T - (k + 1))) < z:
        k += 1
    return k
def prop22_certifies(k, T, z=1.96) -> bool | None:
    """True iff k* <= the Proposition 2.2 threshold at this T -- i.e. deleting only k periods (out of
    T) is enough to flip the champion's pooled win over its binding rival, which certifies the gap is
    NOT significant at the level implied by z, with no bootstrap and no simulation.
    Returns None (not True/False) when the certificate does not apply:
    - k <= 0: no strict pooled winner exists (an exact tie) -- there is nothing to certify.
    - k >= T: the ranking survives deleting every period -- maximally robust; this bound gives no
      constraint either way and must never be misread as "significant" (this bound only ever
      certifies NON-significance).
    False means k exceeds the threshold -- the bound does not certify non-significance here, which is
    NOT the same as certifying significance (a real t-test or the MCS answers that question).

    Raises ValueError for NaN/non-finite k, T, or z. FIXED 2026-09-02 (10-agent code-review pass):
    this function previously performed zero input validation, unlike every other public function
    in this package (alpha, power, weights, K vs T, ... are all explicitly validated elsewhere).
    NaN/negative inputs silently returned a plausible, confident False (every comparison against
    NaN is False in Python) instead of raising."""
    if not all(math.isfinite(x) for x in (k, T, z)):
        raise ValueError(f"prop22_certifies() requires finite k, T, z -- got k={k}, T={T}, z={z}")
    if z <= 0:
        raise ValueError(f"prop22_certifies() requires z > 0 (a critical value) -- got z={z}")
    if k <= 0 or k >= T:
        return None
    return bool(k <= _prop22_threshold(T, z=z))
def certified_tied_subset(L, w=None, z=1.96):
    """A no-bootstrap LOWER BOUND on the tied set's size: the champion plus every rival whose pairwise
    k* is certified non-significant by Proposition 2.2 (§2.3). NOT a confidence set, and NOT nested in
    the Model Confidence Set -- the two use different criteria (iid population-sd here vs. the MCS's
    block-bootstrap), so a real panel can and does show the certified subset diverging from the MCS in
    either direction; ship this labelled as a lower bound, never as an alternative MCS.

    Raises ValueError for non-uniform weights. FIXED 2026-09-02 (10-agent code-review pass,
    CONFIRMED CRITICAL finding): this function used to accept an arbitrary, non-uniform `w` (a
    documented, supported feature elsewhere in this package) and thread it straight into a WEIGHTED
    breakdown_number(), then certify that weighted k* using prop22_certifies()'s threshold -- which
    is only proved for the UNWEIGHTED i.i.d. population-sd statistic (see _prop22_threshold's own
    derivation; the companion paper's own validation of this exact bound always calls
    breakdown_number with equal weights, for the same reason). Adversarial search found
    0.30% of random weighted panels produced a FALSE certification -- a genuinely significant gap
    (|t| up to 3.32) certified "non-significant". A uniform weight vector (all entries equal,
    including the w=None default) is still fully supported: rescaling every period by the same
    constant is a documented no-op for breakdown_number, so it cannot expose this defect."""
    L, w = _unwrap_panel(L, w)
    L = _as_loss_dict(L)
    T = _validate_losses(L, w)
    w = np.ones(T) if w is None else np.asarray(w, float)
    if not np.allclose(w, w.flat[0]):
        raise ValueError(
            "certified_tied_subset() only supports uniform weights (or w=None) -- Proposition "
            "2.2's bound is proved for the unweighted i.i.d. population-sd statistic, and applying "
            "it to a WEIGHTED breakdown point is unsound (it can falsely certify a genuinely "
            "significant gap as non-significant). For a weighted panel, use the Model Confidence "
            "Set (mcs.py) instead, which supports weights correctly."
        )
    champ = pooled_winner(L, w)
    la = np.asarray(L[champ], float)
    subset = {champ}
    for m in sorted(mm for mm in L if mm != champ):
        k, _ = breakdown_number(la, np.asarray(L[m], float), w)
        # k == 0 IS CERTIFIED. FIXED 2026-09-02 (round-2 10-agent review, CONFIRMED SEVERE).
        # `prop22_certifies` returns None for k <= 0 (it is a "not applicable" sentinel, not a
        # verdict), and `if prop22_certifies(...)` treated that None as False -- so the STRONGEST
        # possible evidence of a tie was the one case that failed certification, making the relation
        # NON-MONOTONE in tie strength. Reproduced: five byte-identical models returned a certified
        # tied subset of {m0} (size 1, reading as point identification) while model_confidence_set
        # returned all five; and a rival at k*=2 was certified while a rival byte-identical to the
        # champion, at k*=0, was excluded. k*=0 means the champion does not strictly beat this rival
        # on the full sample at all -- no periods need removing to unseat it -- so |t| is zero or
        # undefined and non-significance is immediate, not merely certifiable.
        if k == 0 or prop22_certifies(k, T, z=z):
            subset.add(m)
    return sorted(subset)
