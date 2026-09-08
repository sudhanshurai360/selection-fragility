"""Round-5 competition-integrity / adversarial-gaming review.

Distinct question from rounds 1-4: not "is the code correct" but "if a forecasting competition
adopted this tool's diagnostics as official methodology, can a STRATEGIC PARTICIPANT who knows the
methodology exploit it." Tests here lock in demonstrated properties (real exploits AND confirmed
defenses) so this analysis stays reproducible rather than a one-off finding that silently rots.
"""
import numpy as np
import pytest

from selection_fragility.fragility import pooled_winner, decision_breakdown, _validate_losses
from selection_fragility.mcs import model_confidence_set


def test_noise_injection_survives_mcs_elimination_despite_genuinely_worse_mean():
    """REAL, REPRODUCIBLE GAMING VECTOR (not a code bug -- an inherent property of any
    equal-predictive-ability test, but undocumented as an ADVERSARIAL concern in this package).

    A model with a genuinely, substantially worse sample mean (fixed at +0.3 above the leader, a
    ~30% relative gap on a loss level of ~1.0) survives in the MCS 100% of the time across 60
    independent trials when its per-period losses carry large idiosyncratic variance (sd=3.0) around
    that same bad mean -- versus 0% of the time when a model with the IDENTICAL bad mean has low
    variance (sd=0.10). The mean gap is identical in both cases; only the noise level differs.

    Mechanism: MCS elimination is driven by a studentized statistic (mean gap / bootstrap SE of the
    gap). Injecting variance inflates the SE without necessarily worsening the realized mean, which
    can suppress the statistic below the elimination threshold regardless of how bad the true
    average performance is. This is the same "limited power at small edges" property mcs.py's own
    docstring already discloses -- but that disclosure frames it as passive statistical caution
    ("read |MCS|>1 as cannot be distinguished, not as evidence of a tie"), not as something an
    adversarial participant could deliberately engineer by submitting noisy forecasts. A competition
    using |MCS| survival as a "co-champion" claim is gameable this way; this is a real caveat for
    whoever writes that adoption guidance, not something fixable in the statistic itself.
    """
    T = 30
    rng = np.random.default_rng(1234)
    leader = rng.normal(1.0, 0.10, T)

    n_survive_noisy, n_survive_honest, trials = 0, 0, 40
    for s in range(trials):
        rng2 = np.random.default_rng(9000 + s)
        leader2 = rng2.normal(1.0, 0.10, T)

        noise = rng2.normal(0, 3.0, T)
        noise -= noise.mean()                      # exact demean: fix the realized sample mean
        shadow_noisy = 1.3 + noise                  # same bad mean, huge dispersion

        honest = rng2.normal(0, 0.10, T)
        honest = honest - honest.mean() + 1.3        # identical bad mean, low dispersion

        surv_noisy, _ = model_confidence_set({"leader": leader2, "shadow": shadow_noisy},
                                              alpha=0.10, B=500, block=3, seed=0)
        surv_honest, _ = model_confidence_set({"leader": leader2, "honest_rival": honest},
                                               alpha=0.10, B=500, block=3, seed=0)
        n_survive_noisy += "shadow" in surv_noisy
        n_survive_honest += "honest_rival" in surv_honest

    assert n_survive_noisy >= trials * 0.9, (
        f"expected the noise-injected model to survive MCS elimination nearly every time despite its "
        f"genuinely bad mean (that's the gaming vector this test documents); got {n_survive_noisy}/{trials}")
    assert n_survive_honest <= trials * 0.1, (
        f"expected the low-variance model with the IDENTICAL bad mean to be reliably eliminated -- "
        f"if this also survives often, the gap isn't about noise-gaming, re-diagnose; "
        f"got {n_survive_honest}/{trials}")


def test_exact_tie_pooled_winner_gameable_by_model_name_when_losses_are_copied():
    """CONFIRMED, but a GENERIC tie-break property, not a tool-specific flaw. pooled_winner's exact-tie
    resolution (sorted model name, documented in fragility.py's docstring) means a participant who can
    engineer an EXACT loss-mean tie with the true leader -- trivially achievable by literally copying
    their submission -- wins the pooled_winner title purely by choosing a name that sorts first. This
    is unavoidable for ANY deterministic tie-break once exact ties are achievable via copying; the
    tool already discloses the rule explicitly (an improvement over the pre-fix silent
    dict-insertion-order behavior), and preventing submission copying is an anti-plagiarism concern
    for competition operators, not something this statistical instrument can or should police. Locked
    in here so the documented tie-break behavior doesn't silently change without notice.
    """
    T = 20
    rng = np.random.default_rng(7)
    true_best = rng.normal(1.0, 0.2, T)
    adversary_clone = true_best.copy()              # exact tie by direct copy

    assert pooled_winner({"zzz_true_best": true_best, "aaa_clone": adversary_clone}) == "aaa_clone"
    assert pooled_winner({"aaa_true_best": true_best, "zzz_clone": adversary_clone}) == "aaa_true_best"


def test_ragged_period_withholding_is_rejected_not_silently_allowed():
    """CONFIRMED DEFENSE (not an exploit): a participant cannot strategically withhold predictions on
    hard periods to shrink their own effective T, since every raw-array entry point
    (_validate_losses, called by pooled_winner/decision_breakdown/fragility/winner_stability) and
    model_confidence_set both require every model to report the SAME number of periods, and raise a
    clear ValueError otherwise rather than silently comparing on whatever subset a participant chose
    to submit. This closes what would otherwise be a real gaming vector (submit only your easy
    periods) at the input-validation layer. Locking this in as a competition-integrity guard, not
    just an input-shape guard.
    """
    T_full = 12
    rng = np.random.default_rng(3)
    full_submission = rng.normal(1.0, 0.2, T_full)
    strategically_short_submission = rng.normal(1.0, 0.2, T_full - 3)   # withheld 3 hard periods

    with pytest.raises(ValueError, match="same number of periods"):
        _validate_losses({"honest": full_submission, "cherry_picker": strategically_short_submission})

    with pytest.raises(ValueError, match="same number of periods"):
        decision_breakdown({"honest": full_submission, "cherry_picker": strategically_short_submission})

    with pytest.raises(ValueError, match="same number of periods"):
        model_confidence_set({"honest": full_submission, "cherry_picker": strategically_short_submission})


def test_participant_controlled_weights_can_flip_the_winner_and_k_star():
    """CONFIRMED, expected behavior of any weighted metric -- a governance caveat, not a code defect.
    If the party being evaluated gets to choose the per-period weight vector `w` (e.g. "self-reported
    confidence" or "which periods matter"), they can flip BOTH the pooled winner and k* entirely,
    independent of genuine forecast quality: a model that's clearly worse under equal weighting
    (loses to a steady rival 2.0 vs 1.0 on 5 periods, wins big 0.1 vs 1.0 on the other 5) becomes the
    winner, and the fragility verdict inverts (k*=1 favoring the rival under equal weights becomes
    k*=5 favoring the gamed model), purely by upweighting the periods where it happens to look good.

    This is correct arithmetic for a WEIGHTED metric, not a bug -- the docstring is explicit that `w`
    should represent something objective (observation counts), never a self-serving choice. The real
    safeguard belongs to competition GOVERNANCE: weights must be operator-controlled, fixed and
    disclosed before results are known, never chosen by (or negotiable with) the evaluated party.
    Locked in as a documented caveat for whoever writes adoption guidance for this tool.
    """
    T = 10
    mine = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1])
    rival = np.full(T, 1.0)
    L = {"mine": mine, "rival": rival}

    assert pooled_winner(L) == "rival"
    k_equal, opp_equal, *_ = decision_breakdown(L)
    assert (k_equal, opp_equal) == (1, "mine")

    w_gamed = np.array([0.001] * 5 + [1.0] * 5)          # upweight only the periods "mine" wins
    assert pooled_winner(L, w_gamed) == "mine"
    k_gamed, opp_gamed, *_ = decision_breakdown(L, w_gamed)
    assert (k_gamed, opp_gamed) == (5, "rival")
