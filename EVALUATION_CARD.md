# Evaluation Card — k\* and the ranking-fragility diagnostics

*A short, structured "card" (in the spirit of model cards / datasheets) describing what these diagnostics are, when to
use them, what they output, and their limitations. Pairs with the `selection-fragility` package and the accompanying paper.*

## What it is
Two decision-oriented diagnostics for forecast **model selection**:
1. **Model Confidence Set (MCS)** — the set of models statistically indistinguishable from the best (Hansen, Lunde &
   Nason 2011). `|MCS| == 1` ⇒ point-identified; `> 1` ⇒ selection is **not identified**.
2. **Decision breakdown point (k\*)** — the fewest evaluation periods whose deletion flips the pooled winner. An exact,
   greedy-optimal quantity on the fixed-weight additive loss-differential margin; a lower envelope over the pairwise
   breakdown points against each opponent.

Companion diagnostics: score/plurality/Condorcet winners, genuine intransitive **cycles** vs pairwise **ties**, margin
**concentration** in a single pivotal period, and a block-bootstrap **winner_stability**.

## Intended use
- Deciding whether a leaderboard's "#1 model" is a real, adoptable choice or a fragile artifact of a few periods.
- Reporting the *identification* status and *fragility* of a model-selection decision alongside point accuracy.
- Especially relevant for shock-prone, real-time official statistics where a single period (e.g. a crisis month) can
  dominate the pooled ranking.

**Not intended for**: adversarial/competitive settings (participants who know the methodology and have incentive to
game it) without safeguards — MCS survival is gameable by high-variance submissions, and weights must never be
participant-controlled. See the README's "Adversarial use / competition settings" section before deploying this in
a competition or benchmark with prizes.

## Inputs
- A per-period **loss** for each candidate model over the same `T` evaluation periods (lower is better) — e.g. per-year
  MASE or RMSSE. Provided as `{model_name: array(T)}` (dict) or a `T × K` matrix.
- Optional per-period **weights** (e.g. observation counts). Default: equal weights.
- For the MCS: a bootstrap **block length** reflecting the serial dependence of the loss series.

## Outputs
See the package README table. Headline fields: `mcs` (surviving set), `k_star`, `k_over_T`, `concentration`,
`reversal`, `condorcet_status`, `winner_stability`, `fragile`.

## How to report (honesty standard)
- **Report an MCS block-length sweep**, not a single block — the non-identification count can depend on block
  length, and **a larger block is not necessarily safer for more-persistent data**: a round-6 simulation
  (K=5, T=30, alpha=0.10, rho=0.7 AR(1) loss-differential dependence) found P(true best wrongly excluded) rose from
  11.4% at block=3 to 16.4% at block=12 — the opposite of the intuition that motivated the old advice to size the
  block to the data's persistence. This package does not characterize block length's effect on empirical size
  under real serial dependence; validate your own choice via simulation and show the sweep regardless.
- **Report k\* with winner_stability** — the vivid integer and its bootstrap-calibrated companion. k\* is a
  descriptive fact about the observed data (a minimum), not a hypothesis test — no multiple-comparisons correction
  applies to it, unlike `mcb_bound`, which is a corrected confidence bound for the same post-hoc rival-selection
  problem.
- Distinguish **genuine cycles from pairwise ties** (`condorcet_status`); do not report every "no Condorcet winner"
  cell as a cycle.
- k\* / MCS describe *identification and fragility*, not causation or which model to prefer.

## Limitations
- **Low statistical power, and not only at small T.** In the accompanying study the equal-predictive-ability test
  detected a true accuracy edge at these rates, under the default (studentized) elimination rule:

  | true edge | detected |
  |---|---|
  | 0% | 0.01 (see note: this panel HAS a best model, so this is correct identification) |
  | 2% | **0.19** |
  | 5% | **0.47** |
  | 10% | **0.68** |
  | 30% | **0.78** |
  | 50% | **0.79** |

  Those rates come from the 12 curated shock-prone macroeconomic series in the accompanying study, with
  **25–36 annual evaluation periods each** (median 33), not from a handful of periods. **These figures replace an
  earlier table that reported 0.00 at every edge up to 10%.** That table came from a superseded simulation design
  which re-centred all six models to an exactly equal pooled mean before injecting the edge, making one
  high-variance benchmark structurally uneliminable; the accompanying study documents the error and its
  correction. The corrected design leaves the panel uncentered. A separate false-discovery rate, measured on
  genuinely null i.i.d. panels where no model is better and averaged over 5 independent seeds (a single-seed
  estimate at this budget can plausibly differ by roughly 2x -- an earlier single-seed figure of 0.007 undershot
  this by about that much), is **0.0092** (0.0098 under the raw elimination rule).

  Read the small-edge rows with their arithmetic ceiling in mind: an edge handed to a model already well behind does
  not make it the best model, and no test could single it out. Restricted to trials where the planted edge actually
  does make the focal model the pooled best, detection at a 10% edge is **0.89** rather than 0.68,
  and at 2% it is **0.39** rather than 0.19. The remaining shortfall is the variance of shock years:
  a single 2020-sized cliff swamps a few-percent difference in mean accuracy.

  So a large tied set (`|MCS| > 1`) can reflect **low power** as much as a genuine tie. Report non-identification as
  "cannot be distinguished at this sample size," never as "these models are equivalent," and do not lean on it as a
  well-powered negative. Specificity is the property this test genuinely has; sensitivity is not.
- The MCS elimination step defaults to Hansen-Lunde-Nason's published studentized (e_max) statistic; pass
  `elimination="raw"` to recover the pre-2026-07-27 raw-mean simplification, which gives identical tied sets on
  every one of the 12 curated series in the accompanying study. Cross-checked against an independent MCS
  implementation (`arch`): the identified/non-identified verdict agrees on all 12 series; exact tied-set
  cardinality agrees on 10 of 12 (the two disagreements each differ by exactly one model and remain
  non-identified under either implementation). Cross-check against an independent MCS implementation for your
  own headline claims too.
- Results are conditional on the evaluation window, metric, aggregation, and data vintage — these are axes of the
  decision, not nuisance parameters. Report across them where feasible.
- k\* is a *deletion*-based robustness quantity; it answers "how few periods flip the winner," not "which model
  generalizes best out of sample."
- The classical-tier fits underlying the benchmark (chiefly the SARIMA and DHR state-space models) do not converge
  cleanly on every window — the frozen forecasts were generated with a substantial rate of MLE non-convergence
  warnings from `statsmodels`, visible in the committed `*_run.log` files. This affects the crowned model in 4 of
  the 12 series in the paper's Table 1 (those whose best model is `sarima` or `dhr`). It does not change any
  reported number — the forecasts are what they are regardless of the optimizer's convergence flag — but a
  practitioner adapting this pipeline to new data should not assume every fit converges cleanly, and should not
  suppress solver warnings globally without checking them at least once.
- **Pooled results can mask hidden heterogeneity** — no diagnostic here asks "does this panel actually contain more
  than one population with different answers," only "is the pooled champion's win real." Confirmed across five
  independent domain scenarios (time-regime confounding, subgroup disparity — architecturally invisible to
  within-panel diagnostics since a subgroup split isn't a period split, shock/event-window reversal, the
  mean-vs-median aggregation choice itself deciding the winner, and a "still winning" verdict resting on stale
  history). Full detail in README's "Pooled results can mask hidden heterogeneity" section.
- **No multiplicity correction across a batch of independent panels.** Each panel's MCS/k\* verdict is computed in
  isolation; running many panels (many states, many SKUs) means some fraction will read "identified" from noise
  alone at a rate above any single panel's own nominal level (~25% on a 40-panel genuinely-null simulation, vs.
  ~10% expected per panel). Treat a single panel's "identified" verdict as a first-pass flag in a batch setting, not
  a final answer. Full detail in README's "Multiplicity across many independent panels" section.
- Two more confirmed, non-hypothetical pitfalls, each found by actually running the scenario — full detail in
  README's "Other domain-specific pitfalls" section: a single extreme observation can decide the whole comparison in
  a way the leaderboard view alone won't reveal (always read k\*/RESOLUTION alongside it); and right-censored/
  truncated observations are accepted with no warning — resolve censoring before computing losses.

## Reproducibility & citation
The diagnostics are deterministic given a seed. **For this package specifically**: **CORRECTED
2026-09-07** (round-4 8-lens PyPI-preflight audit) — this section previously said "the 379-test
suite," a number that had drifted from CHANGELOG.md's own last-recorded count (465, as of round 10)
and from the real current count alike, with no CHANGELOG entry at all for the work done between
them. **CORRECTED AGAIN 2026-09-09** (round-2 stress-review, cross_doc_final lens): 541 had itself
gone stale after the v1.0.1 doc-sync fixes (commit 146d72b) added two new test-suite meta-checks.
**CORRECTED A THIRD TIME 2026-09-09** (round-5 stress-review): 543 had itself gone stale after v1.0.3
added six new regression tests (three for round-5's own new bug fixes, two replacing a pair of
vacuous near-tie tests an independent review found, and one for the `report()` LEADERBOARD fix).
**CORRECTED A FOURTH TIME 2026-09-10** (round-6 10-lens pre-publish due-diligence review): 548 had
itself gone stale after five more regression tests landed for that round's own CLI/resource-
exhaustion/validation/API-alias fixes (still under the same unpublished `[1.0.3]`, not a new bump).
**CORRECTED A FIFTH TIME 2026-09-10** (round-7, the final pre-publish review before v1.0.3 actually
ships): 553 had itself gone stale after three more regression tests landed for that round's own CLI-
encoding, unpivoted-long-frame-warning, and ragged-array fixes. This is the number that ships with
v1.0.3 -- no further review round is planned after this one.
The real, current count (`pytest tests/`, verified directly) is **556 passed, 1 xfailed**.
It includes golden-value regression tests locking down `k_star`, MCS survivor sets, and every
headline statistic against known-correct fixtures, and (as noted above) the MCS implementation is
independently cross-checked against `arch`'s. There is also a CI matrix
(`.github/workflows/selection-fragility-test.yml`, at this repo's own root) verifying a clean
install and full test run across supported Python versions **and** Linux/Windows/macOS on every
change. **CORRECTED 2026-09-07**: this used to point at `../../.github/workflows/...` — a path that
only resolves inside the private development monorepo, not from a PyPI download or a clone of this
package's own standalone repo. A repo-root-relative copy of the same workflow, adapted for
standalone-repo layout (no monorepo-specific path filters), now ships inside this package itself
— see that file's own header comment for the full story. `pyproject.toml`'s dependency floors are the earliest numpy/pandas/scipy versions
with wheels on both the oldest and newest Python this package supports (bumped 2026-08-27; the previous floors
predated `requires-python>=3.10` and had no wheel for Python 3.13) — verified by pinning to exactly those floors and
running the full suite unchanged. `requirements-lock.txt` additionally pins the exact dependency versions the suite
was verified against, for anyone who wants a guaranteed-reproducible environment rather than the flexible range.

**For the accompanying paper's own numbers**: every result regenerates from one command (`python reproduce.py` — see
`REPRODUCE.md` in the paper repo). An earlier draft of this card additionally claimed the paper's frozen inputs are
SHA-256 pinned in a `FROZEN_INPUTS.md` and verifiable via `python code/gates/frozen_manifest.py --check` — **neither
file exists in the repository**; this was aspirational text for gate infrastructure that was never built, not a
description of something that regressed. Removed here rather than left as a dead claim a reader can't verify; the
input-freezing/hashing story for the paper's own reproduction pipeline remains open work.

Please cite both the accompanying paper and this software. The citation metadata, including the minted DOI,
lives in **`CITATION.cff`**. A note on Zenodo's convention (the deposit is live as of 2026-09-08, DOI
`10.5281/zenodo.22652327`, confirmed directly against the record rather than assumed): this repository's
`.zenodo.json` does not exist, and none is needed for an already-minted deposit — `.zenodo.json` is deposit
*input* metadata used to configure a deposit before it is made and never carries the minted DOI back, so
`CITATION.cff` — not `.zenodo.json` — remains the file to check. When recording which snapshot produced a specific result, cite the **version** DOI rather than the
concept DOI; the concept DOI always resolves to the latest release.
