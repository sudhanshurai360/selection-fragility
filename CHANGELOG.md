# Changelog

All notable changes to `selection-fragility` are recorded here. Versions follow
[semantic versioning](https://semver.org/).

## [1.0.0] — unreleased

First public release, accompanying the paper *The Decision Breakdown Point: How Fragile Is "The Best Forecasting
Model," and What Does That Fragility Cost?*

### Added — the v1.0 diagnostic surface (`LossPanel`, identification, resolution, pivot, compare, report)

A full redesign of the reporting layer around one validated entry point, replacing the retired `fragile`/`screen`
verdict (above) with statistics whose calibration was independently verified before shipping, and a discipline of
refusing to answer rather than printing a confident number the data cannot support (R1/R2, in `resolution_report`).

- **`LossPanel`** — the single validated entry point: `.from_losses()` (wide `DataFrame` / 2-D `ndarray` / dict of
  per-model arrays) and `.from_forecasts()` (long forecast-vs-actual frame, computes the loss). Rejects fewer than
  two models or two periods, ragged or non-finite arrays, NaN/inf, duplicate columns (including labels that only
  collide after `str()` coercion), stray metadata columns misread as models, boolean flag columns, mismatched
  weight/label lengths, and loss magnitudes beyond any real per-period metric (`>1e100`, well below where float64
  arithmetic starts to silently overflow). `.save()`/`.load()` round-trip through JSON with the exact same
  validation as building a panel fresh — a saved file is not a shortcut around any of the above.
- **`identified()` / `mcs_size()`** — the Model Confidence Set via `arch.bootstrap.MCS` (`bootstrap='circular'`
  explicitly, since `arch`'s own default is `'stationary'`), column order and dict insertion order independent by
  construction. Refuses non-uniform weights outright rather than silently computing an unweighted MCS while the
  reported champion is weighted.
- **`resolution_report()`, `minimum_detectable_edge()`, `mcb_bound()`, `significance_boundary()`,
  `selection_regret()`** — R1 (is the observed edge against the closest rival actually resolvable at this sample
  size, tested at the correct significance boundary — not the stricter minimum-detectable-edge, a common
  conflation that silently halves the claimed statistical power) and R2 (is the champion point-identified before
  any fragility-adjacent read is answered). `mcb_bound` corrects for having picked the closest rival post-hoc out
  of several candidates (Bonferroni on the critical value). `selection_regret` is a deterministic,
  leave-one-period-out empirical cost of the picking rule.
- **`decision_breakdown()` / `breakdown_number()`** carried over from the original k\* design (below), now
  wired into `prop22_certifies()` / `certified_tied_subset()` — a zero-simulation lower bound on the tied set,
  derived by inverting Proposition 2.2's significance bound directly (no bootstrap at all), certifying
  non-significance only, never significance.
- **`concentration_share()` / `pivot_agreement()`** — replacements for the original `concentration` field
  (rejected after measurement: near-zero power exactly where fragility matters most) and an unshipped
  `pivot_sharpness` candidate (rejected for a structural degeneracy at near-ties). Both were accepted only after
  matched-false-positive-rate testing against real shock configurations, not an AUC comparison.
- **`compare(previous, current)`** — the most-requested missing feature from practitioner review: did the pooled
  champion change since the last run, and is that signal or noise? Identifies genuinely new periods by label (not
  position), refuses to guess when two panels share no periods at all, and reports whether the old champion
  actually left the Model Confidence Set (a real separation) versus still being statistically tied. `.act` is the
  single boolean a CI/pipeline promotion step should gate on.
- **`report(panel)`** — the one-screen summary (VERDICT / LEADERBOARD / RESOLUTION / PIVOT) tying all of the above
  together, with R1/R2 refusal applied consistently so no two sections of the same report can contradict each
  other.
- A CLI (`python -m selection_fragility compare <previous.json> <current.json>`) for gating a promotion step
  without writing Python.

### Fixed after implementation, before shipping

Extensive independent adversarial review (multiple review rounds, several distinct lenses — statistical
correctness, reproducibility, hostile/malformed input, numerical extremes, mutation/aliasing, cross-fix
interaction, statistical calibration) found and fixed real defects across the surface above, including: a churn
statistic that was measuring close to the opposite of its own definition; report sections that could contradict
each other on the same undetermined decision; `LossPanel` silently aliasing (not copying) caller-owned arrays, so
mutating your own data after building a panel could silently change the panel; `LossPanel.load()` bypassing the
same validation `.from_losses()` enforces (accepting negative or all-zero weights, mismatched label lengths, and
non-standard JSON `NaN`/`Infinity` tokens); several statistics whose tie-breaking depended on Python dict
insertion order rather than model identity; and the resolution report's headline "resolved" field claiming a power
target it did not actually deliver (comparing the observed edge against the wrong critical value delivered roughly
50% power at the stated minimum detectable edge, not the intended 80%). None of these affected the retired
`fragility()`/`model_confidence_set()` surface below, which predates this redesign and was not part of it. See the
test suite for the specific regression each fix locks in.

### Fixed in a later, wider round (personal line-by-line review of every source file, not delegated)

Several more rounds of fresh adversarial review, then a personal file-by-file read of the entire package with no
method commitment ("never assume anything is fine"), found real defects that had survived every prior round,
including in files reviewed many times before:

- **`resolution.py`**: the SE estimator used `ddof=0` (population variance) against its own documented intent of
  "sample variance," understating SE worst at small T; `minimum_detectable_edge`/`significance_boundary`/
  `mcb_bound` used a normal (z) critical value when SE was estimated from the same small sample — textbook grounds
  for Student-t (df=T−1) — silently inflating confidence in the `resolved` verdict exactly at small T, the regime
  the tool is meant to serve best; `mcb_bound`'s docstring/`report()` text/README all said "the champion may be up
  to X% *worse*" when the formula computes an upper bound on how much *better* the champion's edge could be —
  math unchanged (already coverage-tested), wording corrected; no T≥2 guard (silent NaN on a 1-period call) and no
  weight-magnitude cap (silent overflow) on the raw-array entry points.
- **`identify.py`**: the exact-tie-breaking jitter's scale was computed panel-wide, so one model with a
  legitimately large-but-valid magnitude (a plausible mixed-units mistake — raw-dollar loss alongside a normalized
  metric) silently corrupted the tie-break for two *unrelated*, decisively-separated small-scale models sharing the
  same panel.
- **`mcs.py`** (the legacy, independently-hand-rolled MCS implementation): no magnitude cap at all — an
  astronomically worse model was silently crowned the *sole* MCS survivor, the opposite of correct, from only a
  `RuntimeWarning`.
- **`panel.py`**: `from_forecasts()` let a model missing a forecast for *some* (not all) rows within a period-group
  silently average over only the rows it did have — comparing models on different, non-comparable denominators
  with no warning; a `(T, 0)`-shaped ndarray crashed with a bare `StopIteration` instead of the clear `ValueError`
  every sibling zero-model path already raises.
- **`report.py`**: two genuinely distinct sub-monthly responsible periods could both truncate to the same `%Y-%m`
  label, printing e.g. `(2020-04, 2020-04)` and reading as a duplicate. Labels are now disambiguated (day-level,
  or an index suffix as a last resort) only within a report whose own responsible-period set actually collides —
  a report with no collision looks exactly as before.
- **`__main__.py`**: every CLI error path (bad file, malformed panel, mismatched models) exited with the same code
  (1) as `--exit-code`'s intentional "block the promotion" signal — indistinguishable to a calling CI pipeline.
  Errors now exit 2.
- **`__init__.py`**: the package's own top-level docstring — the first thing `help()` shows — no longer opens with
  an internal recovery note; it now states what the package does and how to start, matching every other public
  docstring's convention.

### Fixed in a third round (round-2 "does it deliver on its promises" review, 2026-08-26)

The prior round's own CHANGELOG entries above were re-verified against the shipped source rather than trusted, and
three of them were found to describe an intended fix that had never actually been applied — the same defect class
this round exists to catch, recurring inside its own changelog:

- **`__main__.py`**: the "errors now exit 2" claim above was false — no `try`/`except` existed anywhere in the CLI
  dispatch, so a missing/malformed file surfaced as a raw, unhandled Python traceback at exit 1, indistinguishable
  from an actual internal bug. Now genuinely wraps `LossPanel.load()`/`compare()` in a try/except scoped to the
  user-input-class exceptions those functions document (a missing file, corrupt/non-`LossPanel` JSON, a real data
  problem such as a mismatched model set) and exits 2 with a one-line message; anything else still propagates
  uncaught, so a genuine programming error is never mistaken for bad input.
- **`report.py`**: the same-month disambiguation claim above was also false — `_fmt_label` was still a bare
  `strftime("%Y-%m")` with no collision handling anywhere in the file; `k*=2 of 5 periods (2020-04, 2020-04)` was
  reproduced verbatim. `_disambiguate_labels()` now does this for real.
- **`panel.py`**: `from_forecasts()`'s partial-row-NaN guard above was also never shipped — a model with 7 valid
  rows and 3 NaN rows in a period where a sibling model had all 10 built a "clean" panel with no warning, since
  `groupby().mean()` skips NaN by default and the existing NaN/inf check only ever sees the already-averaged
  (NaN-free) result. Now compares each model's non-null row count against the period's true total row count and
  warns, naming the model and the under-covered period(s), before the caller ever gets an "already validated"
  panel with a silently different effective sample size per model.

### Fixed in round 3 (5-persona adoption stress test, 2026-08-26)

Five isolated reviewers, each required to actually install and run the package against a scenario realistic to
their role (an M-competition organizer, an applied forecasting practitioner, a strict IJF referee, an
AMIP-literature-aware researcher, a government/policy adopter) — find only, then a second pass fixed everything
actionable:

- **`EVALUATION_CARD.md`** claimed results were "verified with `python code/gates/frozen_manifest.py --check`"
  against SHA-256-pinned inputs in `FROZEN_INPUTS.md` — neither file was ever built for this package. Corrected to
  honestly describe what verification actually exists (the test suite, the `arch` cross-check, the CI matrix)
  instead of fabricating matching files to make the old claim true. A separate unqualified reference to
  `.zenodo.json`, read as though a Zenodo deposit already existed, was corrected the same way.
- The package's **own docs** (README, CITATION.cff, EVALUATION_CARD.md) had zero engagement with the closest
  prior art (Broderick, Giordano & Meager's AMIP) even though the accompanying paper already handled this
  correctly — a referee or adopter evaluating the standalone package would never see the paper. Ported the
  already-vetted positioning (k\* is exact on AMIP's linear-functional case; the non-reducible contribution is the
  calibrated null plus transport to dependent-data model selection) into the package's own citable docs.
- **`resolution_report()`** didn't accept a `LossPanel` the way `report()`/`compare()` do — passing one gave a
  confusing internal `TypeError` from inside `_validate_losses` instead of a clean top-level message. Fixed for
  API consistency across all three primary entry points.
- Two real, cheap documentation gaps closed: the README quickstart never demonstrated the `labels=` parameter
  (PIVOT printed meaningless period indices instead of real dates — the feature worked, it was just undocumented),
  and never showed how to reshape long/tidy-format data into the Nixtla-style wide shape `from_forecasts()`
  expects.

### Fixed in round 4 (automated fuzzing + zero-context walkthrough, 2026-08-26)

Property-based fuzzing (`hypothesis`, 11 properties, 40–200 examples each) found zero crashes anywhere across the
full public API on valid input, and one real bug:

- **`fragility.py`**: `decision_breakdown`'s k\* was not invariant to positive-scalar rescaling of the weight
  vector, contrary to its documented guarantee ("only relative weights matter"). Root cause:
  `breakdown_number`'s degenerate-margin floor compared a weight-scaled running margin against a threshold
  computed only from the raw loss values, never rescaled by `w` — scaling `w` down far enough could trip the
  floor one period early, understating k\* by 1 (the safer failure direction). Fixed by making the floor
  weight-aware; byte-identical to prior behavior at `w=1` (every pre-existing test).
- The fuzzer's own panel-generator could occasionally draw a column that legitimately triggers `panel.py`'s
  deliberate metadata-column guard — a fuzzer/guard collision, not a product bug — filtered via `hypothesis`'s
  `assume()`.
- Verified release-readiness mechanically for the first time: builds cleanly (sdist+wheel), `twine check` passes,
  installs and imports correctly from a fresh venv, zero known CVEs in dependencies (`pip-audit`).

### Fixed in round 5 (security, reproducibility, extensibility, competition-integrity, R-interop, 2026-08-26)

- **`mcs.py`**: `model_confidence_set`'s bootstrap count `B` had no upper bound — `B=1,000,000` ran unbounded for
  25+ seconds. Fixed with a 100,000 ceiling and a clear error.
- **`pyproject.toml`**: declared dependency floors (`numpy>=1.20`/`pandas>=1.3`/`scipy>=1.7`) predated the
  package's own `>=3.10` Python floor and didn't even build on 3.13. Bumped to the real floors
  (`numpy>=2.1.0`/`pandas>=2.2.3`/`scipy>=1.14.1`), verified by installing at exactly those versions and running
  the full suite unchanged. Added `requirements-lock.txt` pinning the exact currently-tested versions.
- **`__init__.py`**: `from .X import X` re-export pattern silently shadows **four** submodules (`mcs`,
  `fragility`, `compare`, `report`) at their own dotted path — `import selection_fragility.mcs as m;
  m._block_idx` silently returned the wrong object with a confusing `AttributeError`. Fixed with a new,
  documented `_internals.py` giving contributors a stable, non-shadowed path to real submodules without touching
  the existing public API.
- Two real, confirmed (not hypothetical) competition-integrity risks documented, not fixed as code (they're
  correct arithmetic, not bugs): a model with a genuinely worse mean loss can survive MCS elimination up to 100%
  of the time simply by carrying high per-period noise; a participant-controlled weight vector can flip both the
  pooled winner and k\* entirely. See README's "Adversarial use / competition settings" section.
- The package's MCS defaults (`alpha=0.10`, `B=2000`, fixed `block=3`) diverge from R's `MCS::MCSprocedure`
  defaults on all three tunable parameters — previously undocumented. Added a "Using from R" README section with
  a `reticulate` example and an explicit default-mapping table.

### Fixed in round 6 (8-persona forecasting-research + cross-domain review, 2026-08-26)

The headline finding of this round, independently reproduced by 6 of 8 reviewers: **`pooled_winner()`** was not
invariant to positive-scalar rescaling of the weight vector on a near/exact-tied panel — the same
float64-residue-near-a-tie defect class already fixed twice in round 4, in a third independent location. Fixed
with the same relative-floor discipline; verified 0/200,000 mismatches on the exact repro (was 101/200,000).

- **`decision_breakdown`/`pooled_winner`/`winner_stability`/`fragility()`** (the "raw-array tier") gave confusing
  internal errors when passed a `LossPanel` instead of a dict — the same class of gap round 3 found for
  `resolution_report()`. Fixed the same way, via the shared `_unwrap_panel()` helper.
- `report()`'s internal MCS path (`identify.py`, wraps `arch`, default `reps=500`) and calling
  `model_confidence_set()` directly (`mcs.py`'s own implementation, default `B=2000`) can give contradicting
  identification verdicts on the identical panel for highly-correlated candidates (0.998+ correlation). Confirmed
  this is a deliberate cross-check independence, not an accidental mismatch — fixed via explicit
  cross-referencing warnings in both docstrings rather than forcing the defaults to match.
- **The most statistically consequential finding of the whole project**: the package's own guidance to increase
  MCS block length for more persistent/dependent data is actively counterproductive under realistic serial
  dependence. Real simulation (K=5, T=30, alpha=0.10, rho=0.7): P(true best wrongly excluded) is 11.4% at the
  default `block=3`, already above nominal, and 16.4% at `block=12` — the larger block the docstring recommended.
  The counterproductive advice in `mcs.py`/README/EVALUATION_CARD.md was replaced with an honest disclosure
  citing the real numbers, not guidance that doesn't hold up.
- One sentence added distinguishing k\* (descriptive, no multiple-comparisons correction needed) from
  `mcb_bound` (a real corrected confidence bound for the same post-hoc rival-selection structure); an inline
  hint added after VERDICT whenever a result is unresolved, so a reader can't stop at "VERDICT: identified"
  before reaching the power caveat.
- Cross-domain findings disclosed, not code bugs: MCS becomes entirely unavailable under realistic
  traffic-weighted A/B panels (`arch` has no native non-uniform-weight support — the tool correctly refuses
  rather than computing something silently wrong); no native multi-metric reconciliation for recommender-system
  evaluation; Walsh's clinical Fragility Index relationship independently re-verified as related in philosophy
  but not a strict mathematical generalization.

### Fixed in round 7 (integrity audit + realistic long-tenure domain deep-dives, 2026-08-27)

Explicitly requested to answer a direct concern about whether the test suite's "all green" was genuinely real.
A fresh clone from GitHub into an independent venv reproduced the exact claimed test result and independently
re-executed 5 major fixes against that clean install. A hollow-test hunter deliberately reverted two major fixes
and watched tests fail across 8 different files, proving they're load-bearing; found exactly 1 genuinely hollow
test (fixed) and one `xfail`-strictness gap (closed, `xfail_strict = true` now set). Mutation testing (8
controlled, fully-reverted mutations) found 6 of 8 caught immediately by specific tests, and pinpointed 2 real,
previously-unprotected regression gaps — `mcs.py`'s own overflow guard and `panel.py`'s dict-input metadata-column
guard could both have been silently reintroduced-broken with zero test failures; both now have real regression
tests.

- **`resolution_report()`** crashed raw and uncaught on non-uniform weights (only visible with realistic
  enrollment-ramp weighted data, not a smaller uniform example) — fixed internally, preserving real computed
  values rather than falling back to an external NaN pattern.
- **`breakdown_number`**'s "responsible periods" diagnostic finds periods propping up the *current winner's*
  margin specifically — a disruption that temporarily favors the challenger is nearly invisible in that output
  even though something real happened. Not a bug; now stated explicitly in the docstring.
- Domain deep-dives (clinical trials, A/B testing, government/policy at up to 50-year horizons, LLM benchmark
  leaderboard evaluation) found no other bugs.

### Fixed in round 7.5 (first real cross-platform CI run, 2026-08-27)

A CI matrix (Linux/Windows/macOS × Python 3.10–3.13) had existed since round 2 but had never actually triggered
on GitHub — it lived at the wrong path (`.github/workflows/` is only discovered at the repository root, not
inside this package's own subdirectory). Moved to the real root; the first real run found genuine
cross-platform bugs no amount of single-machine testing could have caught:

- The packaging test suite imported the tomllib standard-library module unconditionally, which failed outright on
  Python 3.10 (that module only entered the standard library in 3.11). Added a fallback import and the matching
  conditional test dependency — a CI-environment-level fact verified by the cross-platform matrix itself running
  green, not a single unit test.
- Reading README.md raised a decode error on Windows specifically: any file read without an explicit encoding
  uses the OS default (`cp1252` on Windows, not UTF-8), and README.md contains a real UTF-8 character outside
  that range. Swept the whole tree for unencoded `open()`/`read_text()` calls — 8 sites fixed, including one in
  real library code (`panel.py`), not just tests — again verified by the cross-platform CI matrix itself, not a
  single named function.
- A malformed path (a hardcoded `.rsplit("/tests/", 1)` assuming forward-slash paths, silently wrong on
  Windows' backslash separators) fixed with `pathlib` in two files.
- A CLI stderr-matching test compared a full Windows path string against differently-escaped subprocess output —
  fixed to assert on the filename only.
- A performance-regression guard's 20-second ceiling (calibrated on one fast local machine) was too tight for
  GitHub's shared, slower CI runners — hit on both Linux (20.5s) and Windows (24.2s). Raised to 60 seconds, still
  tight enough to catch a genuine algorithmic regression by a wide margin.

All 12 jobs (3 OSes × 4 Python versions) now pass, confirmed on a real run.

### Fixed in round 8 (10-agent code-validation + 6-domain stress test, 2026-08-27)

The most valuable finding of the whole project came from a holistic, line-by-line fresh-eyes read of the current
codebase (distinct from every prior theme-targeted review): **six public, prominently-exported functions**
(`minimum_detectable_edge`, `significance_boundary`, `mcb_bound`, `selection_regret`, `concentration_share`,
`pivot_agreement`) plus `certified_tied_subset` had never received the `LossPanel`-acceptance fix their sibling
functions got in rounds 3 and 6 — a blind spot no theme-targeted review could structurally catch, since each
prior round's fix was scoped to whichever specific function that round's reviewer happened to test. Fixed via
the existing `_unwrap_panel()` pattern, applied consistently this time. Also corrected a false claim made in a
round-7 fix comment (that `compare()` already degraded gracefully on non-uniform weights) that this same read
caught.

- **`panel.py`**: `LossPanel.save()`/`.load()` was not atomic under concurrent access — a real stress test
  reproduced ~300 corrupted reads per 800 attempts. Fixed with the standard write-to-temp-then-`os.replace()`
  pattern (atomic on both POSIX and Windows). Multiprocessing and threading were both independently confirmed
  safe via real adversarial stress tests otherwise.
- **`compare.py`**: crashed raw and uncaught on non-uniform weights. Fixed to degrade gracefully like its
  documented peers, with a deliberately conservative failure mode — on an MCS computation error it reports
  "undetermined" rather than a default, and forces its action flag to `False`, so an automated pipeline can
  never mistake "couldn't compute" for "genuinely safe to act."
- Four functions that legitimately return `None` per their own accurate docstrings (`condorcet_winner`,
  `prop22_certifies`, `concentration_share`, `pivot_agreement`) had signatures that didn't say so — fixed.
- An exhaustive independent mathematical re-derivation of every formula in the package against its cited
  literature source (Hansen-Lunde-Nason 2011, Kish 1965, Politis-Romano 1992, Bonferroni-corrected MCB bounds)
  found zero discrepancies.
- Six new-domain deep-dives (epidemiology, energy grid, insurance, manufacturing, marketing, sports analytics)
  found zero code bugs and three genuine, now-documented non-bug findings: a pooled `report()` can mask a large
  regime-driven reversal (see README's domain-specific pitfalls section); a single catastrophic observation can
  dominate the whole comparison in a way the leaderboard view alone won't reveal; right-censored data is silently
  accepted with no warning. Epidemiology separately found that evaluating against premature vs. settled ground
  truth can flip the winner in any backfill-revised domain — not a tool defect, permanently regression-locked.
- A formatting defect (squished single-line docstrings in 4 files) and an efficiency issue (`fragility()` ran an
  expensive bootstrap before checking a cheap condition that would discard the result anyway) also fixed.

430 passed, 1 xfailed at the close of round 8 (up from 259 at the close of the initial 6-agent review) — see the
test suite itself for the specific regression each entry above locks in.

### Fixed in round 9 (Phase-2 holistic review of the paper repo, 2026-08-27/28)

- **`fragility.py`**: `concentration`/`conc_by_opp` divided a near-zero-but-technically-positive weighted margin
  `M` through to a huge, meaningless ratio instead of reporting the pair as an effective tie — the same
  degenerate-margin floor `breakdown_number()` already applies (`_MARGIN_REL_FLOOR`) had never been extended to
  these two fields. First fix reused the pre-existing `degenerate` flag's whole-panel scale (`_loss_scale(L)`);
  an independent review found a concrete counterexample where an unrelated, never-winning, large-magnitude model
  elsewhere in the panel inflates that whole-panel scale enough to falsely mark a genuinely decisive margin as
  degenerate — inconsistent with `breakdown_number()`'s own per-pair convention, the function that actually
  computes `k_star`. Corrected to a per-pair scale (`_pair_scale()`, matching `breakdown_number()`'s own formula
  exactly), recomputed independently for each opponent in `conc_by_opp` so one wide-magnitude co-binding
  opponent can't contaminate another's degeneracy read. The pre-existing `degenerate` field was updated to the
  same per-pair scale so it and `concentration` can no longer disagree about whether the same margin is
  degenerate (their documented mutual-consistency contract). A second, fresh-context review independently
  reproduced both the original bug (negative control) and the fix, and confirmed the corrected formula matches
  `breakdown_number()`'s character-for-character.

464 passed, 1 xfailed at the close of round 9.

### Fixed in round 10 (Phase-2 holistic review of the paper repo, 2026-08-28)

- **`fragility.py`**: `breakdown_number()`'s greedy removal order, `np.argsort(c)[::-1]`, does not guarantee
  stability for equal margin contributions (default `kind='quicksort'`), and the trailing `[::-1]` reverses the
  whole result -- so two periods with the exact same contribution could be removed in an order that is an
  accidental byproduct of the sort algorithm rather than a stated convention. k* (the removal count) is
  unaffected -- deleting either member of a tied pair removes the same amount -- but which periods land in
  `removed_period_indices` is not, when the greedy loop stops partway through a tied group. Fixed via
  `np.argsort(-c, kind="stable")`: identical to the old code whenever no two contributions tie exactly, and for
  a genuine tie, deterministically prefers the earlier period (ascending original index). A fresh-context review
  independently reproduced the old-vs-new behavior on a constructed tie-at-the-boundary example, fuzz-tested
  200,000 all-distinct arrays (0 mismatches between the old and new sort) and 50,000 forced-tie arrays (0
  tie-break violations), and confirmed "earliest period" is a more principled convention than a sort-algorithm
  accident given periods (unlike model names) already carry a real chronological order.

465 passed, 1 xfailed at the close of round 10.

### Fixed in rounds 11-13 (11-angle deep review + two further 10-agent audits, 2026-08-29 -- 2026-09-05)

**CORRECTED 2026-09-07** (round-4 8-lens PyPI-preflight audit): this file previously jumped straight
from round 10 to nothing, even though substantial work happened in between (test files carrying
mtimes and in-file comments through 2026-09-05) -- EVALUATION_CARD.md's own test-count claim ("379")
and this file's own last recorded count (465) had both drifted from the real, current figure with no
record of what moved the number. Filling the gap at commit-level detail (full line-item history for
these rounds lives in the paper monorepo's own commit messages and session records, not duplicated
here) rather than leaving it unrecorded:

- **`7be69eb`** (2026-08-29) — closed 6 code/data issues found by an 11-angle deep review, touching
  `test_stage3_kstar_prop22.py` and `test_stage4_pivot.py`.
- **`2d52972`** (2026-09-02) — a 10-agent code review across the paper and this tool together; added
  `test_stage5_compare.py`/`test_stage6_report.py` coverage among 13 files changed.
- **`54669ea`** (2026-09-02, round-2 phase 1) — fixed six confirmed tool bugs, one CRITICAL
  (`pooled_winner()`'s whole-panel-vs-per-pair tie-tolerance defect, see `code/instrument/`'s own
  changelog in the paper repo for the full account); added `test_security_adversarial.py` and more
  of `test_stage3_kstar_prop22.py`.
- **`8a1b0c5`** (2026-09-05, round 3) — a 10-agent audit of round 2's own work, closing findings
  including a `d_max_over_median` mixed-sign fix in `surprise_concentration()` and a hollow-test fix
  in `test_stage3_kstar_prop22.py`'s tie-strength monotonicity check.

### Fixed in round 14 (8-lens external pre-PyPI-publish audit, 2026-09-07)

The first review round to simulate a stranger about to `pip install` this package cold, rather than
auditing it as an insider. Found and fixed real defects none of rounds 1-13 had surfaced, despite
their combined depth -- being correct internally and being safe to hand to an external user turned
out to be different properties:

- **`panel.py`, `from_forecasts()` (BLOCKING)**: the `models=None` auto-inference branch silently
  EXCLUDED any genuinely numeric model column stored as object/string dtype (e.g. from a CSV read
  without explicit dtype control) -- no warning, no error. Reproduced: the actual best model,
  stored as numeric strings, vanished entirely and `report()` printed a confident verdict over the
  remaining models as if nothing were missing. Fixed via scoped `pd.to_numeric` coercion on
  object-dtype columns (explicitly excluding datetime64, which "succeeds" under naive coercion by
  reinterpreting itself as nanosecond-epoch integers).
- **`panel.py`, `from_forecasts()` (HIGH)**: `group` was validated as a column but never used as
  part of the aggregation key, so a duplicated `(group, period)` row (a re-appended CV fold, a
  fan-out join) was silently pooled into that period's mean -- reproduced flipping a verdict with
  zero warning. Fixed with a `ValueError` when `(period, group)` isn't unique, scoped to
  `group`-not-`None` only (a broader version broke a legitimate, intentionally-designed test proving
  multiple rows per period with no group column is a real, supported shape).
- **LossPanel-acceptance gap (HIGH)**: following the README's own Stage 0 -> Stage 1 example
  literally raised a raw internal `TypeError`. `identified()`/`mcs_size()` (identify.py),
  `per_period_winner`/`condorcet_winner`/`condorcet_status`/`surprise_concentration` (fragility.py),
  and `model_confidence_set` (mcs.py) all now accept a `LossPanel` directly via the same shared
  `_unwrap_panel` helper every other panel-accepting function already used -- a completeness sweep
  confirmed every applicable public function now accepts one, with the one correct, documented
  exception (`mcs()`, the raw T x K matrix engine).
- **`report.py` (HIGH)**: `report()` called `identified()`, `mcs_size()`, AND `_run_mcs()`
  separately -- three independent calls each re-running the same `arch` bootstrap MCS computation
  from scratch. Deduped to one call (~3x measured speedup at K=200/T=50: 9.3s -> 3.1s), verified as
  a pure speedup with no behavior change. Added a K-scaling warning to `LossPanel` construction
  (K>100) since no guidance existed anywhere despite `mcs()` having explicit ceilings on other
  parameters.
- **Documentation**: `surprise_concentration()` and the `python -m selection_fragility compare` CLI
  were both real, working, exported/implemented features with zero README mentions -- documented,
  every claim independently verified against real execution.
- **Packaging**: `MANIFEST.in` was missing `requirements-lock.txt` and `tests/*.py` (including
  `conftest.py` -- a downloaded sdist's own test suite gave 140 fixture errors, not a clean pass);
  the CI matrix referenced from EVALUATION_CARD.md only ever lived in the private monorepo, so the
  standalone public repo would have shipped with zero registered CI -- a repo-root-adapted copy now
  ships inside this package itself.

541 passed, 1 xfailed at the close of round 14.

### Fixed in round 15 (final pre-publish real `pip install` + clean-venv test run, 2026-09-07)

Round 14 audited the package's *contents*; this round actually did the thing a first user does --
built fresh sdist/wheel artifacts, installed each into a genuinely clean virtualenv (not the
long-lived dev environment every prior round ran tests in), and ran the real test suite against the
installed copy rather than the source tree. That surfaced one thing internal testing never would,
because the dev environment's own pinned `pandas==2.3.3` (see `requirements-lock.txt`) never
exercises it:

- **`panel.py`, `from_forecasts()` (BLOCKING, pandas 3.x)**: round 14's own numeric-string-column
  fix (above) scoped its `pd.to_numeric` coercion to `is_object_dtype` columns only. pandas 3.0 --
  the version any `pip install selection-fragility` resolves to today, since `pyproject.toml` pins
  no upper bound -- made its dedicated string dtype the DEFAULT representation for string data,
  including a plain `pd.Series(["1.5", "2.3"])`; such a column is no longer `is_object_dtype` at
  all. Reproduced directly against a clean pandas-3.0.5 install: the round-14 fix's own worked
  example (a numeric-string `nbeats` column, the best model) vanished from `model_cols` again,
  silently, the identical failure mode round 14 closed for pandas 2.x. Fixed by also matching
  `isinstance(dtype, pd.StringDtype)`, which correctly catches the new default on 3.x, an explicit
  opt-in `dtype="string"` column on 2.x, and -- confirmed directly -- never a datetime64 column
  (native or object-boxed), so it cannot reopen the datetime-swept-in-as-numeric bug the
  object-only scoping was written to prevent in the first place. Verified against a genuinely clean
  install on both pandas 2.3.3 and 3.0.5: 541/541 and 529/529 (12 tests skip cleanly in the minimal
  sdist-only venv, each for the same reason -- the private golden-master data file those tests read
  isn't shipped with the package or present outside the project's own dev environment, exactly as
  intended) respectively, both from the built sdist, not the source tree. A second, independent
  review then found the fix above still missed a third string-dtype shape (`pd.ArrowDtype`); widened
  to `pd.api.types.is_string_dtype`, confirmed to cover all three shapes and still correctly exclude
  datetime64 -- see the code comment for the full account.
- The one test whose own setup asserted `dtype == object` (`test_numeric_string_column_is_recovered
  _as_a_model`) had the identical pandas-generation assumption baked into its precondition; loosened
  to accept either representation of "a column of number-looking strings," matching the fix above.
