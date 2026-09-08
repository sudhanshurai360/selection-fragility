# selection-fragility

**Is your "best" forecasting model a real choice, or a fragile artifact of a few periods?**

`selection-fragility` provides two complementary, decision-oriented diagnostics for forecast *model selection*:

- **Model Confidence Set** (`model_confidence_set`) — which models are statistically *indistinguishable* from the best
  (Hansen, Lunde & Nason 2011). `|MCS| == 1` ⇒ the best model is **point-identified**; `> 1` ⇒ selection is **not
  identified** (a set of models is tied, and crowning one is not supported by the data).
- **Decision breakdown point** (`decision_breakdown`, the object **k\***) — the *fewest evaluation periods whose
  deletion flips the pooled winner*. A small k\* means the crowned model rests on a handful of periods (often a single
  shock) and should not be trusted as a stable choice.

Plus selection-as-election diagnostics (`fragility`): score (pooled) vs plurality (per-period) vs Condorcet winners,
genuine intransitive **cycles** vs pairwise **ties**, and the concentration of the winning margin in one pivotal period.

## Install

```bash
pip install selection-fragility
# or, from a checkout:
pip install -e .
```

Imports as `selection_fragility`. Note that an unrelated package named `kstar` exists on PyPI (kinase-substrate
analysis) — it is not this project.

Direct dependencies: `numpy`, `pandas`, `arch`, `scipy` (declared in `pyproject.toml`); a clean
install pulls in their own transitive dependencies (`statsmodels`, `patsy`, and a few smaller ones)
on top of those four. Requires Python >=3.10.

## Quick start

```python
import numpy as np
from selection_fragility import LossPanel, report

# Per-period loss for each model over T periods (e.g. per-year MASE); lower is better.
# Ten years of losses for three models: ar1 is steady, nbeats is better on average but
# blows up in one shock year, ets is uniformly worse.
L = {
    "ar1":    np.array([1.00, 0.98, 1.02, 1.01, 0.99, 1.03, 0.97, 1.00, 1.02, 0.98]),
    "nbeats": np.array([0.90, 0.88, 0.92, 0.91, 0.89, 0.93, 0.87, 0.90, 2.60, 0.88]),
    "ets":    np.array([1.20, 1.18, 1.22, 1.21, 1.19, 1.23, 1.17, 1.20, 1.22, 1.18]),
}

panel = LossPanel.from_losses(L)
print(report(panel))
```

> **Name your periods.** `from_losses`/`from_forecasts` both take a `labels=` argument. Without it,
> PIVOT reports periods by bare position (`k*=1 of 10 periods (8)`), which is not actionable on its
> own. Pass real labels and get real dates back:
> ```python
> years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
> panel = LossPanel.from_losses(L, labels=years)
> print(report(panel))   # PIVOT now names the responsible year(s) directly, e.g. "(2023)"
> ```

> **Watch out — loss direction is not checked.** This package assumes *lower is better* throughout
> (it is built on loss panels: MASE, RMSE, absolute error, and similar). If you feed it *accuracy*-style
> data instead — anything where *higher is better*, e.g. classification accuracy, R², or a 0-1 score —
> nothing raises an error. You get a fully confident, silently **inverted** answer: the worst model by
> your own metric gets crowned the "winner." There is no automatic check for this (the library has no
> way to know your metric's direction), so it is entirely on you to pass loss, not accuracy.

This prints a one-screen VERDICT (is the winner identified — the Model Confidence Set), LEADERBOARD
(mean loss, in-MCS, periods won), RESOLUTION (observed edge vs. the minimum detectable edge at this
sample size, an MCB bound), and PIVOT (k\*, the responsible period(s), `concentration_share`). This
is the recommended entry point — see [Staged v1.0 API](#staged-v10-api-recommended) below for the
functions behind each section, and **Core API** for the lower-level `fragility()`/
`model_confidence_set()` diagnostics `report()` is built on.

### Staged v1.0 API (recommended)

| stage | function | answers |
|---|---|---|
| 0 — panel | `LossPanel.from_losses(data)` / `LossPanel.from_forecasts(df, y_true, period, ...)` | build a validated loss panel from a dict, wide DataFrame, ndarray, or a forecasts+actuals table |
| 1 — identify | `identified(L, alpha=0.10)`, `mcs_size(L, alpha=0.10)` | is the best model point-identified (Model Confidence Set)? |
| 2 — resolve | `resolution_report(L, w=None, alpha=0.05, power=0.80, mcs_alpha=0.10)` | does the observed edge actually clear significance at this sample size, or must the read be refused (R1/R2 rules — see the function's docstring)? also exposes `minimum_detectable_edge`, `selection_regret`, `mcb_bound`, `significance_boundary` individually |
| 3 — certify | `prop22_certifies(k_star, T)`, `certified_tied_subset(...)` | a zero-simulation significance certificate for k\* (no bootstrap needed) |
| 4 — pivot | `concentration_share(...)`, `pivot_agreement(...)` | how much of the winning margin comes from the single most influential period |
| 5 — compare | `compare(old_panel, new_panel)` → `ChangeReport` | did the champion change run-over-run, and why |
| 6 — report | `report(panel, alpha=0.10, full=False)` | the one-screen summary of stages 0-4 |

**A note on cost.** Stages 1 and 6 (`identified`/`mcs_size`/`report`) scale steeply with the number
of *models* (K), not the number of periods (T) — measured directly: ~1s at K=100, ~3s at K=200,
potentially minutes at K=500+. `LossPanel` warns once, at construction time, past K≈100. Large
model sweeps (an AutoML search, a big hyperparameter grid) are supported, just slow; narrow to a
candidate shortlist first if you don't need per-model MCS membership on the full set.

**Ingesting forecast frames.** `from_forecasts()` reads a *wide-by-model* cross-validation frame
directly — one row per (series, period), one column per model's prediction — the shape
`statsforecast`/`neuralforecast`'s `cross_validation()` already produce:
```python
# df columns: unique_id, ds, cutoff, y, AutoARIMA, AutoETS, Theta   (one column per model)
panel = LossPanel.from_forecasts(df, y_true="y", period="cutoff", group="unique_id")
```
If your data is *fully long/tidy* instead (one row per series-period-**model**, with a `model` name
column and a single prediction column), pivot it to the wide-by-model shape above first:
```python
wide = df_long.pivot_table(index=["unique_id", "ds", "cutoff", "y"], columns="model",
                            values="y_pred").reset_index()
panel = LossPanel.from_forecasts(wide, y_true="y", period="cutoff", group="unique_id")
```

### Core API

`fragility`, `decision_breakdown`, `breakdown_number`, `winner_stability`, `exchangeable_benchmark`,
`surprise_concentration`, `pooled_winner`, `per_period_winner`, `condorcet_winner`/`condorcet_status`,
`mcs`/`model_confidence_set` are the original, lower-level diagnostics `report()` is built on top
of — not deprecated, just closer to the metal. They are the same functions the accompanying paper's
own results are computed from, so they matter beyond convenience: anyone checking a published
number against the code wants `k_star`/`winner_stability`/`condorcet_status` in this raw form, not
filtered through `LossPanel`'s validation. They remain public because `fragility()` is still the
most direct way to get `k_star`, `concentration`, `condorcet_status`, and per-period winners in one
call — **except its `fragile` field**, which is retired (see the field table below); use
`resolution_report()`/`report()` for a fragility verdict instead.

**`surprise_concentration(L, labels=None)`** answers a different question from everything else in
this package: not "is the champion's win fragile," but "*why* would it be, if it is" — is the
panel's own irreducible task difficulty concentrated in one shock period, or is it a near-tie
leaderboard with no single hard period at all? Per period, difficulty is the best loss *any* model
achieved (so it isolates task difficulty, not any one model's skill); "normal" difficulty is the
median across periods. Returns `sci` (share of all excess difficulty sitting in the single hardest
period, near 1 = one dominant shock), `eff_surprising_periods` (effective count of surprising
periods), `hardest` (which period), and `d_max_over_median` (how many times normal the hardest
period was — `NaN` if the normal baseline and the hardest period have opposite signs, since the
ratio is undefined there). Don't confuse `sci` with `fragility()`'s own `concentration` field:
`sci` is a property of the *data* (task difficulty, independent of any winner); `concentration` is
a property of the *decision* (the champion's own margin concentrated in one period) — a panel can
score high on one and low on the other.

### Command line

`python -m selection_fragility compare` gates a CI/pipeline promotion step without writing any
Python — the same `compare()` that stage 5 exposes, run against two `LossPanel`s previously
written with `.save(path)`:

```bash
python -m selection_fragility compare previous.json current.json [--alpha 0.10] [--exit-code]
```

- `previous`/`current` — paths to `LossPanel`s saved with `panel.save(path)`.
- `--alpha` — the Model Confidence Set level passed through to `compare()` (default `0.10`).
- `--exit-code` — without it, the command always exits `0` after printing the `ChangeReport`; with
  it, exit `1` when the report recommends action (`.act` is `True`) so a CI step can fail on a real
  champion change, and exit `0` when nothing changed.
- A bad input (missing file, corrupt/non-`LossPanel` JSON, mismatched model sets between the two
  panels) prints a one-line message to stderr and exits `2` — distinct from exit `1`, so "the
  comparison itself found something worth flagging" and "the comparison couldn't run at all" are
  never conflated in a pipeline's exit-code check. Any other failure is a real bug, not bad input,
  and is left to raise as an uncaught exception (exit code from Python's own default), not silently
  folded into exit `2`.

## Interpreting the output

Fields returned by the `fragility()` call (see [Core API](#core-api) above). For the
recommended `report()` entry point, the VERDICT/LEADERBOARD/RESOLUTION/PIVOT sections it prints are
documented inline in `report()`'s own docstring.

| field | meaning |
|---|---|
| `k_star` | fewest periods whose deletion flips the pooled winner (small ⇒ fragile) |
| `k_over_T` | k\* as a fraction of all periods |
| `concentration` | share of the winning margin from the single most influential period (`>1` ⇒ one period outweighs the whole net margin) |
| `reversal` | does the pooled (score) winner differ from the per-period (plurality) winner? |
| `condorcet_status` | `'winner'` / `'cycle'` (genuine intransitive) / `'tie'` (pairwise tie) |
| `winner_stability` | fraction of block-bootstrap resamples in which the same model wins. **Not interpretable on its own — read it against `exchangeable_benchmark(K, T)`; see below.** `nan` when the margin is degenerate |
| `degenerate_margin` | `True` when the models are indistinguishable to floating point, so "which one wins" is undefined |
| `binding_opponent` / `binding_opponents` | the rival that overtakes the winner with the fewest deletions; the full tied set when several do |
| `concentration_range` | `(min, max)` of `concentration` across the tied binding opponents — report this, not the point |
| `plurality_winners` | full set of per-period winners when the plurality is itself tied |
| `screen` | convenience **screening** flag: degenerate margin, **or** `winner_stability < 0.60`, **or** `k*/T < 0.25`. A triage heuristic, not a calibrated verdict. |
| `fragile` | fragility **relative to chance**: `True` if the margin is degenerate; otherwise `True`/`False` against `benchmark`'s 5th percentile if you pass one, and **`None` (unknown) if you do not**. It is never `False` merely because no benchmark was given. |

> **⚠️ `winner_stability` is NOT calibrated by itself — always compare it to `exchangeable_benchmark()`.**
> Under panels with *no true skill difference at all*, `winner_stability` does **not** go to `1/K`. At K=6, T=32 its
> null median is about **0.50**, with a 5th percentile near **0.30**. So a value anywhere in the 0.4–0.6 range is what
> **chance alone** produces — it is not evidence of fragility. Compute `exchangeable_benchmark(K, T)` for *your* K and
> T and report your observed value against it: a result is unusually fragile *relative to chance* only if it falls
> below that 5th percentile. On the accompanying study's own 26-cell panel, 14 cells sit below 0.60 but only **2**
> fall below the null's 5th percentile — which is roughly what chance would produce. Never read a bare threshold, and
> never read `winner_stability` against `1/K`.
>
> **Reading `k_star`, `fragile`, `concentration`:** `k_star == 0` means there is *no strict* pooled winner (a tie) —
> read the MCS, not "maximally fragile." `fragile` is `True` when the margin is degenerate, **or**
> `winner_stability < 0.60`, **or** `k*/T < 0.25` — any one of the three suffices (each axis can independently raise
> it). Because the 0.60 cut is *not* calibrated (see above), treat `fragile` as a **screening flag, never a finding**.
> It also does **not** capture MCS non-identification, so always read it *together with* the MCS (an exact tie is
> maximally non-identified yet can show `fragile == False`). `concentration` is `nan` when there is no strict winner.
>
> **`winner_stability` can be `nan`, and that is informative.** When every model's loss is identical to floating point, `degenerate_margin` is `True` and `winner_stability` is `nan` rather than `1.0`. Earlier versions returned `1.0` here, which read as "maximally stable" for a decision that is in fact undefined — the resampled winner never changes only because there is nothing to change. `fragile` is `True` in this case. Guard with `math.isnan(...)` before comparing `winner_stability` to a threshold.
>
> **`concentration` is opponent-specific.** It is measured against the *binding* opponent and is therefore the largest value across rivals. Where several opponents tie at the minimum k\*, use `concentration_range` and report an interval: on the reference panel, one cell is `22.19` against one tied opponent and `1.08` against another.

## Honest reporting (recommended)

- **Low power — and not only at small T.** Under the **default** `elimination="studentized"` rule (Hansen–Lunde–Nason's
  published rule, what this package uses unless you say otherwise), the equal-predictive-ability test detected true
  edges of 2%, 5% and 10% at rates of **0.19**, **0.50** and **0.68**, and a 30% edge 77% of the
  time. That was on 13 shock-prone macro series with **25–36 evaluation periods each** (median 30), so the shortfall
  at small edges is not a short-panel artifact: the variance of shock years swamps a few-percent difference in mean
  accuracy. `|MCS| > 1` can therefore reflect **low power** as much as a genuine tie — read it as *"cannot be
  distinguished at this sample size,"* never as *"these models are equivalent."* False discovery on a genuinely null
  panel is **0.007**; sensitivity at small edges is the weak side.

  **These figures correct an earlier version of this README**, which reported 0.00 at 2%, 5% and 10%. Those came from
  a superseded simulation that re-centred every model to an equal pooled mean before injecting the edge, which made one
  high-variance benchmark impossible to eliminate. They understated this package's capability and must not be quoted.
  Part of the remaining shortfall is arithmetic rather than resolution: restricted to trials where the planted edge
  actually makes the focal model the pooled best, detection at a 10% edge is **0.87**. See `EVALUATION_CARD.md`
  for the full table.
- **`|MCS|` is the headline, not the p-value.** The second return value of `model_confidence_set` / `mcs` is
  `p_at_stop` (the p-value at which elimination halted), **not** a calibrated confidence in the set.
- **Report a block-length sweep** for the MCS, and don't assume a larger block is safer for more-persistent data. A
  round-6 review (2026-08-27) found the opposite in simulation: at realistic AR(1) loss-differential dependence
  (rho=0.7), P(true best wrongly excluded) went from 11.4% at block=3 to 16.4% at block=12 — a bigger block made
  size *worse*. Block length's effect on empirical size under real serial dependence isn't well-characterized by
  this package; validate it against your own data (see `mcs()`'s docstring) and show the sweep rather than a single
  cherry-picked block, regardless of which block you pick.
- **Report k\* alongside `winner_stability`** — k\* is the vivid "how few periods" number; winner_stability is its
  bootstrap-calibrated companion.
- k\* is an exact, greedy-optimal quantity on the fixed-weight additive loss-differential margin; it is a lower
  envelope over the pairwise breakdown points against each opponent. **k\* is a descriptive combinatorial fact about
  the observed data (like a minimum), not a hypothesis test with a controlled error rate** — no multiple-comparisons
  correction is needed or applied when taking the min over opponents. Contrast with `mcb_bound`, which *is* a
  corrected confidence bound for the same post-hoc rival-selection structure.

## Adversarial use / competition settings

The diagnostics here were built for a trusted analyst diagnosing their own model comparison, not for a setting
where the people being evaluated know the methodology and have an incentive to beat it. If you plan to use this
package's output — especially an MCS "co-champion" / "statistically tied" claim — in a competitive setting
(a forecasting competition leaderboard, a benchmark with prizes, anything where participants can see and react to
the rules), two concrete gaming vectors are real and confirmed, not hypothetical:

- **Noise can hide genuine inferiority inside a tie claim.** A model with a real, substantial mean-loss disadvantage
  (~30% relative) can survive MCS elimination in up to 100% of trials if its per-period losses carry high
  idiosyncratic variance, versus 0% at the same mean disadvantage with low variance. This is not a bug — it is an
  inherent property of any equal-predictive-ability test, since noise suppresses the studentized elimination
  statistic below threshold — but it is a real, deliberately-engineerable exploit: a participant could submit
  erratic forecasts specifically to manufacture a statistical-tie claim rather than earn one. **Do not use MCS
  survival alone as a "co-champion" criterion in an adversarial setting; pair it with an independent accuracy
  floor.**
- **Whoever controls the weight vector controls the result.** If `w` is participant-supplied (self-reported
  confidence, "which periods matter") rather than fixed by the competition operator in advance, it can flip both
  the pooled winner and k\* entirely — this is correct arithmetic, not a defect, but it means **weights must always
  be operator-controlled and fixed before results are known, never participant-chosen**, in any setting with an
  adversarial party.

One related concern that is already handled: strategic period-withholding (submitting only "easy" periods to
inflate an apparent record) is closed at the input layer — every entry point requires equal period counts across
all models being compared and raises a clear error otherwise, so there's no way to shrink your own effective sample
size without also being excluded from the comparison.

## Pooled results can mask hidden heterogeneity

**The general principle, established across six independently-run domain scenarios, not one:** every diagnostic in
this package — MCS, k\*, `concentration_share`, PIVOT — answers a question about the *pooled* panel: is the current
champion's win real, and what would flip it? None of them ask "does this panel actually contain more than one
population with different answers?" That second question has no native diagnostic here, and a pooled result that
looks maximally confident (VERDICT "identified," RESOLUTION "resolved," low `concentration_share`) can still be
hiding a real, decision-relevant split underneath — because `concentration_share`/PIVOT are built to explain the
champion's *own* periods, by construction, and generally have no way to flag heterogeneity that isn't a minority
subset of the champion's own removal set. Five concrete, independently-confirmed instances of this same mechanism,
each from actually building and running the scenario against real code, not speculation:

- **Time-regime confounding.** A promo-confounded retail panel where the pooled winner was 3.5x *worse* than the
  alternative specifically during promotional weeks, with nothing in the standard report hinting at it. If your
  periods have known regimes (a marketing calendar, a season with distinct phases, cohorts under different
  conditions), slice or weight by regime deliberately (see `w=` above) — don't trust the pooled read alone.
- **Subgroup disparity — architecturally invisible, not just undetected.** A panel where the pooled comparison and
  a 75%-majority subgroup's own comparison both confidently, independently agree on one model, while the
  minority subgroup's own comparison just as confidently names a *different* model (that model ~2.7x worse for
  that subgroup). Unlike a time-regime split, a subgroup split isn't a split of *periods* at all, so
  `concentration_share`/PIVOT have no "responsible period" to point at even in principle — this class of
  heterogeneity is invisible to every within-panel diagnostic by construction. If your comparison spans populations
  that might respond differently, run it separately per population; a clean pooled result is not evidence they agree.
- **A shock/event window can reverse the pooled winner.** A model that wins the pooled comparison overall can lose
  decisively when the panel is restricted to the periods that matter most operationally (e.g. the weather-shock
  years in a crop-yield panel) — the pooled view gives no hint which periods would reverse the ranking if isolated.
- **The aggregation choice itself, not just the data, can decide the winner.** When many underlying observations
  get collapsed into one loss per period (e.g. many individual properties into one period's valuation error), mean
  vs. median aggregation can crown *different* models from the same underlying data — driven by a small number of
  outlier observations within a period, not by anything about the periods themselves. Pick and disclose an
  aggregation method deliberately; don't assume it's a detail.
- **A "still winning" pooled verdict can rest on stale history.** In a setting where a model's advantage erodes
  over time (e.g. an adversary adapting to whichever detector is currently ahead), the pooled champion can remain
  unchanged even after a real, more-recent reversal — with `decision_breakdown`'s own responsible-period set
  concentrated entirely in the outdated window and nothing in the output flagging that those periods are old. Check
  *when* the responsible periods fall, not just how many there are.

## Multiplicity across many independent panels

If you run this tool across many independent panels or series in one batch — many states, many SKUs, many
regions, the production pattern this package has been validated at — **each panel's "identified" verdict is
computed in total isolation, with no awareness of how many others are being run alongside it.** On a batch of
genuinely null panels (no real signal in any of them), some fraction will still read "identified" from noise
alone, at a rate well above any single panel's own nominal level — a 40-panel simulation with no true signal
anywhere returned "identified" on roughly a quarter of panels, not the ~10% a single panel's own alpha would
suggest. This package has no built-in correction for this: there is no equivalent of a Bonferroni/Benjamini-Hochberg
adjustment applied across a *batch* of MCS calls. If you're running many panels, treat a single panel's
"identified" verdict as a first-pass flag, not a final answer — corroborate it (a held-out check, a
domain-plausibility read, or an explicit multiplicity correction applied to the batch of verdicts yourself) before
acting on any one panel's result in isolation, and expect roughly (batch size × single-panel alpha) false
"identified" reads by chance alone across a large batch of genuinely tied panels.

## Other domain-specific pitfalls (confirmed, not hypothetical)

Two more findings, unrelated to hidden heterogeneity, each from actually building and running the scenario against
real code:

- **A single extreme observation can decide the whole comparison, and the leaderboard alone won't tell you.** In
  domains with genuinely heavy-tailed losses (e.g. insurance claim severity, any "one catastrophic event" data),
  one outlier period can flip the pooled winner and crash k\* from a robust double-digit value to 1 — correct
  arithmetic, and the tool's own diagnostics (k\*, `concentration_share`) name the responsible period precisely if
  you check them. The risk is a reader who only glances at LEADERBOARD's declared winner: that view alone looks
  identical whether the win is genuinely robust or driven by one point. Always read k\*/RESOLUTION alongside the
  leaderboard, not instead of it.
- **Right-censored or truncated observations are accepted with no warning.** If an outcome isn't fully known yet
  at your observation cutoff (equipment that hasn't failed, a patient still enrolled, anything "still ongoing"),
  nothing in this package can detect that a loss computed against a censored outcome is really a proxy, not the
  true value — it will be treated like any other real loss. This package was not designed for survival-style
  censored data; if your domain has this shape, resolve censoring before computing losses, not after.

## Related work — how k\* relates to AMIP

The closest prior art to k\* is the **Approximate Maximum Influence Perturbation (AMIP)** — Giordano, Meager &
Broderick (2026), *"An automatic finite-sample robustness metric: when can dropping a little data change
conclusions? Part I: definitions and experiments,"* Philosophical Transactions of the Royal Society A 384(2321):20250001
(preprint: arXiv:2011.14999, 2020). AMIP asks the general question "how little data would need to be dropped to
flip a conclusion?" for smooth estimators (OLS, IV, GMM, MLE, variational Bayes).

We are precise about the relationship, not evasive: k\* is computed on the additive, pairwise loss-differential
margin, which is a **linear** functional — and for a linear target, AMIP's influence-function approximation is
already exact. We do **not** claim k\* is exact where AMIP is only approximate; that claim would be false. The
non-reducible contributions here are (a) a **calibrated block-bootstrap null** quantifying how much of the observed
fragility is attributable to chance, which AMIP does not provide, and (b) the **transport** of the drop-to-flip
concept specifically to a **model-selection decision** on **dependent, vintaged** real-world data — a different and
more applied setting than AMIP's original estimator targets.

Also relevant, in a different domain we do not compete with: **Huang, Shen, Wei & Broderick (2025), "Dropping Just a
Handful of Preferences Can Change Top Large Language Model Rankings"** (arXiv:2508.11847) applies the same style of
worst-case small-fraction-dropping robustness check to Bradley-Terry LLM leaderboard rankings (Chatbot Arena,
MT-Bench). We cite it as the closest instance of "drop-to-flip" applied to preference-ranking systems, and
explicitly do not claim primacy over it — our contribution there is the calibrated null it lacks, applied to a loss-
panel model-selection target rather than a Bradley-Terry preference model.

Other lineage: Hampel / Donoho-Huber's classical breakdown point and influence functions; Hansen, Lunde & Nason's
Model Confidence Set (the set-valued companion object this package builds directly on); Walsh's (2014) clinical
Fragility Index (the closest named analog outside forecasting — differentiated on target: model selection, not
hypothesis significance — and on providing an exact greedy algorithm plus a calibrated null, which the Fragility
Index lacks).

## Using from R

This is a Python package, but the Model Confidence Set literature it builds on (Hansen, Lunde & Nason) and its
applied-econometrics audience skew heavily R. It's usable from R via `reticulate`; the return types are all
R-safe with no special handling:

```r
library(reticulate)
sf <- import("selection_fragility")

L <- list(model_a = c(1.1, 0.9, 1.3, 0.8), model_b = c(1.0, 1.2, 0.7, 1.1))
result <- sf$decision_breakdown(L, w = rep(1, 4))
# plain int / str / list -- converts to an R list with zero special handling
```

`decision_breakdown()`, `resolution_report()`, and similar functions return plain `int`/`float`/`str`/`bool`/`list`/
`dict`, all of which `reticulate` converts natively. The one exception is `LossPanel` — a real Python class instance,
which `reticulate` surfaces as an opaque Python object requiring `$`-based attribute/method access rather than
feeling native to R. Workable, not idiomatic; build panels from a plain named list via `from_losses` and you avoid
holding onto the object at the R side at all.

**Defaults differ from R's `MCS::MCSprocedure` (Bernardi & Catania, CRAN) — this is a parameter mismatch, not an
algorithmic disagreement, and it is not currently documented anywhere else:**

| | this package (`mcs`/`model_confidence_set`) | R `MCS::MCSprocedure` |
|---|---|---|
| `alpha` | 0.10 | 0.15 |
| `B` (bootstrap reps) | 2000 | 5000 |
| block length | fixed integer, `block=3` | auto-selected (AR(p) fit to loss differentials) |

Moving between R and this package with default arguments on both sides will generally *not* reproduce the same
survivor set — not because the algorithms disagree, but because the defaults do. To align results, pass matching
`alpha`/`B` explicitly and choose `block` to match whatever R's auto-selection picked for your data (there is no
adaptive block-length option here; you supply it).

**Disclosed gap**: this package's numeric agreement is currently cross-checked only against Python's
`arch.bootstrap.MCS`, not against R's `MCS` package directly. Treat that as untested parity, not claimed parity,
until a side-by-side run exists.

## Citing

If you use `selection-fragility`, please cite **both** the accompanying paper and this software. Machine-readable
metadata is in `CITATION.cff`, and the evaluation card — what the diagnostics do and where they should not be
trusted — is in `EVALUATION_CARD.md`. **CORRECTED 2026-09-07** (round-4 8-lens PyPI-preflight audit): neither file
ships inside a normal `pip install` (only the wheel's Python modules ship; docs like these live in the sdist and
the source repo, not the installed package itself) — a plain `pip install selection-fragility` will not have a
local copy of either. Read them at the package's
[GitHub repository](https://github.com/sudhanshurai360/selection-fragility) instead of expecting a local file.

Rai, S. (2026). *The Decision Breakdown Point: How Fragile Is "The Best Forecasting Model," and What Does That
Fragility Cost?* [DOI pending — see `CITATION.cff`.]

When citing the software, use the **version** DOI rather than the concept DOI if you are recording which snapshot
produced a particular result; the concept DOI always resolves to the latest release.

## Contributing and getting help

See `CONTRIBUTING.md` for how to run the tests, known code patterns worth knowing before you touch related code,
and how to report a bug (`.github/ISSUE_TEMPLATE/`, at the repository root). This is currently a private
repository — once it's public, GitHub Issues is the primary support channel.

## License

See `LICENSE`.
