# Test suite organization

This suite grew from an original 6-stage design plan into 8+ rounds of adversarial review, each
adding its own test files. Three naming conventions coexist as a result — this doc is the map so
"where are the tests for X" has a fast answer without opening 43 files.

No files were renamed or moved to write this doc: a full audit found the coexisting names are the
safe, correct outcome, not a delayed cleanup. Every rename/reorg risks a subtly broken collection
path or cross-reference across 43 files, 12 real cross-platform CI jobs, and no compensating
navigability win over a good index — the two literal name collisions found (`test_determinism`,
`test_negative_weight_raises`, each appearing in two files) are legitimate: same generic name,
different function under test in each case (verified by reading both bodies), not duplicates.

## 1. `test_stageN_*.py` — the original design-plan tests

Pre-dates the review-round process. Follows `release/TOOL_TEST_PLAN.md`'s own stage numbering; a
stage's tests may pre-date that stage's implementation (some were written to fail until the code
shipped — check each file's own docstring for which parts were live-when-written vs. aspirational).

| File | Covers |
|---|---|
| `conftest.py` | Shared fixtures (F1-F11, named to match `TOOL_TEST_PLAN.md` exactly) |
| `test_stage0_losspanel.py` | `LossPanel` construction: `from_losses`/`from_forecasts`/`from_wide_dataframe`, save/load, validation guards |
| `test_stage1_identified.py` | `identified`/`mcs_size`, the `arch`-delegation MCS wrapper |
| `test_stage2_resolution.py` | `minimum_detectable_edge`, `selection_regret`, `mcb_bound`, refusal rules |
| `test_stage3_kstar_prop22.py` | `k*`/`decision_breakdown`, Proposition 2.2 inversion, certified-tied subset |
| `test_stage4_pivot.py` | `concentration_share`, `pivot_agreement` |
| `test_stage5_compare.py` | `compare(previous, current)` |
| `test_stage6_report.py` | `report()` / `__repr__` — renders Stages 0-5's output, written last |
| `test_crosscutting.py` | Cross-stage tests (X1-X6) that don't belong to one stage |

## 2. Bare thematic names — early review-round findings (rounds 2, 4, 5)

Named for the defect class or property they guard, not a round number — written before the
`test_roundN_*` convention below was established.

| File | Round | Covers |
|---|---|---|
| `test_packaging.py` | 2 | README/CHANGELOG claims vs. actual shipped package (dependency list, install requirements) |
| `test_property_based.py` | 4 | `hypothesis`-based fuzzing — random-but-valid input, invariants across the whole domain |
| `test_reproducibility.py` | 5 | Determinism: cross-process byte-identical output, immunity to global RNG state |
| `test_security_adversarial.py` | 5 | Deliberate hostile-input attacks (not random fuzzing) — malicious files, boundary attacks |
| `test_extensibility_review.py` | 5 | What a contributor extending the package (not just consuming it) can rely on |

## 3. `test_roundN_*.py` / `test_roundN_domain_*.py` — persona and domain reviews (round 3 onward)

The bulk of the suite. Each file is one review round's persona or domain stress test, self-
contained with its own findings documented in its module docstring — read that first, it's usually
more specific than the filename. One outlier: `test_ab_testing_domain_review.py` is round 6's work
but predates this file's round-number-prefix convention (round 7 added a *second*, deeper A/B-testing
file, `test_round7_ab_testing_realistic_data.py` — both are real, neither is redundant, see each
docstring for what's distinct).

| Round | Files | Lens |
|---|---|---|
| 3 | `test_round3_reviews.py` | 5 personas: M-competition organizer, applied practitioner, strict IJF referee, AMIP-literature researcher, government/policy adopter |
| 5 | `test_round5_competition_integrity.py` | Adversarial-gaming: can a strategic participant exploit this as official competition methodology |
| 6 | `test_ab_testing_domain_review.py`, `test_round6_clinical_biostatistics.py`, `test_round6_financial_adversarial.py`, `test_round6_foundation_model_researcher.py`, `test_round6_production_scale.py`, `test_round6_recsys_cross_domain.py` | 6 of 8 round-6 personas (forecasting-research + cross-domain), each a distinct field |
| 7 | `test_round7_ab_testing_realistic_data.py`, `test_round7_clinical_realistic_data.py`, `test_round7_government_policy_realistic_data.py`, `test_round7_llm_leaderboard_realistic_data.py` | Realistic long-tenure/high-volatility data, deeper than round 6's illustrative examples |
| 8 | `test_round8_concurrency_safety.py`, `test_round8_formula_rederivation.py`, `test_round8_losspanel_completeness.py`, `test_round8_type_annotations.py`, `test_round8_energy_grid_realistic_data.py`, `test_round8_epidemiology_realistic_data.py`, `test_round8_insurance_actuarial_realistic_data.py`, `test_round8_manufacturing_realistic_data.py`, `test_round8_marketing_attribution_realistic_data.py`, `test_round8_sports_analytics_realistic_data.py` | 4 code-validation lenses (concurrency, exhaustive math re-derivation, a `LossPanel`-completeness bugfix, static typing) + 6 new domains |
| 9 | `test_round9_agriculture_realistic_data.py`, `test_round9_climate_weather_realistic_data.py`, `test_round9_election_polling_realistic_data.py`, `test_round9_fraud_detection_realistic_data.py`, `test_round9_real_estate_realistic_data.py`, `test_round9_subgroup_heterogeneity_realistic_data.py` | 6 more new domains (round 9's expert-reviewer files land in top-level docs, not here — see `docs/POSITIONING.md`/README/EVALUATION_CARD.md for those findings) |

## Finding a specific test fast

- **"Is function X tested for property Y?"** — `grep -rl "def.*X" tests/` narrows fast, but the
  `test_stageN_*` files are the primary/most exhaustive coverage per function; the `test_roundN_*`
  files add scenario-specific edge cases on top, not a second copy of the basics.
- **"Was domain/persona Z already stress-tested?"** — check the round tables above first; if Z isn't
  listed, it hasn't been (add a new `test_roundN_<domain>_realistic_data.py` file when it is).
- **"Why does this test exist / what bug does it guard?"** — every test file's own module docstring
  states this explicitly; read that before the code.
