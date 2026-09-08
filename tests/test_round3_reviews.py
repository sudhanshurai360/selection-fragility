"""Round-3 persona-review guard tests (2026-08-26).

Five personas (M-competition organizer, applied practitioner, strict IJF referee, an
AMIP-literature-aware researcher, a government/policy domain adopter) stress-tested the shipped
package end to end, in isolation, each against a scenario realistic to their role. These tests lock
in the concrete, checkable findings from that round. Nothing here re-tests round-1/round-2 findings
-- see test_packaging.py for those.
"""
import pathlib
import re
import time

import numpy as np
import pytest

from selection_fragility import LossPanel, compare, model_confidence_set

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent.parent  # Paper3_UIClaims_Forecasting/ -- some docs reference paths there


# ---- 1. performance-regression guard (M-competition-veteran finding) --------------------------

def test_mcs_performance_at_competition_scale():
    """model_confidence_set must stay fast at the scale that actually matters for adoption.

    The round-3 M-competition-organizer persona measured ~62s for model_confidence_set alone at
    T=100,000/K=6/B=2000 (M4 scale), ~77s end to end including report(). That's fine for a
    one-time post-competition job. This test uses a smaller T=10,000 slice with a generous ceiling
    so it stays fast enough for routine CI while still catching an actual O(T^2)-class regression,
    which would blow well past the ceiling even at this reduced scale.

    CEILING RAISED 2026-08-27 (first real GitHub Actions run, 12-job Linux/Windows/macOS matrix):
    20s wasn't generous enough for shared CI hardware -- ubuntu-latest took 20.5s, windows-latest
    24.2s, both failing a ceiling calibrated only against one fast local macOS dev machine. 60s
    gives real headroom over the worst observed CI time while still catching a genuine O(T^2)-class
    regression, which would blow past even 60s by a large multiple.
    """
    rng = np.random.default_rng(0)
    T, K = 10_000, 6
    L = {f"m{i}": rng.normal(1.0, 0.3, T) for i in range(K)}
    t0 = time.perf_counter()
    survivors, p = model_confidence_set(L, alpha=0.10, B=2000, seed=0)
    elapsed = time.perf_counter() - t0
    assert 1 <= len(survivors) <= K
    assert elapsed < 60.0, (
        f"model_confidence_set(T=10_000, K=6, B=2000) took {elapsed:.1f}s (ceiling 60s) -- "
        f"a regression here would make the package impractical at real competition/agency scale"
    )


# ---- 2. many-near-tied-models guard (M-competition-veteran finding) ---------------------------

def test_mcs_many_near_tied_models_returns_plausible_multi_survivor_set():
    """A real top-heavy leaderboard (many models clustered near the best) should not spuriously
    collapse to a singleton MCS, and decision_breakdown's tie-handling should surface more than one
    binding opponent -- currently the only tie case exercised anywhere in the suite is a 2-model
    tie (test_mixed_exact_tie_survivor_set_invariant_to_column_order in test_stage1_identified.py)."""
    from selection_fragility.fragility import decision_breakdown

    rng = np.random.default_rng(3)
    T, K_tied, K_worse = 60, 10, 2
    tied_mean = 1.0
    L = {}
    for i in range(K_tied):
        L[f"tied{i}"] = rng.normal(tied_mean, 0.25, T)
    for i in range(K_worse):
        L[f"worse{i}"] = rng.normal(tied_mean + 1.5, 0.25, T)

    survivors, _p = model_confidence_set(L, alpha=0.10, B=1000, seed=0)
    assert len(survivors) >= 2, (
        f"a {K_tied}-way genuinely-tied cluster against {K_worse} clearly worse models collapsed "
        f"to a singleton MCS ({survivors}) -- this is the exact scenario a top-heavy competition "
        f"leaderboard produces and a spurious singleton here would be actively misleading"
    )
    assert set(survivors) <= {f"tied{i}" for i in range(K_tied)}

    w = np.ones(T)
    champ = min(L, key=lambda m: float(np.mean(L[m])))
    _k, _opp, _removed, ties = decision_breakdown(L, w, a=champ, return_ties=True)
    assert ties is not None and len(ties) >= 1, (
        "decision_breakdown(..., return_ties=True) on a many-way-tied cluster should surface at "
        "least one binding tied opponent, not silently report none"
    )


# ---- 3. Nixtla-CV-frame positive control (M-competition-veteran finding) ----------------------

def test_from_forecasts_ingests_nixtla_style_wide_cv_frame():
    """LossPanel.from_forecasts() must keep ingesting the exact wide cross-validation shape
    (unique_id/cutoff/y/one-column-per-model) that StatsForecast/NeuralForecast produce -- the
    de facto standard tooling shape and, per the M-competition-organizer persona, the actual
    on-ramp for real adoption. Locked in as a positive control so this never regresses silently."""
    import pandas as pd

    rng = np.random.default_rng(5)
    n_series, n_folds = 50, 4
    rows = []
    for sid in range(n_series):
        for cutoff in range(n_folds):
            y = float(rng.normal(100, 10))
            rows.append({
                "unique_id": f"series_{sid}",
                "cutoff": f"2024-{cutoff + 1:02d}",
                "y": y,
                "model_a": y + float(rng.normal(0, 2)),
                "model_b": y + float(rng.normal(0, 5)),
            })
    df = pd.DataFrame(rows)

    panel = LossPanel.from_forecasts(df, y_true="y", period="cutoff", group="unique_id",
                                      models=["model_a", "model_b"])
    assert set(panel.models) == {"model_a", "model_b"}
    assert len(panel.labels) == n_folds


# ---- 4. doc-claims-existence guard (strict-IJF-referee finding) -------------------------------

_DISCLAIM_PHRASES = (
    "does not exist", "doesn't exist", "neither file exists", "no longer exist",
    "was never built", "never shipped", "removed", "not yet", "aspirational",
)

def _paragraphs(text):
    return re.split(r"\n\s*\n", text)


def _referenced_paths(paragraph):
    return re.findall(r"`([A-Za-z0-9_./\-]+\.(?:py|md|json|toml|cfg|ya?ml|txt|csv))`", paragraph)


def test_documented_file_paths_exist_or_are_explicitly_disclaimed():
    """Every file path README.md/EVALUATION_CARD.md cite as something a reader can go run or open
    must actually exist -- either in this package or (for paths explicitly scoped to 'the paper
    repo') one level up. A path is exempt only if its own paragraph explicitly says it does not
    exist (the FROZEN_INPUTS.md/frozen_manifest.py case this test is modeled on: EVALUATION_CARD.md
    used to claim these were runnable/present when they never were; the current text instead
    discloses their absence honestly, and this test must not re-flag that honest disclosure).

    Strict IJF-referee persona (round 3): reproduced the exact failure a referee hits running
    EVALUATION_CARD.md's own claimed `python code/gates/frozen_manifest.py --check` command --
    the file did not exist. This guards against a new dead reference shipping the same way.
    """
    problems = []
    for doc_name in ("README.md", "EVALUATION_CARD.md"):
        text = (ROOT / doc_name).read_text(encoding="utf-8")
        for para in _paragraphs(text):
            paths = _referenced_paths(para)
            if not paths:
                continue
            disclaimed = any(p in para.lower() for p in _DISCLAIM_PHRASES)
            if disclaimed:
                continue
            for rel in paths:
                if (ROOT / rel).exists() or (REPO_ROOT / rel).exists():
                    continue
                problems.append(f"{doc_name}: `{rel}` referenced with no disclaimer, but does not exist")
    assert not problems, "dead file reference(s) found:\n" + "\n".join(problems)


# ---- 5. CHANGELOG-claim-audit guard (strict-IJF-referee finding) ------------------------------

def _changelog_fixed_bullets():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = re.split(r"\n(?=#+ .*Fixed)", text)
    bullets = []
    for sec in sections:
        if "Fixed" not in sec.split("\n", 1)[0]:
            continue
        for m in re.finditer(r"^- \*\*.*$(?:\n(?!- \*\*|\n#).*$)*", sec, re.M):
            bullets.append(m.group(0))
    return bullets


_BUILTIN_STOPLIST = {"help", "print", "str", "repr", "len", "int", "float", "list", "dict", "set",
                     "true", "false", "none"}


_BUILTIN_EXC_RE = re.compile(r"^[A-Z][A-Za-z]*(Error|Warning|Exception)$")


def _candidate_identifiers(bullet):
    names = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)\(?\)?`", bullet)
    return [n for n in names
            if not n.endswith(("py", "md", "json")) and not n.isupper()
            and n.lower() not in _BUILTIN_STOPLIST and len(n) >= 4
            and not _BUILTIN_EXC_RE.match(n)]


def _meaningful_tokens(identifier):
    """Split a snake_case identifier into tokens worth matching on; drop short/generic ones (e.g.
    'from', 'get', 'a') that would make the substring match too permissive to mean anything."""
    return [t for t in identifier.lstrip("_").split("_") if len(t) >= 6]


def _test_function_names():
    names = []
    for f in (ROOT / "tests").glob("test_*.py"):
        names.extend(re.findall(r"def (test_\w+)", f.read_text(encoding="utf-8")))
    return names


def test_changelog_fixed_claims_have_regression_coverage():
    """Every CHANGELOG.md 'Fixed' bullet that names a specific function should have SOME test
    whose own name references it (a much stronger signal than mere source-body presence, since a
    private helper like `_disambiguate_labels` is easy to exercise only indirectly through
    report()'s output without ever being tested by name).

    This is a lighter-weight heuristic than full semantic verification (not mechanically checkable
    in general) -- it would not have caught all three CHANGELOG claims found false this session
    (the __main__.py 'errors exit 2' bullet names no specific function at all, so this heuristic
    has nothing to check there), but it DOES have real teeth for the other two: verified by hand
    that `_disambiguate_labels` matches no test name anywhere before round 2's fix (the matching
    test, test_report_pivot_disambiguates_same_month_periods, was added BY that fix), and that
    `from_forecasts` only matches a dedicated test by name
    (test_from_forecasts_partial_nan_row_warns) as of that same fix -- 23 other from_forecasts
    mentions in test source are incidental usage in unrelated tests, not evidence this specific
    claim was covered, which is exactly why this checks test NAMES, not test source bodies.
    """
    test_names_lower = " ".join(_test_function_names()).lower()
    uncovered = []
    for bullet in _changelog_fixed_bullets():
        candidates = _candidate_identifiers(bullet)
        tokens = [tok for ident in candidates for tok in _meaningful_tokens(ident)]
        if not tokens:
            continue  # nothing checkable in this bullet (e.g. no function name at all)
        if not any(tok.lower() in test_names_lower for tok in tokens):
            uncovered.append(bullet.strip().splitlines()[0][:100])
    assert not uncovered, (
        "CHANGELOG 'Fixed' bullet(s) naming a function with no matching test NAME anywhere:\n"
        + "\n".join(uncovered)
    )


# ---- 6. stakeholder-paraphrasable smoke test (applied-practitioner finding) --------------------

_JARGON_TERMS = ("MCS", "MDE", "k*", "MCB")
_GLOSS_MARKERS = ("VERDICT", "LEADERBOARD", "RESOLUTION", "PIVOT", "identified", "resolved",
                  "cannot determine", "champion")


def test_report_output_is_stakeholder_paraphrasable_smoke():
    """Loose regression guard, not a strict content check: report()'s output for a genuinely
    'not identified' panel should keep pairing any bare statistical jargon with nearby
    plain-language framing (the existing VERDICT/LEADERBOARD/RESOLUTION/PIVOT section headers and
    surrounding prose), not let raw jargon terms accumulate with zero explanatory context nearby --
    that's the applied-practitioner persona's finding: the STATISTICS are trustworthy, but a
    non-technical stakeholder currently needs a human translator for report()'s own text. This
    doesn't (and can't) assert a full plain-language mode exists; it only flags future drift
    toward MORE unexplained jargon than exists today.
    """
    from selection_fragility.report import report as _report

    rng = np.random.default_rng(7)
    T, K = 20, 3
    L = {f"m{i}": rng.normal(1.0, 0.05, T) for i in range(K)}  # tiny spread -> not identified
    panel = LossPanel.from_losses(L)
    out = _report(panel)

    assert any(term in out for term in ("VERDICT", "LEADERBOARD", "RESOLUTION", "PIVOT")), (
        "report() output no longer has its section headers -- a bigger regression than this "
        "test is meant to catch, but worth failing loudly on"
    )
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if any(term in line for term in _JARGON_TERMS):
            window = "\n".join(lines[max(0, i - 2):i + 3])
            assert any(marker in window for marker in _GLOSS_MARKERS), (
                f"jargon term found with no nearby plain-language section marker: {line!r}"
            )


# ---- 7. compare() defensibility-language lock-in (applied-practitioner finding) ----------------

def test_compare_states_old_champion_was_excluded_at_decision_time_on_reversal():
    """This is the tool's actual audit-defense value proposition (per the applied-practitioner
    persona: 'if my forecast model got challenged later... would this give me a defensible paper
    trail') and it had no dedicated test. Construct a champion reversal where the new champion was
    PREVIOUSLY EXCLUDED from the MCS, and lock in that ChangeReport communicates the old champion
    was excluded/tied at decision time -- the actual 'why didn't you switch sooner' defense."""
    rng = np.random.default_rng(11)
    T, K = 40, 3
    base = rng.normal(1.0, 0.15, (T, K))
    base[:, 0] -= 0.9   # model 0 dominant and clearly identified as champion
    prev = LossPanel.from_losses({f"m{i}": base[:, i] for i in range(K)})

    base2 = rng.normal(1.0, 0.15, (T, K))
    base2[:, 1] -= 0.9  # model 1 (previously excluded, clearly worse) now dominant
    curr = LossPanel.from_losses({f"m{i}": base2[:, i] for i in range(K)})

    r = compare(prev, curr)
    assert r.champion_changed is True
    assert r.previous_champion == "m0"
    assert r.current_champion == "m1"
    assert r.old_champion_still_in_mcs is False, (
        "constructed a hard, unambiguous reversal (old champion had a 0.9 edge, new champion has "
        "a 0.9 edge the OTHER way) -- old champion should be clearly excluded from the current MCS"
    )
    assert r.act is True
    assert "left the MCS" in repr(r), (
        "ChangeReport's own repr must state the old champion left the MCS -- this is the exact "
        "audit-defense sentence a practitioner would need to paste into a memo"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
