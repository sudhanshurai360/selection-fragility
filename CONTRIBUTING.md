# Contributing to selection-fragility

Thanks for considering a contribution. This project was built solo, but the intent is for it to be a real,
maintained piece of open-source statistical infrastructure — external contributions are genuinely welcome.

## The one house rule: verify, don't assume

The single most distinctive thing about how this codebase has been built and reviewed is a strict discipline:
**every claim about behavior — "this function does X," "this bug is fixed," "this test passes" — is checked by
actually running it, not inferred from reading code or trusting a docstring.** This caught real bugs repeatedly:
several `CHANGELOG.md` entries documented a fix that, on direct re-verification, turned out never to have
actually shipped. If you're fixing a bug, reproduce it first with a runnable example before you touch the fix. If
you're claiming a fix works, show the before/after, not just the diff. If you're touching a function with
sibling functions (see "known code shapes" below), check whether the same issue exists in the siblings too — the
biggest bug found in this project's history was a fix applied to one function but silently missed in six
siblings with the identical shape.

## Setting up and running the tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

The suite is intentionally large (400+ tests) and organized loosely into: `test_stageN_*.py` (the core v1.0
diagnostic surface, by pipeline stage), `test_roundN_*.py` (regression tests added by a specific review round,
often testing a specific realistic domain or a specific class of bug), and a handful of thematic files
(`test_crosscutting.py`, `test_extensibility_review.py`, `test_property_based.py` for `hypothesis`-driven
fuzzing). If you're looking for existing coverage of a function, `grep` the test files for its name rather than
guessing which file it lives in.

CI runs the full suite on Linux, Windows, and macOS across Python 3.10–3.13 on every push
(`.github/workflows/selection-fragility-test.yml`, at the repository root — GitHub only discovers workflows
there, not inside this package's own directory).

## Known code shapes worth knowing before you touch related code

- **`_unwrap_panel(L, w)`** (`fragility.py`): every public function that takes a raw `{model: array}` dict should
  also transparently accept a `LossPanel` object. If you add a new public function taking a loss dict, call this
  helper first, or import the equivalent local pattern already used in `resolution.py`/`pivot.py`.
- **The degenerate-margin floor** (`_MARGIN_REL_FLOOR`, used in `fragility.py`'s `breakdown_number`,
  `pooled_winner`, and `decision_breakdown`): any comparison near an exact tie on float64 data needs a relative
  tolerance, not a strict `==`/`<`/`>` — floating-point summation is not associative, and a genuine exact tie in
  the true (real-number) sense can come out on either side of a strict comparison depending on incidental
  factors like weight scale. If you're writing a new comparison that could land on a near-tie, use the same
  pattern.
- **Atomic file writes** (`panel.py`'s `save()`): any function that writes a file another process might read
  concurrently should write to a temp file and `os.replace()` it, never write in place.
- **Windows/cross-platform gotchas already fixed once, don't reintroduce them**: always pass `encoding="utf-8"`
  explicitly to `open()`/`Path.read_text()` — the OS default differs (`cp1252` on Windows can't decode some real
  README content); build file paths with `pathlib.Path`, never manual string concatenation with `/`.

## Reporting a bug

Open a GitHub issue (see `.github/ISSUE_TEMPLATE/bug_report.md`) with: your Python version, OS, `pip show
selection-fragility` output, and a minimal reproducible example — ideally a plain Python snippet that fails, not
a description of failing behavior in prose. If you can identify which sibling functions might share the same
underlying issue (see "verify, don't assume" above), checking and reporting that too is genuinely valuable.

## Getting help

GitHub Issues is the primary support channel.
