---
name: selection-fragility bug report
about: Report a bug in the selection-fragility package
title: "[selection-fragility] "
labels: bug
---

<!-- ADDED 2026-09-07 (round-4 8-lens PyPI-preflight audit, CONFIRMED LOW, found alongside the same-
class CI-workflow gap fixed the same day -- see .github/workflows/selection-fragility-test.yml's own
header comment for the full story). CONTRIBUTING.md points readers at
`.github/ISSUE_TEMPLATE/bug_report.md`, but the only template that existed lived in the private
monorepo's own `.github/ISSUE_TEMPLATE/selection_fragility_bug_report.md` -- a dead reference from
both a PyPI download and this package's own eventual standalone repo, the same way the CI workflow
was. This is a repo-root-adapted copy (renamed to match CONTRIBUTING.md's stated filename exactly,
and with the `about:` line's release/selection-fragility/-relative path removed, since once this is
its own repo there is no such subdirectory). -->

**Environment**
- `selection-fragility` version (`pip show selection-fragility`):
- Python version (`python --version`):
- OS (Linux / Windows / macOS, and version):

**Minimal reproducible example**

```python
# A plain Python snippet that fails -- not a description of the failure in prose.
```

**Expected behavior**

**Actual behavior** (full traceback if there is one)

**Have you checked whether a sibling function has the same issue?**
Several real bugs in this project's history affected one function but not its closest siblings (or vice versa)
— see CONTRIBUTING.md's "known code shapes" section. If you've checked, say what you found; if not, no need to.
