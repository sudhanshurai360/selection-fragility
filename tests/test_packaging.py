"""Packaging/documentation-accuracy tests (round-2 review, 2026-08-26).

These guard against the exact defect class round 2 found: a README/CHANGELOG claim about
install requirements, dependencies, or fixed behavior that quietly stops matching the shipped
package. Nothing here tests statistical correctness -- see the stage test files for that.
"""
import pathlib
import re

try:
    import tomllib  # stdlib, Python >=3.11
except ModuleNotFoundError:
    import tomli as tomllib  # backport, Python 3.10

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _pyproject():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _readme_text():
    return (ROOT / "README.md").read_text(encoding="utf-8")


# ---- direct-dependency claim in the README must match pyproject.toml --------------------------

def test_readme_dependency_list_matches_pyproject():
    """README's 'Direct dependencies' line must name exactly pyproject.toml's declared deps.

    Round-2 review found the README claiming 'Only dependency: numpy' while a real clean install
    pulls in 9 packages (4 direct: numpy/pandas/arch/scipy, plus their own transitive deps). This
    locks the README's claim to pyproject.toml's actual `dependencies` list so the two can't drift
    apart silently again.
    """
    declared = _pyproject()["project"]["dependencies"]
    direct_names = sorted(re.match(r"[A-Za-z0-9_.-]+", d).group(0).lower() for d in declared)

    readme = _readme_text()
    m = re.search(r"Direct dependencies:\s*(.+?)\.", readme)
    assert m, "README no longer has a 'Direct dependencies: ...' line to check"
    named_in_readme = sorted(
        n.strip("`").lower() for n in re.findall(r"`([A-Za-z0-9_.-]+)`", m.group(1))
    )
    assert named_in_readme == direct_names, (
        f"README's direct-dependency list {named_in_readme} does not match pyproject.toml's "
        f"declared dependencies {direct_names} -- one of them changed without the other"
    )


def test_readme_requires_python_matches_pyproject():
    """README's stated Python floor must match pyproject.toml's requires-python."""
    declared = _pyproject()["project"]["requires-python"]
    m = re.search(r"requires-python", declared)
    floor = re.search(r">=(\d+\.\d+)", declared).group(1)
    readme = _readme_text()
    assert f">={floor}" in readme, (
        f"README does not mention the actual Python floor ({declared}) anywhere -- "
        f"round-2 review found this drifted before (claimed 3.9, actually needed 3.10)"
    )


# ---- debris-file guard -------------------------------------------------------------------------

def test_no_leftover_debris_files_in_package_tree():
    """Fail the build if any .bak*/.recovered* file exists in src/, tests/, or examples/.

    This exact class of leftover (from an earlier incomplete reassembly) was cleaned up twice this
    project: 14 files in the initial tool-phase fix round, then one more
    (tests/conftest.py.gen_tuple_backup) found and removed in round 2. This guard exists so a third
    instance doesn't silently ship.
    """
    debris_patterns = ("*.bak*", "*.recovered*", "*.recovered_old")
    found = []
    for sub in ("src", "tests", "examples"):
        d = ROOT / sub
        if not d.exists():
            continue
        for pattern in debris_patterns:
            found.extend(str(p.relative_to(ROOT)) for p in d.rglob(pattern))
    assert not found, f"leftover debris files found, should have been removed: {found}"


# ---- the loss/accuracy sign-confusion warning must stay documented ----------------------------

def test_sign_confusion_warning_documented_in_readme():
    """The 'feeding accuracy into a loss API silently inverts the winner' warning must survive.

    Round-2 review found this is the one misuse case (of 4 tried) that produces a confidently
    WRONG answer with no error or warning -- worse than a crash, and a very plausible real mistake
    (loss/accuracy sign confusion is common). The library can't detect this automatically (it has
    no way to know a metric's direction), so the only guard is documentation. This test only
    checks the documentation doesn't quietly disappear; it cannot catch the underlying misuse
    itself.
    """
    readme = _readme_text().lower()
    assert "lower is better" in readme
    assert "inverted" in readme and "accuracy" in readme, (
        "the loss-vs-accuracy sign-confusion warning appears to be missing from README.md"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
