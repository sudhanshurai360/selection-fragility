"""selection_fragility._internals — stable access to real submodules for contributors.

`selection_fragility/__init__.py` re-exports several public functions/classes under the SAME
name as the submodule that defines them (`from .fragility import fragility`, `from .mcs import
mcs`, `from .compare import compare`, `from .report import report`). Each of those `from .X
import X` lines rebinds the package attribute `selection_fragility.X` to the re-exported
function/class, shadowing the actual SUBMODULE that was at that attribute path a moment
earlier. `import selection_fragility.mcs as m; m._block_idx` therefore fails with a confusing
`AttributeError: 'function' object has no attribute '_block_idx'` -- `m` is the `mcs()`
function, not the `mcs` module. Found in round-5 review (2026-08-27) while testing whether a
contributor could reach package internals to build an extension.

This is deliberate, unchanged public-API behavior (`from selection_fragility import mcs` must
keep returning the function for existing users) -- not something to "fix" by breaking v1.0's
API on the eve of release. This module is the fix for CONTRIBUTORS instead: a stable, explicit,
documented path to the real submodule objects (and their private helpers) that never shadows.

Usage, instead of the confusing `import selection_fragility.mcs as m`:
    from selection_fragility._internals import mcs, fragility, compare, report
    mcs._block_idx(...)             # the real submodule, its private helpers included
    fragility._MAX_ABS_LOSS         # etc.

The other submodules (panel, identify, resolution, prop22, pivot) were never shadowed --
their re-exported names differ from the submodule name -- but are re-exposed here too, for one
consistent import path regardless of which internals a contributor needs.
"""
import sys as _sys

_PKG = __name__.rsplit(".", 1)[0]  # "selection_fragility"

# Each of these is looked up in sys.modules by dotted name, NOT via `from . import X` or
# `getattr(package, X)` -- both of those would just return whatever the (possibly-shadowed)
# package attribute currently holds. sys.modules always holds the actual module object,
# regardless of what the package's own top-level attribute of the same name was rebound to.
# Requires this module to be imported only after selection_fragility/__init__.py has already
# executed its own `from .X import ...` lines (which is when Python registers each submodule
# in sys.modules) -- true whenever a caller does `from selection_fragility._internals import
# ...`, since importing the submodule `selection_fragility._internals` always first finishes
# importing and running its parent package `selection_fragility/__init__.py`.
fragility = _sys.modules[f"{_PKG}.fragility"]
mcs = _sys.modules[f"{_PKG}.mcs"]
compare = _sys.modules[f"{_PKG}.compare"]
report = _sys.modules[f"{_PKG}.report"]
panel = _sys.modules[f"{_PKG}.panel"]
identify = _sys.modules[f"{_PKG}.identify"]
resolution = _sys.modules[f"{_PKG}.resolution"]
prop22 = _sys.modules[f"{_PKG}.prop22"]
pivot = _sys.modules[f"{_PKG}.pivot"]

__all__ = [
    "fragility", "mcs", "compare", "report", "panel", "identify", "resolution", "prop22", "pivot",
]
