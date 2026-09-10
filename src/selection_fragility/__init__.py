"""selection_fragility — decision breakdown point (k*) and ranking-fragility diagnostics for
forecast model selection.

Given per-period losses for a set of candidate models, answers: is the "best model" a real,
stable choice, or a fragile artifact of a few periods?

Start with the staged v1.0 pipeline:
    from selection_fragility import LossPanel, report
    panel = LossPanel.from_losses({"ar1": ..., "nbeats": ..., "ets": ...})
    print(report(panel))

`report()` prints identification (Model Confidence Set), the leaderboard, resolution (observed
edge vs. minimum detectable edge, MCB bound), and the pivot (k*, responsible periods,
concentration). See the project's README (rendered on its PyPI page) for the full quick start and
`EVALUATION_CARD.md` (in the source repository -- CORRECTED 2026-09-07, round-4 audit: this file
does not ship inside a `pip install`, only the source repo has it) for what each diagnostic does
and does not support.

CONTRIBUTORS/EXTENDERS: `selection_fragility.fragility`, `.mcs`, `.compare`, and `.report` are
FUNCTIONS here (re-exported below), not the submodules of the same name -- `import
selection_fragility.mcs as m` gives you the `mcs()` function, not the module, so `m._block_idx`
fails confusingly. Use `from selection_fragility._internals import mcs, fragility, compare,
report` (or `panel`/`identify`/`resolution`/`prop22`/`pivot`) for a stable, non-shadowed path to
the real submodules and their private helpers. See `_internals.py` for the full explanation.
"""
from .fragility import (
    fragility,
    decision_breakdown,
    breakdown_number,
    winner_stability,
    exchangeable_benchmark,
    surprise_concentration,
    pooled_winner,
    per_period_winner,
    condorcet_winner,
    condorcet_status,
)
from .mcs import mcs, model_confidence_set

# --- Round 10 (v1.0) surface: the staged pipeline the redesign introduced ------------
from .panel import LossPanel                                            # Stage 0
from .identify import identified, mcs_size                              # Stage 1
from .resolution import (                                               # Stage 2
    resolution_report,
    minimum_detectable_edge,
    selection_regret,
    mcb_bound,
    significance_boundary,
)
from .prop22 import prop22_certifies, certified_tied_subset             # Stage 3
from .pivot import concentration_share, pivot_agreement                 # Stage 4
from .compare import compare, ChangeReport                              # Stage 5
from .report import report                                              # Stage 6

__version__ = "1.0.3"

__all__ = [
    # instrument
    "fragility", "decision_breakdown", "breakdown_number", "winner_stability",
    "exchangeable_benchmark", "surprise_concentration", "pooled_winner",
    "per_period_winner", "condorcet_winner", "condorcet_status",
    "mcs", "model_confidence_set",
    # v1.0 staged surface
    "LossPanel",
    "identified", "mcs_size",
    "resolution_report", "minimum_detectable_edge", "selection_regret", "mcb_bound",
    "significance_boundary",
    "prop22_certifies", "certified_tied_subset",
    "concentration_share", "pivot_agreement",
    "compare", "ChangeReport",
    "report",
    "__version__",
]
