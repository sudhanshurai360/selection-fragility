"""selection_fragility quickstart — a self-contained, synthetic demonstration.

Run: python examples/quickstart.py   (after `pip install -e .`)

We build two near-tied models over 30 periods where a single 'shock' period flips the pooled winner, then show that
(a) the MCS reports non-identification and (b) k* is small with the shock as the responsible period.
"""
import numpy as np
from selection_fragility import fragility, model_confidence_set

rng = np.random.default_rng(0)
T = 30
labels = list(range(2000, 2000 + T))

# two near-tied models + one that is only better because of a single shock period
base_a = np.abs(rng.normal(1.0, 0.15, T))
base_b = base_a + rng.normal(0.0, 0.05, T)          # ~indistinguishable from a
model_a = base_a.copy()
model_b = base_b.copy()
model_a[15] = 8.0                                    # a huge shock period where model_a does badly...
model_b[15] = 0.5                                    # ...and model_b happens to do well -> flips the pooled winner

L = {"model_a": model_a, "model_b": model_b}

print("=== Model Confidence Set (is the best model even identified?) ===")
tied, p = model_confidence_set(L, alpha=0.10, block=3, seed=0)   # `p` = p-value at which elimination stopped; the
print(f"  MCS = {tied}  (|MCS|={len(tied)} -> "                  # headline is the SET size, not p (see the card).
      f"{'point-identified' if len(tied) == 1 else 'NOT identified: a tied set'})")

print("\n=== Ranking fragility (how fragile is the pooled 'best' pick?) ===")
d = fragility(L, labels=labels)
print(f"  pooled (score) winner : {d['pooled_winner']}")
print(f"  per-period (plurality): {d['per_period_winner']}   reversal={d['reversal']}")
print(f"  k*                    : {d['k_star']}   (fewest periods to delete to flip the winner)")
print(f"  concentration         : {d['concentration']:.2f}   (>1 => one period outweighs the whole net margin)")
print(f"  responsible period(s) : {d['responsible']}")
print(f"  winner_stability      : {d['winner_stability']:.2f}   fragile={d['fragile']}")

print("\nReading: a k* of 1 with concentration > 1 and the shock year as the responsible period means the 'winner' is a")
print("minority winner propped up by a single pivotal period -- exactly the fragility kstar is built to expose.")
