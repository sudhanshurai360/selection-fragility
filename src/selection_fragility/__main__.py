"""python -m selection_fragility -- CLI entry point. Currently one subcommand: `compare`, so
`compare()` can gate a CI/pipeline promotion step without writing any Python."""
import argparse
import json
import sys
from .panel import LossPanel
from .compare import compare
def main(argv=None):
    # FIXED 2026-09-10 (round-7 final pre-publish review, cross_platform_encoding lens): with no
    # encoding safeguard, a model/label name containing characters outside stdout's codepage (CJK,
    # emoji -- not just accented Latin, which encodes fine under cp1252) crashed with an unhandled
    # UnicodeEncodeError whenever stdout isn't UTF-8, the DEFAULT on Windows once output is
    # redirected/piped -- exactly the CI-capture scenario this CLI exists for. That crash exited 1
    # with a raw traceback, indistinguishable from the internal-bug case the round-2/round-6 fixes
    # specifically eliminated for every other failure path. reconfigure(errors=...) degrades to a
    # readable escaped form instead of crashing; guarded by hasattr since not every stdout-like
    # object supports it (e.g. under pytest's capture).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="selection_fragility")
    sub = parser.add_subparsers(dest="command", required=True)
    p_compare = sub.add_parser("compare", help="compare two saved LossPanels (previous vs current run)")
    p_compare.add_argument("previous", help="path to a LossPanel saved with .save()")
    p_compare.add_argument("current", help="path to a LossPanel saved with .save()")
    p_compare.add_argument("--alpha", type=float, default=0.10, help="Model Confidence Set level")
    p_compare.add_argument("--exit-code", action="store_true",
                            help="exit 1 when the report recommends action (.act=True); exit 0 otherwise")
    args = parser.parse_args(argv)
    if args.command == "compare":
        # FIXED 2026-08-26 (round-2 review, "does it deliver on its promises" lens): CHANGELOG.md
        # already claimed "every CLI error path... exits 2", but nothing here ever caught anything --
        # a missing/malformed file surfaced as a raw Python traceback at exit 1 (Python's default for
        # an unhandled exception), indistinguishable from a genuine internal bug. Catch only the
        # user-input-class failures LossPanel.load()/compare() are documented to raise for bad data
        # (a missing file, corrupt/non-LossPanel JSON, or a real data problem like a mismatched model
        # set) and exit 2 with a clean one-line message; anything else (a real programming error) is
        # deliberately left to propagate as an uncaught exception so it isn't mistaken for bad input.
        #
        # WIDENED 2026-09-10 (round-6 stress-review, cli_end_to_end lens): the fix above only caught
        # FileNotFoundError, not its OSError siblings -- pointing `compare` at a directory (a very
        # plausible path mistake) raised IsADirectoryError, and an unreadable file (plausible under a
        # CI runner's restrictive permissions) raised PermissionError, both as a raw traceback at
        # exit 1 -- the SAME exit code --exit-code uses to mean "champion changed, act on it,"
        # exactly the ambiguity this whole except-block exists to eliminate. FileNotFoundError is
        # itself an OSError subclass, so catching OSError directly covers every everyday file-access
        # failure from the bare open() inside LossPanel.load() without narrowing anything already
        # caught.
        try:
            prev = LossPanel.load(args.previous)
            curr = LossPanel.load(args.current)
            r = compare(prev, curr, alpha=args.alpha)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"selection_fragility compare: {e}", file=sys.stderr)
            sys.exit(2)
        print(r)
        if args.exit_code and r.act:
            sys.exit(1)
        sys.exit(0)
if __name__ == "__main__":
    main()
