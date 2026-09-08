"""Round-8 static-typing-audit finding, fixed: condorcet_winner/prop22_certifies/concentration_share/
pivot_agreement can each legitimately return None per their own accurate docstrings, but none of their
signatures declared this -- a caller trusting an IDE/type-checker instead of reading the full docstring
would not have been warned. Locked in here so the annotations can't silently regress."""
import inspect
import typing

from selection_fragility.fragility import condorcet_winner
from selection_fragility.prop22 import prop22_certifies
from selection_fragility.pivot import concentration_share, pivot_agreement


def _is_optional(annotation) -> bool:
    return typing.get_origin(annotation) in (typing.Union, getattr(__import__("types"), "UnionType", None)) \
        and type(None) in typing.get_args(annotation)


def test_public_none_returning_functions_are_type_annotated():
    for fn in (condorcet_winner, prop22_certifies, concentration_share, pivot_agreement):
        ret = inspect.signature(fn).return_annotation
        assert ret is not inspect.Signature.empty, f"{fn.__name__} has no return annotation at all"
        assert _is_optional(ret), (
            f"{fn.__name__}'s return annotation ({ret!r}) does not declare it can return None, "
            "even though its own docstring documents real None-returning cases"
        )
