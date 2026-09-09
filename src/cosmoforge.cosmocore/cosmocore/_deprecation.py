"""Shared machinery for one-release deprecated aliases (ADR-0018).

A leaf module on purpose: ``core`` imports ``basis``, so anything ``basis``
also needs cannot live in ``core``.
"""

from __future__ import annotations

import warnings


def resolve_alias(new, old, new_name: str, old_name: str, *, stacklevel: int = 3):
    """Return whichever of the two spellings the caller used, warning on the old.

    ``None`` is the "not given" sentinel on both, so a parameter whose real
    default is not ``None`` defaults to ``None`` in the signature and resolves
    afterwards. Passing both is an error rather than a silent precedence rule.

    Parameters
    ----------
    new, old : object
        The value under the current name and under the deprecated one.
    new_name, old_name : str
        The two spellings, as they appear in the messages.
    stacklevel : int, default 3
        Frames between :func:`warnings.warn` and the user's call. Three suits
        a direct caller; add one per wrapper in between.
    """
    if old is None:
        return new
    if new is not None:
        raise TypeError(
            f"pass only {new_name}=; {old_name}= is the deprecated alias for it"
        )
    warnings.warn(
        f"{old_name}= is deprecated; use {new_name}= (ADR-0018)",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return old
