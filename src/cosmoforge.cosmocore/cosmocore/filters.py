"""Pixel-space filters, recorded as restriction to the operator's range.

A linear pixel-space filter ``F`` enters the estimator as a *restriction*: the
analysis is carried out in the coordinates ``Σ Wᵀ d`` on ``range(F)``, never by
regularising a singular ``F C Fᵀ``. Conjugated QML and the pixel likelihood are
invariant under invertible maps of the data, so only the subspace and the
grading matter, and both are read off the thin SVD ``F = U Σ Wᵀ`` truncated at
a user-chosen rank.

The :class:`Filter` record here holds that truncated SVD and nothing else.
Operators that build one (band nulling, scan-template deprojection, transfer
functions) live in the constructors of this module. See ADR-0020.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from hashlib import blake2b

import numpy as np

from .geometry import active_pixel_index

#: Hard floor on ``range_epsilon``. The restricted covariance carries ``Σ²``,
#: so its condition number goes as ``(σ₁/σ_r)²``; a machine-precision cut would
#: put a graded operator at condition ~1e26. sqrt(eps) keeps the square inside
#: double precision.
MIN_RANGE_EPSILON = float(np.sqrt(np.finfo(np.float64).eps))

#: Warn when the kept/discarded singular-value ratio exceeds this, i.e. when the
#: truncation does not sit in a clean spectral gap.
GAP_WARN_RATIO = 0.1

#: Tolerance for calling a set of singular values "all one", i.e. a projector.
_PROJECTOR_ATOL = 1e-10


def mask_fingerprint(mask):
    """
    Fingerprint the active-pixel ordering a filter was built against.

    Parameters
    ----------
    mask : numpy.ndarray
        Analysis mask, shape ``(npix, ncomponents)``.

    Returns
    -------
    bytes
        16-byte blake2b digest of :func:`~cosmocore.geometry.active_pixel_index`.
        Deliberately not the builtin ``hash``, which is salted per process and
        would differ across MPI ranks.
    """
    return blake2b(active_pixel_index(mask).tobytes(), digest_size=16).digest()


def _complement(A, epsilon):
    """Orthonormal basis of the complement of ``range(A)``, and of ``range(A)``.

    Returns ``(kept, removed)`` where ``removed`` spans the leading directions
    of ``A`` above the relative cut ``epsilon`` and ``kept`` spans everything
    orthogonal to them.
    """
    U, s, _ = np.linalg.svd(A, full_matrices=True)
    n_removed = int(np.count_nonzero(s > epsilon * s[0])) if s.size else 0
    if n_removed >= A.shape[0]:
        raise ValueError(
            f"the {A.shape[1]} columns handed in span all {A.shape[0]} pixels "
            f"above epsilon={epsilon:g}, leaving a rank-0 filter; the geometry "
            "cannot support this many independent templates"
        )
    return np.ascontiguousarray(U[:, n_removed:]), np.ascontiguousarray(U[:, :n_removed])


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Filter:
    """A pixel-space filter as its truncated thin SVD ``F = U Σ Wᵀ``.

    Attributes
    ----------
    W : numpy.ndarray
        ``(n_pixels, rank)`` orthonormal domain-side basis. Raw pixel-space
        inputs are restricted as ``Σ Wᵀ x``.
    U : numpy.ndarray
        ``(n_pixels, rank)`` orthonormal range-side basis. Inputs that already
        carry the filter are restricted as ``Uᵀ x``. For a projector this *is*
        ``W`` (the same object), so the two conventions agree by construction.
    sigma : numpy.ndarray
        ``(rank,)`` kept singular values, descending. All ones for a projector.
    rank : int
        Working dimension ``r`` of every restricted object.
    is_projector : bool
        Stored, not inferred at read time: consumers branch on it.
    range_epsilon, range_rank : float or None, int or None
        The knob that set the truncation. Exactly one is not ``None``.
        ``range_epsilon`` records the *effective* cut, after any clipping to
        :data:`MIN_RANGE_EPSILON`.
    Q_removed : numpy.ndarray or None
        ``(n_pixels, n_pixels - rank)`` basis of the removed subspace, kept so
        that intersection is complement-of-union rather than principal angles
        on an ``n × n`` product. ``None`` when the removed directions were not
        formed (graded filters).
    fingerprint : bytes or None
        Digest of the active-pixel ordering this filter was built against, or
        ``None`` when no mask was supplied. Checked once, in
        ``Core.setup_covariance_matrices``.
    n_pixels : int
        Pixel-space dimension ``n`` the filter expects.

    Notes
    -----
    The discarded singular values are not stored and are not recoverable: ``F``
    itself is never retained.
    """

    W: np.ndarray
    U: np.ndarray
    sigma: np.ndarray
    rank: int
    is_projector: bool
    range_epsilon: float | None
    range_rank: int | None
    Q_removed: np.ndarray | None
    fingerprint: bytes | None
    n_pixels: int

    def __post_init__(self):
        if self.W.ndim != 2:
            raise ValueError(f"W must be 2-D, got shape {self.W.shape}")
        if self.W.shape != (self.n_pixels, self.rank):
            raise ValueError(
                f"W has shape {self.W.shape}, expected ({self.n_pixels}, {self.rank})"
            )
        if self.U.shape != self.W.shape:
            raise ValueError(
                f"U has shape {self.U.shape}, expected {self.W.shape} (same as W)"
            )
        if self.sigma.shape != (self.rank,):
            raise ValueError(
                f"sigma has shape {self.sigma.shape}, expected ({self.rank},)"
            )
        if self.is_projector and self.U is not self.W:
            raise ValueError(
                "a projector's U must be the same object as W, so that raw and "
                "pre-filtered inputs land in identical coordinates"
            )

    def __repr__(self):
        kind = "projector" if self.is_projector else "graded"
        knob = (
            f"range_rank={self.range_rank}"
            if self.range_rank is not None
            else f"range_epsilon={self.range_epsilon:g}"
        )
        return f"Filter({kind}, n_pixels={self.n_pixels}, rank={self.rank}, {knob})"

    @property
    def scaling(self):
        """``sigma``, or ``None`` for a projector so callers can skip the scaling."""
        return None if self.is_projector else self.sigma

    @classmethod
    def _from_factors(
        cls,
        W,
        U,
        sigma,
        rank,
        is_projector,
        range_epsilon,
        range_rank,
        Q_removed,
        fingerprint,
        n_pixels,
    ):
        """Rebuild a record from already-validated factors, skipping every check.

        For MPI transport, where each rank attaches to a shared-memory ``W``
        that the validating path would re-check for no reason.
        """
        obj = object.__new__(cls)
        for name, value in (
            ("W", W),
            ("U", U),
            ("sigma", sigma),
            ("rank", rank),
            ("is_projector", is_projector),
            ("range_epsilon", range_epsilon),
            ("range_rank", range_rank),
            ("Q_removed", Q_removed),
            ("fingerprint", fingerprint),
            ("n_pixels", n_pixels),
        ):
            object.__setattr__(obj, name, value)
        return obj

    @classmethod
    def from_operator(cls, F, *, range_epsilon=None, range_rank=None, mask=None):
        """
        Build a filter from a dense linear operator via its thin SVD.

        Parameters
        ----------
        F : numpy.ndarray
            ``(n, n)`` pixel-space operator.
        range_epsilon : float, optional
            Relative cut on ``F``'s own singular values, ``σ_i > ε σ_1``.
            Clipped to :data:`MIN_RANGE_EPSILON` with a warning.
        range_rank : int, optional
            Exact number of singular values to keep. Mutually exclusive with
            ``range_epsilon``; one of the two is required.
        mask : numpy.ndarray, optional
            Analysis mask, used only to record the active-pixel fingerprint.

        Returns
        -------
        Filter
        """
        if range_epsilon is None and range_rank is None:
            raise ValueError(
                "a rank threshold is required: pass range_epsilon or range_rank. "
                "There is no default because the right cut depends on the "
                "operator's spectrum, and the restricted covariance carries Σ²"
            )
        if range_epsilon is not None and range_rank is not None:
            raise ValueError(
                "range_epsilon and range_rank are mutually exclusive; pass one"
            )

        F = np.asarray(F, dtype=np.float64)
        if F.ndim != 2:
            raise ValueError(f"F must be 2-D, got shape {F.shape}")
        n = F.shape[0]

        U_f, s, Wt = np.linalg.svd(F, full_matrices=False)

        effective_epsilon = None
        if range_rank is not None:
            k = int(range_rank)
            if not 1 <= k <= s.size:
                raise ValueError(f"range_rank={k} outside 1..{s.size} for this operator")
        else:
            eps = float(range_epsilon)
            if eps < MIN_RANGE_EPSILON:
                warnings.warn(
                    f"range_epsilon={eps:g} is below MIN_RANGE_EPSILON="
                    f"{MIN_RANGE_EPSILON:g} and has been clipped: the restricted "
                    "covariance carries Σ², so a smaller cut puts it beyond "
                    "double precision",
                    stacklevel=2,
                )
                eps = MIN_RANGE_EPSILON
            effective_epsilon = eps
            k = int(np.count_nonzero(s > eps * s[0]))
            if k == 0:
                raise ValueError(
                    f"no singular value of F exceeds {eps:g} × σ₁; the filter "
                    "would have rank 0"
                )

        cls._warn_on_truncation(s, k)

        sigma = np.ascontiguousarray(s[:k])
        is_projector = bool(np.allclose(sigma, 1.0, atol=_PROJECTOR_ATOL))
        W = np.ascontiguousarray(Wt[:k].T)
        if is_projector:
            # U aliases W: same subspace, and crucially the same rotation
            # within it, so Σ Wᵀ and Uᵀ are the same map.
            U = W
            Q_removed = np.ascontiguousarray(Wt[k:].T) if Wt.shape[0] == n else None
        else:
            U = np.ascontiguousarray(U_f[:, :k])
            Q_removed = None

        return cls(
            W=W,
            U=U,
            sigma=sigma,
            rank=k,
            is_projector=is_projector,
            range_epsilon=effective_epsilon,
            range_rank=None if range_rank is None else int(range_rank),
            Q_removed=Q_removed,
            fingerprint=None if mask is None else mask_fingerprint(mask),
            n_pixels=n,
        )

    @classmethod
    def from_deprojection(cls, T, *, template_epsilon=1e-3, mask=None):
        """
        Build the projector that annihilates a stack of templates.

        Parameters
        ----------
        T : numpy.ndarray
            ``(n, n_templates)`` stack of pixel-space templates to remove.
        template_epsilon : float
            Relative cut on ``T``'s singular values, deciding how many template
            directions are genuinely independent.
        mask : numpy.ndarray, optional
            Analysis mask, used only to record the active-pixel fingerprint.

        Returns
        -------
        Filter
            A projector, ``rank = n - (independent template directions)``.

        Notes
        -----
        Template column norms are meaningful within one family and
        incommensurable across families, so a single relative cut on a stacked
        matrix can delete a weaker family wholesale. Build one filter per
        family and combine them with ``&``.
        """
        T = np.asarray(T, dtype=np.float64)
        if T.ndim != 2:
            raise ValueError(f"T must be 2-D (n_pixels, n_templates), got {T.shape}")
        n = T.shape[0]
        W, Q_removed = _complement(T, float(template_epsilon))
        return cls(
            W=W,
            U=W,
            sigma=np.ones(W.shape[1]),
            rank=W.shape[1],
            is_projector=True,
            range_epsilon=None,
            range_rank=W.shape[1],
            Q_removed=Q_removed,
            fingerprint=None if mask is None else mask_fingerprint(mask),
            n_pixels=n,
        )

    @classmethod
    def from_subspace(cls, W, *, mask=None):
        """
        Build the projector onto an already-orthonormal subspace.

        Parameters
        ----------
        W : numpy.ndarray
            ``(n, r)`` orthonormal basis of the subspace to keep. Orthonormality
            is the caller's contract; only the column norms are checked.
        mask : numpy.ndarray, optional
            Analysis mask, used only to record the active-pixel fingerprint.

        Returns
        -------
        Filter
        """
        W = np.ascontiguousarray(W, dtype=np.float64)
        if W.ndim != 2:
            raise ValueError(f"W must be 2-D (n_pixels, rank), got {W.shape}")
        if W.shape[1] > W.shape[0]:
            raise ValueError(
                f"W has {W.shape[1]} columns for {W.shape[0]} pixels; a subspace "
                "cannot exceed the space"
            )
        norms = np.linalg.norm(W, axis=0)
        if not np.allclose(norms, 1.0, atol=1e-8):
            raise ValueError(
                "W's columns must be orthonormal; column norms range over "
                f"[{norms.min():g}, {norms.max():g}]"
            )
        return cls(
            W=W,
            U=W,
            sigma=np.ones(W.shape[1]),
            rank=W.shape[1],
            is_projector=True,
            range_epsilon=None,
            range_rank=W.shape[1],
            Q_removed=None,
            fingerprint=None if mask is None else mask_fingerprint(mask),
            n_pixels=W.shape[0],
        )

    @staticmethod
    def _warn_on_truncation(s, k):
        """Warn when the cut does not sit in a spectral gap, or does not cut."""
        if k < s.size:
            gap = s[k] / s[k - 1]
            if gap > GAP_WARN_RATIO:
                largest = int(np.argmin(s[1:] / s[:-1])) + 1
                warnings.warn(
                    f"filter truncated at rank {k} without a spectral gap: "
                    f"σ_{k + 1}/σ_{k} = {gap:.3g}. The largest gap in the "
                    f"spectrum is at rank {largest} "
                    f"(ratio {s[largest] / s[largest - 1]:.3g})",
                    stacklevel=3,
                )
        else:
            cond = s[0] / s[-1]
            warnings.warn(
                f"filter kept every singular value (rank {k}); it is invertible "
                f"and restriction leaves the estimator unchanged, but the "
                f"restricted covariance carries condition {cond**2:.3g}",
                stacklevel=3,
            )

    def __and__(self, other):
        """Intersect two projectors' kept subspaces. See :func:`intersect`."""
        return intersect(self, other)


def intersect(f1, f2):
    """
    Intersect two projectors: keep only what both keep.

    Parameters
    ----------
    f1, f2 : Filter
        Projectors over the same pixel space, both carrying ``Q_removed``.

    Returns
    -------
    Filter
        Projector onto ``range(W1) ∩ range(W2)``, computed as the complement of
        the union of the removed subspaces (``k ≪ n`` columns) rather than by
        principal angles on an ``n × n`` product.

    Raises
    ------
    ValueError
        If either filter is graded: a transfer function is not a subspace, so
        there is nothing to intersect.
    """
    for f in (f1, f2):
        if not f.is_projector:
            raise ValueError(
                "intersection is defined for projectors only; a graded filter "
                "is a transfer function, not a subspace"
            )
        if f.Q_removed is None:
            raise ValueError(
                "intersection needs Q_removed, which from_subspace does not "
                "form; build the operands with from_deprojection"
            )
    if f1.n_pixels != f2.n_pixels:
        raise ValueError(
            f"pixel-space dimensions differ: {f1.n_pixels} and {f2.n_pixels}"
        )
    if (
        f1.fingerprint is not None
        and f2.fingerprint is not None
        and f1.fingerprint != f2.fingerprint
    ):
        raise ValueError(
            "the two filters were built against different active-pixel orderings"
        )

    stack = np.column_stack([f1.Q_removed, f2.Q_removed])
    W, Q_removed = _complement(stack, MIN_RANGE_EPSILON)
    return Filter(
        W=W,
        U=W,
        sigma=np.ones(W.shape[1]),
        rank=W.shape[1],
        is_projector=True,
        range_epsilon=None,
        range_rank=W.shape[1],
        Q_removed=Q_removed,
        fingerprint=f1.fingerprint or f2.fingerprint,
        n_pixels=f1.n_pixels,
    )
