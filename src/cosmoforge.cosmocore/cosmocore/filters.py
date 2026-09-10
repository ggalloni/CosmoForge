"""Pixel-space filters, recorded as restriction to the operator's range.

A linear pixel-space filter ``F`` enters the estimator as a *restriction*: the
analysis is carried out in the coordinates ``Σ Wᵀ d`` on ``range(F)``, never by
regularising a singular ``F C Fᵀ``. Conjugated QML and the pixel likelihood are
invariant under invertible maps of the data, so only the subspace and the
grading matter, and both are read off the thin SVD ``F = U Σ Wᵀ`` truncated at
a user-chosen rank.

The :class:`Filter` record here holds that truncated SVD and nothing else.
Operators that build one (harmonic deprojection, scan-template deprojection,
transfer functions) live in the constructors of this module. See ADR-0020.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from hashlib import blake2b

import healpy as hp
import numpy as np
from numpy.polynomial.legendre import legvander

from .basics import svd
from .geometry import _as_columns, active_pixel_index, active_pixels
from .spectrum_key import Slot

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

        U_f, s, Wt = svd(F)

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


def _field_columns(spins, ncomponents):
    """Component column indices of each field, in declaration order.

    A spin-0 field owns one component, a spin-2 field owns two (Q and U). This
    is the ``CONTEXT.md:97`` field-versus-component distinction: ``spins`` is
    per *field*, mask columns are per *component*, and the two counts differ
    the moment any field is spin-2.
    """
    columns, cursor = [], 0
    for spin in spins:
        width = 1 if spin == 0 else 2
        columns.append(tuple(range(cursor, cursor + width)))
        cursor += width
    if cursor != ncomponents:
        raise ValueError(
            f"spins={list(spins)} describes {cursor} component(s) "
            f"(spin-0 owns one, spin-2 owns two) but the mask has "
            f"{ncomponents}; spins is per field, mask columns are per component"
        )
    return columns


def _scan_frame(pole, npix):
    """Polar angles of every HEALPix pixel in the frame whose z axis is ``pole``.

    Returns ``(theta_scan, phi_scan)``, each of length ``npix``. The azimuth
    zero point is arbitrary, fixed by an arbitrary choice of x axis; callers
    must remove it (see :func:`_centred_phi`) rather than rely on it.
    """
    z = hp.ang2vec(float(pole[0]), float(pole[1]))
    seed = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = seed - z * (seed @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    v = np.asarray(hp.pix2vec(hp.npix2nside(npix), np.arange(npix)))
    return np.arccos(np.clip(z @ v, -1.0, 1.0)), np.arctan2(y @ v, x @ v)


def _centred_phi(phi):
    """Map a contiguous run of azimuths onto ``[-1, 1]`` for a polynomial fit.

    The azimuth branch cut is a gauge choice of :func:`_scan_frame`, and a cut
    falling inside the patch would split one contiguous scan into two, turning
    a smooth ramp into a step and the polynomial basis into nonsense. Wrapping
    about the circular mean puts the cut opposite the patch.
    """
    mean = np.arctan2(np.sin(phi).mean(), np.cos(phi).mean())
    centred = np.mod(phi - mean + np.pi, 2 * np.pi) - np.pi
    lo, hi = centred.min(), centred.max()
    if hi - lo > 0.95 * 2 * np.pi:
        warnings.warn(
            f"a stripe spans {np.degrees(hi - lo):.0f}° of azimuth, so no cut-free "
            "parameterisation of it exists and the polynomial basis is not "
            "meaningful; a scan encircling the pole is not a constant-elevation "
            "stripe",
            stacklevel=3,
        )
    if hi <= lo:
        return np.zeros_like(centred)
    return 2.0 * (centred - lo) / (hi - lo) - 1.0


def hits_weighting(hits, *, mask, range_epsilon=None, range_rank=None):
    """
    Build inverse-noise (hits) weighting as a filter: the no-op control.

    The weight is ``sqrt(hits)`` per pixel, the whitening for noise going as
    ``1 / sqrt(hits)``. It is diagonal and positive, hence invertible, and a
    conjugated estimator is invariant under invertible maps of the data, so
    restriction leaves the answer unchanged. That is the point: this is the
    control that shows the filter machinery is transparent when it should be,
    not a filter anyone needs to apply.

    Parameters
    ----------
    hits : numpy.ndarray
        Hit counts, ``(npix,)`` to share one map across components or
        ``(npix, ncomponents)`` for per-component counts.
    mask : numpy.ndarray
        Analysis mask, ``(npix, ncomponents)``. Sets the active-pixel ordering
        and is recorded as the fingerprint.
    range_epsilon, range_rank : float or None, int or None
        Rank threshold, passed through to :meth:`Filter.from_operator`. One is
        required; there is deliberately no default even here, because a real
        hit map spans orders of magnitude inside the footprint and the
        restricted covariance carries ``Σ²``.

    Returns
    -------
    Filter
        Graded, and full rank for a strictly positive hit map, in which case
        :meth:`Filter.from_operator` warns that it kept every singular value.

    Notes
    -----
    Pixels with zero hits inside the mask contribute a zero singular value and
    are cut by the threshold, lowering the rank: an unobserved pixel inside the
    footprint is removed from the analysis, which is the honest behaviour but
    is not a no-op.

    The dense ``(n, n)`` diagonal handed to the SVD is wasteful for an operator
    whose factorisation is a sorted permutation. That is deliberate: a control
    routed through a special-cased path proves less about the generic path than
    one routed through the same code as every other operator.
    """
    mask = _as_columns(mask)
    hits = np.asarray(hits, dtype=np.float64)
    if hits.ndim == 1:
        hits = np.repeat(hits[:, np.newaxis], mask.shape[1], axis=1)
    if hits.shape != mask.shape:
        raise ValueError(
            f"hits has shape {hits.shape}, expected {mask.shape} (or (npix,) to "
            "share one hit map across components)"
        )
    if np.any(hits < 0.0):
        raise ValueError("hit counts must be non-negative")

    # The transpose is load-bearing: active_pixel_index indexes the
    # component-major flattening, so raveling the (npix, ncomponents) hit map
    # as-is would scatter every count onto the wrong component.
    weights = np.sqrt(hits.T.ravel()[active_pixel_index(mask)])
    return Filter.from_operator(
        np.diag(weights),
        range_epsilon=range_epsilon,
        range_rank=range_rank,
        mask=mask,
    )


def scan_polynomial(*, mask, pole, n_stripes, degree, template_epsilon=1e-3):
    """
    Deproject low-order polynomials along the scan, one family per stripe.

    The sky is cut into ``n_stripes`` bands of constant polar angle about
    ``pole`` (constant elevation, if ``pole`` is the scan axis), and within each
    band, for each component independently, Legendre polynomials up to
    ``degree`` in the along-band azimuth are removed. This is the pixel-space
    image of per-scan baseline removal in the time domain.

    The operator itself is geometry, not instrument physics: polynomials in one
    coordinate over bands of another. Every instrument-specific number is a
    required argument, so nothing about a particular scan is assumed here.

    Parameters
    ----------
    mask : numpy.ndarray
        Analysis mask, ``(npix, ncomponents)``. Templates are built per
        component, so no spin information is needed.
    pole : tuple of float
        ``(theta, phi)`` in radians of the scan axis, in the same convention as
        :func:`healpy.pix2ang`.
    n_stripes : int
        Number of constant-elevation bands. **Required, with no default and no
        derivation from the geometry**, see the Notes.
    degree : int
        Maximum Legendre degree per stripe. ``degree=0`` removes a per-stripe
        offset, ``degree=1`` adds the along-scan ramp. Templates before
        orthogonalisation number ``n_stripes * ncomponents * (degree + 1)``.
    template_epsilon : float
        Relative cut deciding how many template directions are independent.

    Returns
    -------
    Filter
        A projector, ``rank = n_active - (independent template directions)``.

    Notes
    -----
    ``n_stripes`` is required because deriving it from the patch's polar-angle
    span is a trap. Rotating ``pole`` from the patch centre towards its edge
    roughly doubles the span of the same pixels, so a derived stripe count
    would double, so the template count would double, so the filter's **rank**
    would change: a sweep over ``pole`` would then compare estimators of
    different rank and read the difference as scan geometry. Nothing raises and
    the numbers stay plausible. Requiring the argument removes the derivation
    that could be confounded.

    The band *edges* still track ``pole``, as they must, since they are the
    scan geometry. What a fixed ``n_stripes`` fixes is the rank, which is what
    makes a sweep comparable.

    Edges are set from the polar-angle span of the union of the components'
    active pixels, not per component, because the scan is one scan: differing
    T and P masks must not give differing stripe boundaries.

    A stripe holding fewer pixels than ``degree + 1`` is rank-deficient. The
    SVD in :meth:`Filter.from_deprojection` absorbs it correctly, but it is
    warned about rather than passed over, since it means the requested degree
    was not achievable there.
    """
    mask = _as_columns(mask)
    npix, ncomponents = mask.shape
    n_stripes = int(n_stripes)
    degree = int(degree)
    if n_stripes < 1:
        raise ValueError(f"n_stripes must be at least 1, got {n_stripes}")
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")

    per_component = active_pixels(mask)
    offsets = np.cumsum([0, *(len(a) for a in per_component)])
    theta, phi = _scan_frame(pole, npix)

    observed = np.unique(np.concatenate(per_component))
    edges = np.linspace(theta[observed].min(), theta[observed].max(), n_stripes + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)

    columns = []
    for component, pixels in enumerate(per_component):
        stripe_of = np.searchsorted(edges, theta[pixels], side="right") - 1
        for stripe in range(n_stripes):
            within = np.flatnonzero(stripe_of == stripe)
            if within.size == 0:
                continue
            if within.size < degree + 1:
                warnings.warn(
                    f"stripe {stripe} of component {component} holds "
                    f"{within.size} pixel(s), fewer than the {degree + 1} "
                    f"polynomials requested; the extra templates are dependent "
                    "and are dropped by the orthogonalisation",
                    stacklevel=2,
                )
            # Legendre rather than raw powers: the stack is orthogonalised by
            # SVD either way, but a monomial stack is near-singular by degree 4
            # and template_epsilon would start eating real directions.
            basis = legvander(_centred_phi(phi[pixels[within]]), degree)
            rows = offsets[component] + within
            for order in range(degree + 1):
                column = np.zeros(offsets[-1])
                column[rows] = basis[:, order]
                columns.append(column)

    return Filter.from_deprojection(
        np.column_stack(columns), template_epsilon=template_epsilon, mask=mask
    )


def harmonic_deprojection(*, mask, spins, ells, slot, template_epsilon=1e-3):
    """
    Deproject the subspace spanned by whole multipoles of one slot.

    For every multipole in ``ells``, the ``2 * ell + 1`` real cut-sky harmonics
    of the requested slot are removed, on every field carrying that slot's spin.

    The templates are rows of the estimator's own harmonic operator ``V``
    (:func:`~cosmocore.basis.harmonic_operator`), evaluated directly on the
    active pixels. No spherical-harmonic transform and no quadrature is
    involved, so the deprojected subspace is *exactly* the modes the Fisher
    works in rather than a resynthesised approximation of them.

    Parameters
    ----------
    mask : numpy.ndarray
        Analysis mask, ``(npix, ncomponents)``.
    spins : sequence of int
        Field spins in declaration order, the same tuple handed to ``Fisher``.
        Required because mask columns are indexed by *component*: ``ncomponents
        = 2`` is Q and U for ``spins=[2]`` and two temperature maps for
        ``spins=[0, 0]``, and nothing in the mask distinguishes them.
    ells : int or sequence of int
        Multipoles to remove. Need not be contiguous.
    slot : Slot or str
        Which slot: ``Slot.S``/``"S"``/``"T"``, ``Slot.G``/``"G"``/``"E"``, or
        ``Slot.C``/``"C"``/``"B"``.
    template_epsilon : float
        Relative cut deciding how many of the templates are independent.

    Returns
    -------
    Filter
        A projector.

    Raises
    ------
    ValueError
        If no field carries the slot's spin, or if a requested multipole is
        below the slot's spin.

    Notes
    -----
    **This is not purification.** A cut-sky pure-B harmonic is not orthogonal
    to the E harmonics, so removing the span of the cut-sky B modes is a
    well-defined subspace removal and is *not* the removal of B power.
    Smith-style purification is a different operator and is not this function.

    Low multipoles on a small patch are strongly degenerate on the cut sky, so
    the rank removed is well below ``sum(2 ell + 1)``. That is not a defect:
    ``template_epsilon`` is counting genuinely independent directions, and a
    harmonic that barely exists on the patch contributes a negligible singular
    value and is dropped, as it should be. In-band annihilation therefore holds
    to ``template_epsilon``, not to machine precision.

    **The removal is not sharp in multipole, and its reach is set by the
    patch, not by** ``ells``. This is why the function is not called a band
    stop: it has no edges. A patch of radius ``r`` degrees first resolves
    ``ell ~ 180 / r``, and below that it cannot tell an unnamed harmonic apart
    from the ones that were deprojected, so their power goes too. Measured at
    nside 16 for ``ells = 0..3``: on an 18 degree cap, multipole 4 retains 0.9
    per cent of its norm, 9 retains 15 per cent and 25 retains 98 per cent; on
    a 45 degree cap, where ``180 / r`` is about 4, the same multipoles retain
    20, 95 and 99.8 per cent. Choose ``ells`` knowing the filter reaches well
    above the largest of them.

    The same non-orthogonality is what makes an ``E``-slot deprojection bite
    into ``B``, which is the concrete reason this is not purification.

    The beam does not enter, and could not: ``V`` is beam-free by construction,
    and scaling individual templates by per-multipole constants leaves the
    subspace they span unchanged.

    Deproject one slot per filter and combine with ``&``. A single stack mixing
    slots would put one relative cut across families whose column norms are
    incommensurable.
    """
    from .basis import ell_mode_index, harmonic_operator
    from .conventions.cmb import slot_from_label

    mask = _as_columns(mask)
    npix, ncomponents = mask.shape
    nside = hp.npix2nside(npix)
    slot = slot_from_label(slot)

    ells = np.atleast_1d(np.asarray(ells, dtype=int)).ravel()
    if ells.size == 0:
        raise ValueError("ells is empty; there is nothing to deproject")
    if ells.min() < abs(slot.spin):
        raise ValueError(
            f"slot {slot.label} has spin {slot.spin}, so multipole "
            f"{ells.min()} does not exist for it"
        )
    lmin, lmax = int(ells.min()), int(ells.max())

    columns_of = _field_columns(spins, ncomponents)
    fields = [i for i, spin in enumerate(spins) if spin == slot.spin]
    if not fields:
        raise ValueError(
            f"slot {slot.label} needs a spin-{slot.spin} field, but "
            f"spins={list(spins)} has none"
        )

    per_component = active_pixels(mask)
    offsets = np.cumsum([0, *(len(a) for a in per_component)])
    rows_of_ell = ell_mode_index(lmin, lmax)

    per_field = sum(2 * int(ell) + 1 for ell in ells)
    templates = np.zeros((offsets[-1], len(fields) * per_field))
    for position, field in enumerate(fields):
        columns = columns_of[field]
        # A spin-2 field's two components share one active-pixel set
        # (CONTEXT.md:98), so one pointing set drives the whole block.
        pixels = per_component[columns[0]]
        theta, phi = hp.pix2ang(nside, pixels)
        V = harmonic_operator(theta, phi, spin=slot.spin, lmin=lmin, lmax=lmax)
        n_modes = V.shape[0] // 2 if slot.spin else V.shape[0]
        # Spin-2 rows are [E | B]; Slot.G is the E block, Slot.C the B block.
        row_base = n_modes if slot is Slot.C else 0
        n_pix = len(pixels)

        rows = [row_base + r for ell in ells for r in rows_of_ell[int(ell)]]
        selected = V[rows]
        first = position * per_field
        for block, component in enumerate(columns):
            start = offsets[component]
            templates[start : start + n_pix, first : first + per_field] = selected[
                :, block * n_pix : (block + 1) * n_pix
            ].T

    return Filter.from_deprojection(
        templates, template_epsilon=template_epsilon, mask=mask
    )
