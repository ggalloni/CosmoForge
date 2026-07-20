"""
Multipole binning for power spectrum estimation.

Provides the Bins class for defining multipole bins with configurable
widths and weights. Used by QUBE for binned QML estimation, where
per-bin derivative matrices replace per-ell derivatives.

When delta_ell=1, each bin contains a single multipole and the binned
estimator reduces to the standard per-multipole QML estimator.
"""

from __future__ import annotations

import warnings

import numpy as np


class Bins:
    """
    Multipole binning specification.

    Defines bins by their lower and upper multipole bounds. Both bounds
    are **inclusive**: a bin with lmin=2, lmax=4 contains multipoles
    {2, 3, 4}. Bins must not overlap and must be sorted in ascending
    order. Gaps between bins are allowed (those multipoles are simply
    excluded from the analysis).

    The binning operator formalism follows Bond, Jaffe & Knox (1998).
    The implementation is adapted from the xQML package (Vanneste et al.
    2018, Phys. Rev. D 98, 103526), extended with input validation and a
    configurable ``lmin_floor`` for monopole/dipole-aware analyses.

    Parameters
    ----------
    lmins : array_like
        Lower bound of each bin (inclusive). Bins with lmax < 2 are
        automatically discarded since CMB power spectra start at ell=2.
    lmaxs : array_like
        Upper bound of each bin (inclusive). Must satisfy
        lmaxs[i] >= lmins[i] for each bin.

    Attributes
    ----------
    lmins : np.ndarray
        Lower bounds after filtering (ell >= 2).
    lmaxs : np.ndarray
        Upper bounds after filtering (ell >= 2).
    nbins : int
        Number of bins.
    lmid : np.ndarray
        Bin midpoint ``(lmins + lmaxs) / 2``. This is a cheap label, **not**
        the effective multipole: where a bandpower sits depends on the noise,
        mask and Fisher weighting, none of which ``Bins`` knows. For the
        effective multipole use ``Fisher.get_effective_ells()`` (ADR-0019).
    dl : np.ndarray
        Width of each bin (lmaxs - lmins + 1).
    lmin : int
        Global minimum multipole.
    lmax : int
        Global maximum multipole.

    Examples
    --------
    Uniform bins of width 3 from ell=2 to ell=10:

    >>> bins = Bins.fromdeltal(2, 10, 3)
    >>> bins.lmins
    array([2, 5, 8])
    >>> bins.lmaxs
    array([4, 7, 10])

    Custom non-uniform bins:

    >>> bins = Bins([2, 5, 10], [4, 9, 20])
    >>> bins.dl
    array([3, 5, 11])
    """

    def __init__(self, lmins, lmaxs, lmin_floor=2):
        if len(lmins) != len(lmaxs):
            raise ValueError(
                f"lmins and lmaxs must have the same length, "
                f"got {len(lmins)} and {len(lmaxs)}"
            )

        lmins = np.asarray(lmins, dtype=int)
        lmaxs = np.asarray(lmaxs, dtype=int)
        if lmin_floor < 0:
            raise ValueError(f"lmin_floor must be >= 0, got {lmin_floor}")
        # Reject bins entirely below lmin_floor (default 2 reproduces the
        # legacy spin-2 floor; pass lmin_floor=1 for dipole or 0 for
        # monopole-aware analyses).
        keep = np.logical_and(lmaxs >= lmin_floor, lmins >= lmin_floor)
        self.lmins = lmins[keep]
        self.lmaxs = lmaxs[keep]
        self.lmin_floor = int(lmin_floor)

        self._validate_and_derive()

    @classmethod
    def fromdeltal(cls, lmin, lmax, delta_ell):
        """Create uniform bins with constant width.

        ``lmin`` doubles as the bin floor; values below 2 are honoured
        (e.g. ``Bins.fromdeltal(1, 4, 1)`` includes the dipole).
        """
        nbins = (lmax - lmin + 1) // delta_ell
        lmins = lmin + np.arange(nbins) * delta_ell
        lmaxs = lmins + delta_ell - 1
        return cls(lmins, lmaxs, lmin_floor=lmin)

    def _validate_and_derive(self):
        if len(self.lmins) == 0:
            raise ValueError(
                f"No valid bins (all bins below lmin_floor={self.lmin_floor})"
            )

        for i, (l1, l2) in enumerate(zip(self.lmins, self.lmaxs)):
            if l1 > l2:
                raise ValueError(f"Bin {i}: lmin={l1} > lmax={l2}")

        # Check for overlaps (bins must be non-overlapping and sorted)
        order = np.argsort(self.lmins)
        self.lmins = self.lmins[order]
        self.lmaxs = self.lmaxs[order]
        for i in range(len(self.lmins) - 1):
            if self.lmins[i + 1] <= self.lmaxs[i]:
                raise ValueError(
                    f"Bins {i} and {i + 1} overlap: "
                    f"[{self.lmins[i]}, {self.lmaxs[i]}] and "
                    f"[{self.lmins[i + 1]}, {self.lmaxs[i + 1]}]"
                )

        self.lmin = int(np.min(self.lmins))
        self.lmax = int(np.max(self.lmaxs))
        self.nbins = len(self.lmins)
        self.lmid = (self.lmins + self.lmaxs) / 2.0
        self.dl = self.lmaxs - self.lmins + 1

    @property
    def lbin(self):
        """Deprecated alias for :attr:`lmid`.

        Renamed because ``lbin`` implied an *effective* multipole, which
        ``Bins`` cannot know (ADR-0019). Kept for one release as a
        warn-and-forward shim per the post-1.0 deprecation policy
        (ADR-0018); removed in the next release.
        """
        warnings.warn(
            "Bins.lbin is deprecated and will be removed in the next release; "
            "use Bins.lmid for the bin midpoint, or "
            "Fisher.get_effective_ells() for the effective multipole.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.lmid

    def shape_weights(self, convention="Cl"):
        """Per-ℓ bandpower shape weight ``w_ℓ`` (ADR-0019).

        The weight declares the in-bin spectrum shape the binned QML
        derivative assumes: ``dC^b = Σ_{ℓ∈b} w_ℓ · b²_ℓ · dC^ℓ``.

        Parameters
        ----------
        convention : {"Cl", "Dl"}
            ``"Cl"`` declares a flat-C_ℓ shape (``w_ℓ = 1``); ``"Dl"``
            declares a flat-D_ℓ shape (``w_ℓ = 2π / (ℓ(ℓ+1))``).
            Case-insensitive.

        Returns
        -------
        np.ndarray
            Weight vector indexed by multipole, shape ``(lmax + 1,)``.
            The weight is a pure function of ℓ, so entries at ℓ outside any
            bin still carry a value (``1`` for ``"Cl"``; ``2π/(ℓ(ℓ+1))`` for
            ``"Dl"``, with ``w[0] = 0``). The binned derivative sum reads
            only ℓ that fall inside a bin, so those gap entries are never
            consumed.

        Raises
        ------
        ValueError
            If ``convention`` is not ``"Cl"`` or ``"Dl"``, or if the
            ``"Dl"`` shape is requested while a bin includes ℓ = 0
            (``D_ℓ`` is undefined at the monopole). ℓ = 1 is well defined
            (``w = π``).
        """
        key = str(convention).strip().lower()
        if key not in ("cl", "dl"):
            raise ValueError(
                f"Unknown shape convention '{convention}'. Must be 'Cl' or 'Dl'."
            )
        if key == "cl":
            return np.ones(self.lmax + 1)

        if self.lmin == 0:
            raise ValueError(
                "The flat-D_ell shape is undefined at the monopole "
                "(2*pi/(ell*(ell+1)) diverges at ell=0), but a bin includes "
                "ell=0. Use the 'Cl' shape for monopole bins, or raise "
                "lmin_floor above 0."
            )
        ell = np.arange(self.lmax + 1, dtype=np.float64)
        w = np.zeros(self.lmax + 1)
        w[1:] = 2 * np.pi / (ell[1:] * (ell[1:] + 1))
        return w
