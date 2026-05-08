"""
Multipole binning for power spectrum estimation.

Provides the Bins class for defining multipole bins with configurable
widths and weights. Used by QUBE for binned QML estimation, where
per-bin derivative matrices replace per-ell derivatives.

When delta_ell=1, each bin contains a single multipole and the binned
estimator reduces to the standard per-multipole QML estimator.
"""

from __future__ import annotations

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
    2018, Phys. Rev. D 98, 103526), extended with input validation,
    D_ell weighting, covariance binning, and lmin zero-padding support.

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
    lbin : np.ndarray
        Effective multipole for each bin (midpoint).
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
        self.lbin = (self.lmins + self.lmaxs) / 2.0
        self.dl = self.lmaxs - self.lmins + 1

    def bins(self):
        """Return (lmins, lmaxs) tuple."""
        return (self.lmins, self.lmaxs)

    def _bin_operators(self, *, Dl=False, cov=False):
        """
        Build binning (P) and unbinning (Q) operator matrices.

        Parameters
        ----------
        Dl : bool
            If True, weight by ell*(ell+1)/(2*pi).
        cov : bool
            If True, build Q for covariance binning.

        Returns
        -------
        p : np.ndarray
            Binning matrix, shape (nbins, lmax+1).
        q : np.ndarray
            Unbinning matrix, shape (lmax+1, nbins).
        """
        if Dl:
            ell2 = np.arange(self.lmax + 1)
            ell2 = ell2 * (ell2 + 1) / (2 * np.pi)
        else:
            ell2 = np.ones(self.lmax + 1)
        p = np.zeros((self.nbins, self.lmax + 1))
        q = np.zeros((self.lmax + 1, self.nbins))

        for b, (a, z) in enumerate(zip(self.lmins, self.lmaxs)):
            dl = z - a + 1
            p[b, a : z + 1] = ell2[a : z + 1] / dl
            if cov:
                q[a : z + 1, b] = 1 / ell2[a : z + 1] / dl
            else:
                q[a : z + 1, b] = 1 / ell2[a : z + 1]

        return p, q

    def bin_spectra(self, spectra, Dl=False, lmin=0):
        """
        Average spectra in bins.

        Parameters
        ----------
        spectra : array_like
            Power spectra, last axis is multipole.
        Dl : bool
            If True, weight by ell*(ell+1)/(2*pi).
        lmin : int
            Starting multipole of the input spectra. If > 0, the input
            is zero-padded for ell < lmin before binning.

        Returns
        -------
        np.ndarray
            Binned spectra.
        """
        spectra = np.asarray(spectra)
        if lmin > 0:
            pad = np.zeros((*spectra.shape[:-1], lmin))
            spectra = np.concatenate([pad, spectra], axis=-1)
        minlmax = np.min([spectra.shape[-1] - 1, self.lmax])

        _p, _q = self._bin_operators(Dl=Dl)
        return np.dot(spectra[..., : minlmax + 1], _p.T[: minlmax + 1, ...])

    def bin_covariance(self, clcov):
        """Bin a covariance matrix: P @ clcov @ Q."""
        p, q = self._bin_operators(cov=True)
        return np.matmul(p, np.matmul(clcov, q))
