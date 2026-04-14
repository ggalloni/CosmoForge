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

    Defines bins by their lower and upper multipole bounds. Provides
    operators for binning spectra and covariance matrices.

    Parameters
    ----------
    lmins : array_like
        Lower bound of each bin (inclusive).
    lmaxs : array_like
        Upper bound of each bin (inclusive).

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
        Width of each bin (lmax - lmin + 1).
    lmin : int
        Global minimum multipole.
    lmax : int
        Global maximum multipole.
    """

    def __init__(self, lmins, lmaxs):
        if len(lmins) != len(lmaxs):
            msg = "Incoherent inputs"
            raise ValueError(msg)

        lmins = np.asarray(lmins)
        lmaxs = np.asarray(lmaxs)
        cutfirst = np.logical_and(lmaxs >= 2, lmins >= 2)
        self.lmins = lmins[cutfirst]
        self.lmaxs = lmaxs[cutfirst]

        self._derive_ext()

    @classmethod
    def fromdeltal(cls, lmin, lmax, delta_ell):
        """Create uniform bins with constant width."""
        nbins = (lmax - lmin + 1) // delta_ell
        lmins = lmin + np.arange(nbins) * delta_ell
        lmaxs = lmins + delta_ell - 1
        return cls(lmins, lmaxs)

    def _derive_ext(self):
        for l1, l2 in zip(self.lmins, self.lmaxs):
            if l1 > l2:
                msg = "Incoherent inputs"
                raise ValueError(msg)
        self.lmin = np.min(self.lmins)
        self.lmax = np.max(self.lmaxs)
        if self.lmin < 1:
            msg = "Input lmin is less than 1."
            raise ValueError(msg)
        if self.lmax < self.lmin:
            msg = "Input lmax is less than lmin."
            raise ValueError(msg)

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
