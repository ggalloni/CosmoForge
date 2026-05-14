"""Beam handling: window functions and beam smoothing managers.

Moved from ``cosmocore.harmonic`` (2026-05). See ADR-0002 follow-up
(architecture backlog #5) for the rename rationale.

References
----------
.. [1] Page, L. et al. "First-Year Wilkinson Microwave Anisotropy Probe (WMAP)
   Observations: Beam Profiles and Window Functions"
   Astrophys. J. Suppl. 148, 39-50 (2003)
.. [2] Mitra, S., Rocha, G., Gorski, K.M. et al. "Fast and Efficient Template
   Fitting of Deterministic Anisotropic Cosmological Models Applied to WMAP
   Data" Astrophys. J. Suppl. 193, 5 (2011)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cosmocore import BaseField, InputParams
    from cosmocore.spectra_io import SpectraManager


def coswinbeam(nside, ell1=None, ell2=None):
    """
    Generate a cosine apodizing window beam function.

    Implements the raised-cosine kernel of Akrami+ 2020 (NPIPE, Eq. 15) and
    Aghanim+ 2019 / Benabed+ 2009 (Planck Legacy, Eq. 4):

        b(ℓ) = 1                                           for ℓ ≤ ℓ₁
        b(ℓ) = ½ [1 + cos(π (ℓ − ℓ₁) / (ℓ₂ − ℓ₁))]         for ℓ₁ < ℓ ≤ ℓ₂
        b(ℓ) = 0                                           for ℓ > ℓ₂

    Two named conventions are exposed via ``BeamManager``:
    Legacy uses ``ell1 = nside`` (Benabed+ 2009; Aghanim+ 2019),
    NPIPE uses ``ell1 = 1`` (Akrami+ 2020 §4.2; lower threshold reduced
    from N_side to suppress ringing in the 857-GHz maps).
    Both use ``ell2 = 3 * nside``.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter; sets the returned array length to ``4*nside+1``.
    ell1 : int, optional
        Lower transition multipole. Defaults to ``nside`` (Legacy convention).
    ell2 : int, optional
        Upper transition multipole. Defaults to ``3 * nside``.

    Returns
    -------
    numpy.ndarray, shape (4*nside + 1,)
        Beam window function b(ℓ) for ℓ = 0 to ℓ = 4*nside.
    """
    if ell1 is None:
        ell1 = nside
    if ell2 is None:
        ell2 = 3 * nside

    if not (0 <= ell1 < ell2 <= 4 * nside):
        raise ValueError(
            f"coswinbeam transition multipoles must satisfy "
            f"0 <= ell1 < ell2 <= 4*nside; got ell1={ell1}, ell2={ell2}, nside={nside}."
        )

    L = 4 * nside + 1
    beam = np.zeros(L, dtype=np.float64)
    beam[: ell1 + 1] = 1.0
    ell = np.arange(ell1 + 1, ell2 + 1)
    beam[ell1 + 1 : ell2 + 1] = 0.5 * (1.0 + np.cos((ell - ell1) * np.pi / (ell2 - ell1)))
    return beam


class BeamManager:
    """
    Manages beam functions for a collection of cosmological fields.

    This class handles the computation and application of instrumental beam
    window functions that model the finite angular resolution of cosmological
    surveys. It supports multiple beam types (Gaussian, cosine window, or
    custom from file) and manages beam assignment to different field types.

    The manager computes beam functions for temperature and polarization
    measurements separately, as they may have different instrumental responses.
    It integrates with SpectraManager to apply beam smoothing to theoretical
    power spectra.

    Parameters
    ----------
    fields : list of BaseField
        Collection of cosmological fields for which to manage beam functions.
        Each field should have defined labels and spin properties.

    Attributes
    ----------
    fields : list of BaseField
        The input collection of fields.

    Examples
    --------
    >>> from cosmoforge.cosmocore import ScalarField, PolarizationField
    >>> temp_field = ScalarField(['T'], spin=0, lmax=100)
    >>> pol_field = PolarizationField(['E', 'B'], spin=2, lmax=100)
    >>> beam_mgr = BeamManager([temp_field, pol_field])
    >>> # Set Gaussian beam with 5 arcmin FWHM
    >>> params = InputParams(lmax=100, nside=512, smoothing_type=1, fwhmarcmin=5.0)
    >>> beam_mgr.set_beams_from_params(params)

    Notes
    -----
    Supported beam types:
        - "none": No smoothing (beam = 1)
        - "gaussian": Gaussian beam with specified FWHM
        - "cosine_legacy": Benabed+ 2009 / Aghanim+ 2019 cosine kernel
          (flat to ℓ=nside, cosine roll-off to ℓ=3·nside)
        - "cosine_npipe": Akrami+ 2020 NPIPE cosine kernel
          (flat to ℓ=1, cosine roll-off to ℓ=3·nside)
        - "file": Custom beam from file
    """

    def __init__(self, fields: list[BaseField]):
        self.fields = fields
        self._beam_dict = {}

    def compute_beams(
        self, lmax: int, nside: int, smoothtype: str, fwhmarcmin: float, beam_file: str
    ) -> dict[str, np.ndarray]:
        """
        Compute beam functions based on smoothing type.

        Parameters:
        -----------
        lmax : int
            Maximum multipole
        nside : int
            HEALPix nside parameter
        smoothtype : str
            Type of smoothing: ``"none"``, ``"gaussian"``, ``"cosine_legacy"``,
            ``"cosine_npipe"``, or ``"file"``.
        fwhmarcmin : float
            FWHM in arcminutes for Gaussian beam
        beam_file : str
            Path to beam file for smoothtype="file"

        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with beam functions for T, E, B
        """
        import healpy as hp

        if smoothtype == "none":
            beam = np.ones((3, lmax + 1), dtype=np.float64)
        elif smoothtype == "gaussian":
            # fwhmarcmin in arcminutes → fwhm_rad. healpy's gauss_beam returns
            # an ℓ-indexed array of shape (lmax+1, 4) for (T, E, B, TE);
            # drop the TE column and transpose to (3, lmax+1).
            beam = np.array(
                hp.gauss_beam(np.deg2rad(fwhmarcmin / 60.0), lmax=lmax, pol=True)[
                    : lmax + 1, :-1
                ],
                dtype=np.float64,
            ).T
        elif smoothtype == "cosine_legacy":
            b = coswinbeam(nside, ell1=nside, ell2=3 * nside)[: lmax + 1]
            beam = np.column_stack([b] * 3).T
        elif smoothtype == "cosine_npipe":
            b = coswinbeam(nside, ell1=1, ell2=3 * nside)[: lmax + 1]
            beam = np.column_stack([b] * 3).T
        elif smoothtype == "file":
            # Beam file must contain at least 3 columns: T, E, B window functions.
            # Additional columns (e.g., cross-terms like T-E, T-B) are ignored
            # as they are not needed for power spectrum smoothing.
            bls = hp.read_cl(beam_file.strip()).astype(np.float64)
            if bls.shape[0] < 3:
                raise ValueError(
                    f"Beam file must have at least 3 columns (T, E, B), "
                    f"got {bls.shape[0]}"
                )
            beam = np.column_stack([bls[i][: lmax + 1] for i in range(3)]).T
        else:
            raise ValueError(f"Unknown smoothtype='{smoothtype}'")

        if beam.shape[0] != 3 or beam.shape[1] != lmax + 1:
            raise ValueError(
                f"Beam shape mismatch: expected (3, {lmax + 1}), got {beam.shape}"
            )

        return {
            "T": beam[0, :],
            "E": beam[1, :],
            "B": beam[2, :],
        }

    def set_beams_from_params(self, params: InputParams, lmax: int | None = None) -> None:
        """
        Set beams for all fields using parameter configuration.

        This method computes the appropriate beam functions based on the
        parameter settings and assigns them to each field in the collection.
        Scalar fields receive a single beam function, while polarization
        fields receive separate E and B mode beam functions.

        Parameters
        ----------
        params : InputParams
            Configuration object containing beam parameters including:
            - lmax: Maximum multipole
            - nside: HEALPix resolution parameter
            - smoothing_type: Type of beam ("none", "gaussian",
              "cosine_legacy", "cosine_npipe", "file")
            - fwhmarcmin: FWHM in arcminutes (for Gaussian beams)
            - beam_file: Path to beam file (for custom beams)
        lmax : int, optional
            Maximum multipole to use. If None, uses params.lmax.
            This allows computing beams up to a different lmax than params.lmax.

        Notes
        -----
        The method automatically handles field-beam matching. For scalar fields,
        it uses the 'T' beam. For polarization fields, it uses 'E' and 'B' beams.
        If specific labels aren't found, it falls back to available beams.

        Examples
        --------
        >>> params = InputParams(lmax=100, smoothing_type=1, fwhmarcmin=5.0)
        >>> beam_mgr.set_beams_from_params(params)
        """
        effective_lmax = lmax if lmax is not None else params.lmax
        beam_dict = self.compute_beams(
            lmax=effective_lmax,
            nside=params.nside,
            smoothtype=params.smoothing_type,
            fwhmarcmin=params.fwhmarcmin,
            beam_file=params.beam_file,
        )

        # Build internal beam dictionary with field labels
        self._beam_dict = {}
        for field in self.fields:
            if field.spin == 0:
                # Scalar field - single beam
                beam_label = field.labels[0]
                if beam_label in beam_dict:
                    self._beam_dict[beam_label] = beam_dict[beam_label]
                else:
                    # Fallback to generic beam if label not found
                    self._beam_dict[beam_label] = beam_dict.get(
                        "T", list(beam_dict.values())[0]
                    )
            elif field.spin == 2:
                # Polarization field - E and B beams
                e_beam = beam_dict.get(
                    "E", beam_dict.get("P", list(beam_dict.values())[0])
                )
                b_beam = beam_dict.get(
                    "B", beam_dict.get("P", list(beam_dict.values())[0])
                )
                self._beam_dict[field.labels[0]] = e_beam
                self._beam_dict[field.labels[1]] = b_beam

        # Only set beams on fields if lmax matches field.lmax
        # (otherwise the validation would fail)
        if lmax is None or lmax == self.fields[0].lmax:
            for field in self.fields:
                if field.spin == 0:
                    field.set_beam(self._beam_dict[field.labels[0]])
                elif field.spin == 2:
                    beam_array = np.column_stack(
                        [
                            self._beam_dict[field.labels[0]],
                            self._beam_dict[field.labels[1]],
                        ]
                    )
                    field.set_beam(beam_array)

    def get_beam_dict(self) -> dict[str, np.ndarray]:
        """
        Get dictionary mapping field labels to beam functions.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with field labels as keys (e.g., 'T', 'E', 'B') and
            corresponding beam window functions as values. Each beam is an
            ℓ-indexed 1D array of length ``lmax + 1`` (``beam[ell]``).

        Raises
        ------
        ValueError
            If beams have not been set.

        Notes
        -----
        For scalar fields, the dictionary contains one entry per field label.
        For polarization fields, the beam array is split into separate E and B
        mode entries.

        Examples
        --------
        >>> beam_dict = beam_mgr.get_beam_dict()
        >>> print(beam_dict.keys())  # ['T', 'E', 'B']
        >>> t_beam = beam_dict['T']  # Temperature beam function
        """
        # Return internally stored beams (set by set_beams_from_params)
        if not self._beam_dict:
            raise ValueError("Beams have not been set. Call set_beams_from_params first.")
        return self._beam_dict.copy()

    def get_beam(self, label: str) -> np.ndarray:
        """
        Get beam window function for a specific field label.

        Parameters
        ----------
        label : str
            Field label (e.g., 'T', 'E', 'B').

        Returns
        -------
        np.ndarray
            Beam window function B(ℓ) for ℓ = 2 to ℓ = lmax.

        Raises
        ------
        ValueError
            If beams have not been set or if label is not found.
        """
        beam_dict = self.get_beam_dict()
        if label not in beam_dict:
            raise ValueError(f"No beam found for label '{label}'")
        return beam_dict[label]

    def apply_smoothing(
        self, spectra_manager: SpectraManager, lmax: int | None = None
    ) -> None:
        """
        Apply beam smoothing to power spectra.

        This method applies instrumental beam effects to all power spectra
        managed by the SpectraManager. The smoothing accounts for both the
        beam window functions and the geometric conversion factors.

        Parameters
        ----------
        spectra_manager : SpectraManager
            The SpectraManager instance containing the power spectra to smooth.
            The spectra are modified in-place.
        lmax : int, optional
            Maximum multipole to use. If None, uses the field's lmax.
            This allows applying smoothing up to a different lmax.

        Notes
        -----
        The smoothing operation multiplies each power spectrum by:
        C_ℓ^smoothed = C_ℓ^theory × B₁(ℓ) × B₂(ℓ)

        where B₁(ℓ) and B₂(ℓ) are the beam functions for the two fields
        involved in the spectrum.

        Examples
        --------
        >>> beam_mgr.set_beams_from_params(params)
        >>> spectra_mgr.set_cls_from_file('theory_cls.dat', params)
        >>> beam_mgr.apply_smoothing(spectra_mgr)  # Apply instrumental effects
        """
        # Use precomputed smoothing factors
        smoothing_factors = spectra_manager.compute_smoothing_factors(self, lmax=lmax)

        for label in spectra_manager.labels:
            if label not in spectra_manager._cls_dict:
                raise ValueError(f"No power spectrum found for {label}.")

            if label in smoothing_factors:
                # Apply smoothing factor
                spectra_manager._cls_dict[label] *= smoothing_factors[label]

        # Update matrix from dictionary
        for idx, label in enumerate(spectra_manager._spectra_labels):
            spectra_manager._cls_matrix[:, idx] = spectra_manager._cls_dict[label]
