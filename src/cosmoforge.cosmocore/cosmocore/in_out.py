"""
Input/output operations for cosmological data analysis.

This module provides functions for reading and writing various data formats
used in cosmological analysis, including covariance matrices, power spectra,
masks, geometry files, and map data from FITS files.

Functions
---------
read_covmat
    Read noise covariance matrix from binary file.
write_covmat_reduced
    Write reduced covariance matrix to binary file.
read_mask
    Read HEALPix mask from FITS file.
output_geometry
    Write geometry information to text file.
readcl
    Read power spectra from text file with header.
writecl
    Write power spectra to text file.
write_out_matrix
    Write matrix in formatted text output.
read_maps
    Read map data from multi-simulation FITS files.
get_field_index
    Extract field indices from FITS header information.

Notes
-----
This module handles various file formats commonly used in CMB analysis:
- Binary covariance matrices for noise modeling
- HEALPix FITS files for masks and maps
- Text files for power spectra with flexible header formats
- Multi-simulation FITS files with structured field organization
"""

import os

import healpy as hp
import numpy as np
from astropy.io import fits

from .settings import InputParams


def _ensure_output_directory(filepath):
    """
    Ensure the parent directory of the given filepath exists.

    Parameters
    ----------
    filepath : str
        Path to a file for which the parent directory should be created.
    """
    output_dir = os.path.dirname(filepath.strip())
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


def read_covmat(covmatfile, npix, nmaps, active, C):
    """
    Read noise covariance matrix from binary file and extract active pixel submatrix.

    Parameters
    ----------
    covmatfile : str
        Path to binary file containing full covariance matrix.
    npix : int
        Number of pixels per map.
    nmaps : int
        Number of maps/fields.
    active : numpy.ndarray
        1D array of active pixel indices in flattened format.
    C : numpy.ndarray
        Output array to store the extracted covariance submatrix.

    Returns
    -------
    numpy.ndarray
        Filled covariance matrix C for active pixels only.

    Notes
    -----
    Reads a full-sky covariance matrix stored in binary format and extracts
    only the relevant submatrix corresponding to active (unmasked) pixels.
    The input matrix is assumed to be stored as float64 in row-major order.
    """
    full_size = int(npix * nmaps)

    NCVMfull = np.fromfile(covmatfile.strip(), dtype=np.float64)
    NCVMfull = NCVMfull.reshape((full_size, full_size))

    C[:] = NCVMfull[np.ix_(active, active)]
    return C


def read_covmat_reduced(covmatfile, C):
    """
    Read a pre-reduced covariance matrix from binary file.

    Parameters
    ----------
    covmatfile : str
        Path to binary file containing the reduced covariance matrix
        (active pixels only).
    C : numpy.ndarray
        Output array of shape (n_active, n_active) to store the matrix.

    Returns
    -------
    numpy.ndarray
        Filled covariance matrix C.
    """
    ntot = C.shape[0]
    filepath = covmatfile.strip()
    data = np.fromfile(filepath, dtype=np.float64)
    expected_size = ntot * ntot

    if data.size != expected_size:
        expected_bytes = expected_size * data.dtype.itemsize
        actual_bytes = data.size * data.dtype.itemsize
        raise ValueError(
            f"Invalid reduced covariance matrix size in '{filepath}': "
            f"expected {expected_size} float64 values "
            f"({expected_bytes} bytes) for shape ({ntot}, {ntot}), "
            f"but found {data.size} values ({actual_bytes} bytes)."
        )
    data = data.reshape((ntot, ntot))
    C[:] = data
    return C


def write_covmat_reduced(outcovmatfile, C):
    """
    Write reduced covariance matrix to binary file.

    Parameters
    ----------
    outcovmatfile : str
        Output filename for the reduced covariance matrix.
    C : numpy.ndarray
        Covariance matrix to write, typically for active pixels only.

    Notes
    -----
    Writes the matrix in binary format using numpy's tofile method.
    The output can be read back using numpy.fromfile with appropriate reshaping.
    """
    _ensure_output_directory(outcovmatfile)

    with open(outcovmatfile.strip(), "wb") as f:
        C.tofile(f)


def read_mask(maskfile, mask):
    """
    Read HEALPix mask from FITS file.

    Parameters
    ----------
    maskfile : str
        Path to FITS file containing HEALPix mask(s).
    mask : numpy.ndarray
        Output array of shape (npix, nmaps) to store mask values.

    Returns
    -------
    numpy.ndarray
        Mask array with shape (npix, nmaps) where each column represents
        a different field mask.

    Notes
    -----
    Uses HEALPix's read_map function to load mask data and transposes
    the result to have pixels as rows and maps as columns for consistent
    indexing with the analysis framework.
    """
    nmaps, _ = mask.shape
    m = hp.read_map(maskfile.strip(), field=range(nmaps), dtype=np.float64)
    return np.array(m).T  # transpose to (npix, nmaps)


def output_geometry(filegeometry, npixs, point_vectors, active):
    """
    Write geometry information to text file.

    Parameters
    ----------
    filegeometry : str
        Output filename for geometry data.
    npixs : list of int
        Number of active pixels for each field.
    point_vectors : tuple of numpy.ndarray
        Pointing vectors for each field, each array shape (n_active, 3).
    active : numpy.ndarray or tuple
        Active pixel indices for each field.

    Notes
    -----
    Writes geometry information in a structured text format containing:
    - Field header with field index, number of active pixels
    - For each active pixel: original pixel index and 3D pointing vector

    This format can be used for geometry debugging and external processing.
    """
    nmaps = len(npixs)

    _ensure_output_directory(filegeometry)

    with open(filegeometry.strip(), "w") as f:
        for field_idx in range(nmaps):
            ntemp = npixs[field_idx]
            f.write(f"{field_idx + 1:6d}{ntemp:6d}{0:6d}{0:6d}\n")
            for i in range(ntemp):
                idx = active[field_idx, i]
                vec = point_vectors[field_idx][i, :]
                f.write(f"{idx:6d}{vec[0]:24.16e}{vec[1]:24.16e}{vec[2]:24.16e}\n")


def _parse_convention(value):
    """Normalize a convention string to 'Cl' or 'Dl' (case-insensitive)."""
    canonical = {"cl": "Cl", "dl": "Dl"}
    key = value.strip().lower()
    if key not in canonical:
        raise ValueError(f"Unknown spectra convention '{value}'. Must be 'Cl' or 'Dl'.")
    return canonical[key]


def convert_spectra_normalization(cls_dict, from_norm, to_norm, lstart=2):
    """Convert between Cl and Dl = l(l+1)/(2pi) Cl conventions.

    Modifies cls_dict in place and returns it.

    Parameters
    ----------
    cls_dict : dict
        Dictionary mapping spectrum labels to power spectrum arrays.
    from_norm : str
        Input convention ("Cl" or "Dl", case-insensitive).
    to_norm : str
        Output convention ("Cl" or "Dl", case-insensitive).
    lstart : int, optional
        Starting multipole (default: 2).

    Returns
    -------
    dict
        The same cls_dict, modified in place.
    """
    from_norm = _parse_convention(from_norm)
    to_norm = _parse_convention(to_norm)
    if from_norm == to_norm:
        return cls_dict
    n_ell = next(iter(cls_dict.values())).shape[0]
    ell = np.arange(lstart, lstart + n_ell, dtype=np.float64)
    if from_norm == "Dl" and to_norm == "Cl":
        factor = 2 * np.pi / (ell * (ell + 1))
    else:  # Cl -> Dl
        factor = ell * (ell + 1) / (2 * np.pi)
    for label in cls_dict:
        cls_dict[label] = cls_dict[label] * factor[: len(cls_dict[label])]
    return cls_dict


def readcl(inputclfile, Params: InputParams, logger=None, lmax: int | None = None):
    """
    Read power spectra from text file with header.

    Parameters
    ----------
    inputclfile : str
        Path to text file containing power spectra data.
    Params : InputParams
        Analysis parameters containing lmax and feedback level.
    logger : CosmoLogger, optional
        CosmoLogger instance for output. If None, uses print for backward compatibility.
    lmax : int, optional
        Maximum multipole to read. If None, uses Params.lmax.
        This allows loading Cls up to a different lmax than the analysis lmax.

    Returns
    -------
    dict
        Dictionary mapping spectrum labels to power spectrum arrays.
        Each array has length (lmax-1) corresponding to l=2 to lmax.

    Raises
    ------
    ValueError
        If the first line doesn't start with '#' as expected header format.

    Notes
    -----
    Expected file format:
    - First line: header starting with '#' containing column labels
    - Subsequent lines: numerical data with columns for different spectra
    - 'ell' column is automatically skipped if present
    - Power spectra are truncated to the specified lmax
    """
    effective_lmax = lmax if lmax is not None else Params.lmax
    with open(inputclfile.strip()) as f:
        header = f.readline()
        if not header.lstrip().startswith("#"):
            raise ValueError("First line must be a header starting with '#'")
        labels = header.strip().lstrip("#").split()
        arr = np.loadtxt(f, dtype=np.float64)
        if logger is not None:
            logger.log_with_feedback(f"Read Cls: {arr.shape} {labels}", level=4)
        # Find where ell=2 starts (files may begin at ell=0 or ell=2)
        ell_col = None
        for i, label in enumerate(labels):
            if label.lower() == "ell":
                ell_col = i
                break

        if ell_col is not None:
            ell_values = arr[:, ell_col].astype(int)
            start_row = int(np.searchsorted(ell_values, 2))
        else:
            start_row = 0  # assume data starts at ell=2

        n_ell = effective_lmax - 1  # number of multipoles from ell=2 to lmax
        cls_dict = {}
        for i, label in enumerate(labels):
            if label.lower() == "ell":
                continue
            cls_dict[label] = arr[start_row : start_row + n_ell, i]

    input_conv = _parse_convention(getattr(Params, "input_convention", "Cl"))
    if input_conv != "Cl":
        convert_spectra_normalization(cls_dict, from_norm=input_conv, to_norm="Cl")

    return cls_dict


def writecl(filename: str, power_spectra: np.ndarray):
    """
    Write power spectra array to text file.

    Parameters
    ----------
    filename : str
        Output filename for power spectra data.
    power_spectra : numpy.ndarray
        Power spectra array to write to file.

    Notes
    -----
    Uses high precision format (%.16e) to preserve full double precision.
    Could be enhanced to support more sophisticated formats with headers and labels.
    """
    # Use high precision format to preserve full double precision
    np.savetxt(filename, power_spectra, fmt="%.16e")


def write_out_matrix(outfilematrix, matrix):
    """
    Write matrix in formatted text output.

    Parameters
    ----------
    outfilematrix : str
        Output filename for the formatted matrix.
    matrix : numpy.ndarray
        2D array to write in formatted text format.

    Notes
    -----
    Writes the matrix with scientific notation formatting (24.16E format)
    with each row on a separate line. Suitable for matrices that need
    to be human-readable or imported into other analysis tools.
    The 16 decimal places preserve full double precision.
    """
    n = matrix.shape[0]

    _ensure_output_directory(outfilematrix)

    with open(outfilematrix, "w") as f:
        for i in range(n):
            line = "".join(f"{matrix[i, j]:24.16E}" for j in range(n))
            f.write(line + "\n")


def read_maps(maps, filename, pixact, field_labels, calibration: float = 1.0):
    """
    Read map data from files.

    Supports three formats:
    1. Numpy format (.npy): 3D array of shape (n_fields, n_pix, n_sims).
       The first ``len(field_labels)`` entries along axis 0 are assumed to
       be ordered exactly as ``field_labels``.
    2. Multi-simulation FITS: HDUs named "SIM_XXX" with FIELDS header.
    3. Simple HEALPix FITS: Standard HEALPix file (for single simulation).

    Parameters
    ----------
    maps : numpy.ndarray
        Output array of shape (n_total_active, n_sims) to store map data.
    filename : str
        Path to data file (.npy or .fits).
    pixact : list of numpy.ndarray
        List of active pixel indices for each field.
    field_labels : list of str
        Labels identifying which fields to read from each simulation.
        These should be individual field names (e.g., ["T", "Q", "U"]).
        For numpy format, the first ``len(field_labels)`` entries in the
        file are assumed to match this ordering exactly.
    calibration : float, optional
        Calibration factor to multiply all map values. Default is 1.0.

    Raises
    ------
    AssertionError
        If maps array shape doesn't match expected total active pixels.
    ValueError
        If field labels don't match expected format or pixel indices are invalid.

    Notes
    -----
    For numpy format:
    - File must contain a 3D array of shape (n_fields, n_pix, n_sims)
    - Fields are ordered to match the config's ``physical_labels``
    - No header metadata; field ordering is determined by convention

    For multi-simulation FITS:
    - Each HDU named "SIM_XXX" contains one simulation
    - HDU headers contain "FIELDS" keyword describing field organization

    For simple HEALPix format (nsims=1):
    - Reads directly using healpy
    - Field mapping: T=0, Q=1, U=2, E=1, B=2

    Applies calibration factor to all loaded data.
    """
    assert maps.shape[0] == sum(len(p) for p in pixact), "maps array has incorrect shape"
    nsims = maps.shape[1]

    if len(field_labels) != len(pixact):
        raise ValueError(
            f"Mismatch between pixact length ({len(pixact)}) and "
            f"field labels ({len(field_labels)}): {field_labels}"
        )

    if filename.endswith(".npy"):
        _read_maps_numpy(maps, filename, pixact, nsims)
    else:
        _read_maps_fits(maps, filename, pixact, field_labels, nsims)

    maps *= calibration


def _read_maps_numpy(maps, filename, pixact, nsims):
    """Read maps from numpy .npy file (n_fields, n_pix, n_sims)."""
    all_data = np.load(filename)

    n_fields = len(pixact)
    if all_data.ndim != 3:
        raise ValueError(
            f"Numpy map file must be 3D (n_fields, n_pix, n_sims), "
            f"got shape {all_data.shape}"
        )
    if all_data.shape[0] < n_fields:
        raise ValueError(
            f"Numpy map file has {all_data.shape[0]} fields, need at least {n_fields}"
        )
    if all_data.shape[2] != nsims:
        raise ValueError(f"Numpy map file has {all_data.shape[2]} sims, expected {nsims}")

    pixact_int = [p.astype(int) for p in pixact]
    counter = 0
    for field_idx in range(n_fields):
        pixels = pixact_int[field_idx]
        n_active = pixels.size
        maps[counter : counter + n_active, :] = all_data[field_idx][pixels, :]
        counter += n_active


def _read_maps_fits(maps, filename, pixact, field_labels, nsims):
    """Read maps from FITS file (multi-sim or simple HEALPix)."""
    pixact_int = [p.astype(int) for p in pixact]

    with fits.open(filename) as hdul:
        has_sim_format = "SIM_000" in hdul

        if has_sim_format:
            # Cache field indices from first sim (same for all sims)
            field_indices = []
            for field_idx in range(len(pixact)):
                label = field_labels[field_idx]
                field_indices.append(get_field_index(hdul["SIM_000"], label))

            # Precompute output offsets
            offsets = []
            counter = 0
            for field_idx in range(len(pixact)):
                offsets.append(counter)
                counter += pixact_int[field_idx].size

            # Read all sims
            for isim in range(nsims):
                sim_data = hdul[f"SIM_{isim:03d}"].data

                for field_idx in range(len(pixact)):
                    pixels = pixact_int[field_idx]
                    n_active = pixels.size
                    maps[offsets[field_idx] : offsets[field_idx] + n_active, isim] = (
                        sim_data[field_indices[field_idx]][pixels]
                    )
        else:
            if nsims != 1:
                raise ValueError(
                    f"Simple HEALPix format only supports nsims=1, got nsims={nsims}"
                )

            field_index_map = {"T": 0, "Q": 1, "U": 2, "E": 1, "B": 2}

            all_maps = hp.read_map(filename, field=None)
            if all_maps.ndim == 1:
                all_maps = all_maps.reshape(1, -1)

            counter = 0
            for field_idx in range(len(pixact)):
                label = field_labels[field_idx]
                if label not in field_index_map:
                    raise ValueError(
                        f"Unknown field label '{label}' for simple HEALPix format"
                    )

                map_idx = field_index_map[label]
                if map_idx >= all_maps.shape[0]:
                    raise ValueError(
                        f"Field '{label}' (index {map_idx}) not available in file "
                        f"with {all_maps.shape[0]} fields"
                    )

                pixels = pixact_int[field_idx]
                n_active = pixels.size
                maps[counter : counter + n_active, 0] = all_maps[map_idx][pixels]
                counter += n_active


def get_field_index(hdu, field_name):
    """
    Extract field index from FITS HDU header information.

    Parameters
    ----------
    hdu : astropy.io.fits.HDU
        FITS HDU containing field information in header.
    field_name : str
        Field name to look up (e.g., "T", "Q", "U", "T1", "E2").

    Returns
    -------
    int
        Field index corresponding to the requested field name.

    Raises
    ------
    ValueError
        If requested field is not found in the HDU header.

    Notes
    -----
    Supports two header formats for the "FIELDS" keyword:
    - Comma-separated: "T,Q,U" or "T1,T2,T3"
    - Concatenated: "TQU" (for single-character field names)

    Field expansion is now handled at the InputParams level, so this function
    expects individual field names only.
    """
    fields_str = hdu.header.get("FIELDS", "")

    # Detect format: comma-separated vs concatenated
    if "," in fields_str:
        # Comma-separated format: "T,Q,U" or "T1,T2,T3"
        available_fields = [f.strip() for f in fields_str.split(",")]
    else:
        # Concatenated format: "TQU" -> ["T", "Q", "U"]
        available_fields = list(fields_str.strip())

    # Look up the field name
    if field_name not in available_fields:
        raise ValueError(f"Field '{field_name}' not found in {available_fields}")

    return available_fields.index(field_name)
