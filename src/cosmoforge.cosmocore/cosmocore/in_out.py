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
    ntot = active.size
    full_size = int(npix * nmaps)

    NCVMfull = np.fromfile(covmatfile.strip(), dtype=np.float64)
    NCVMfull = NCVMfull.reshape((full_size, full_size))

    for i in range(ntot):
        for j in range(i, ntot):
            C[i, j] = NCVMfull[active[i], active[j]]
            C[j, i] = C[i, j]
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
        cls_dict = {}
        for i, label in enumerate(labels):
            if label.lower() == "ell":
                continue  # skip the ell column
            cls_dict[label] = arr[: effective_lmax - 1, i]
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
    Read map data from multi-simulation FITS files.

    Parameters
    ----------
    maps : numpy.ndarray
        Output array of shape (n_total_active, n_sims) to store map data.
    filename : str
        Path to FITS file containing simulation data.
    pixact : list of numpy.ndarray
        List of active pixel indices for each field.
    field_labels : list of str
        Labels identifying which fields to read from each simulation.
        These should be individual field names (e.g., ["T", "Q", "U"]).
        Field expansion is handled at the parameter level.
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
    Reads map data from FITS files with structure:
    - Each HDU named "SIM_XXX" contains one simulation
    - HDU headers contain "FIELDS" keyword describing field organization
    - Supports both comma-separated ("T,Q,U") and concatenated ("TQU") header formats
    - Applies calibration factor to all loaded data
    - Field expansion (e.g., "QU" -> ["Q", "U"]) is handled at InputParams level
    """
    assert maps.shape[0] == sum(len(p) for p in pixact), "maps array has incorrect shape"
    nsims = maps.shape[1]

    # Ensure we have the right number of labels
    if len(field_labels) != len(pixact):
        raise ValueError(
            f"Mismatch between pixact length ({len(pixact)}) and "
            f"field labels ({len(field_labels)}): {field_labels}"
        )

    with fits.open(filename) as hdul:
        for isim in range(nsims):
            sim_data = hdul[f"SIM_{isim:03d}"].data

            counter = 0
            for field_idx in range(len(pixact)):
                label = field_labels[field_idx]
                field_index = get_field_index(hdul[f"SIM_{isim:03d}"], label)

                pixels = pixact[field_idx].astype(int)

                if len(pixels) > 0 and (
                    pixels.min() < 0 or pixels.max() >= sim_data.shape[1]
                ):
                    raise ValueError(f"Pixel indices out of bounds for field {field_idx}")

                n_active = pixels.size
                maps[counter : counter + n_active, isim] = sim_data[field_index][pixels]
                counter += n_active

    maps *= calibration


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
