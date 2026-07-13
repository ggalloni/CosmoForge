"""Test I/O operations functionality from cosmocore."""

import inspect

import healpy as hp
import numpy as np
from astropy.io import fits

from cosmocore import read_covmat, read_maps, read_mask, write_out_matrix, writecl


def test_read_mask(tmp_path):
    """Test mask reading functionality with correct signature."""
    # Create test mask file
    nside = 4
    npix = hp.nside2npix(nside)
    nmaps = 2

    # Create mask for T, Q
    mask_data = np.ones((nmaps, npix), dtype=np.float32)
    mask_data[0, npix // 2 :] = 0  # Mask half of T pixels
    mask_data[1, npix // 3 :] = 0  # Mask 2/3 of Q pixels

    mask_file = tmp_path / "test_mask.fits"
    hp.write_map(str(mask_file), mask_data, overwrite=True)

    # Test reading mask with pre-allocated array
    mask_output = np.zeros((nmaps, npix), dtype=np.float64)
    result = read_mask(str(mask_file), mask_output)

    # The function returns (npix, nmaps) transposed shape
    assert result.shape == (npix, nmaps)

    # Check that masking worked correctly (accounting for transpose)
    assert np.sum(result[:, 0]) == npix // 2  # T field
    assert np.sum(result[:, 1]) == npix // 3  # Q field


def test_read_covmat_basic(tmp_path):
    """Test covariance matrix reading with correct signature."""
    # Create test covariance matrix
    npix = 4
    nmaps = 2
    n_active = 6  # Some subset of pixels are active

    # Create active pixel indices
    active = np.array(
        [0, 1, 4, 5, npix, npix + 1], dtype=np.int32
    )  # Mix from both fields

    # Create a full covariance matrix
    full_size = npix * nmaps
    full_cov = np.random.randn(full_size, full_size).astype(np.float64)
    full_cov = full_cov @ full_cov.T  # Make positive definite
    full_cov = 0.5 * (full_cov + full_cov.T)  # Ensure symmetry

    # Save as binary file
    cov_file = tmp_path / "test_covmat.bin"
    full_cov.tofile(str(cov_file))

    # Test reading covariance matrix
    C_output = np.zeros((n_active, n_active), dtype=np.float64)
    result = read_covmat(str(cov_file), npix, nmaps, active, C_output)

    # Check output shape
    assert result.shape == (n_active, n_active)

    # Verify it's symmetric
    np.testing.assert_allclose(result, result.T, rtol=1e-12)


def test_write_matrix(tmp_path):
    """Test covariance matrix writing."""
    # Create test covariance matrix
    npix = 4
    nmaps = 2

    # Create a full covariance matrix
    full_size = npix * nmaps
    full_cov = np.random.randn(full_size, full_size).astype(np.float64)
    full_cov = full_cov @ full_cov.T  # Make positive definite
    full_cov = 0.5 * (full_cov + full_cov.T)  # Ensure symmetry

    # Save as text file
    cov_file = tmp_path / "test_matrix.txt"
    write_out_matrix(str(cov_file), full_cov)

    # Test reading covariance matrix
    C_output = np.loadtxt(str(cov_file))

    # Check output shape
    assert C_output.shape == (full_size, full_size)

    # Verify it's symmetric
    np.testing.assert_allclose(C_output, C_output.T, rtol=1e-12)
    np.testing.assert_allclose(C_output, full_cov, rtol=1e-12)


def test_writecl_basic(tmp_path):
    """Test power spectrum writing with correct signature."""
    # Create test power spectra array
    lmax = 10
    ncols = 4  # ell, TT, EE, BB for example

    power_spectra = np.zeros((lmax + 1, ncols))
    power_spectra[:, 0] = np.arange(lmax + 1)  # ell values
    power_spectra[2:, 1] = 1.0 / np.arange(2, lmax + 1) ** 2  # TT
    power_spectra[2:, 2] = 0.5 / np.arange(2, lmax + 1) ** 2  # EE
    power_spectra[2:, 3] = 0.1 / np.arange(2, lmax + 1) ** 2  # BB

    # Test writing power spectra
    cl_file = tmp_path / "test_cls.dat"
    writecl(str(cl_file), power_spectra)

    # Verify file was created and has content
    assert cl_file.exists()
    assert cl_file.stat().st_size > 0

    # Test reading back the data
    data = np.loadtxt(str(cl_file))
    assert data.shape == power_spectra.shape
    np.testing.assert_allclose(data, power_spectra)


def test_read_maps(tmp_path):
    """Test map reading with simplified setup."""
    # Create a simple test case for read_maps
    nside = 2
    npix = hp.nside2npix(nside)  # 48 pixels
    n_sims = 10

    input_maps = []
    hdus = [fits.PrimaryHDU()]
    for i in range(n_sims):
        header = fits.Header()
        header["SIM_NUM"] = i
        header["FIELDS"] = "T1,T2,T3"  # Use multi-character field names

        map_data = np.ones((3, npix))
        input_maps.append(map_data)
        hdus.append(fits.ImageHDU(data=map_data, header=header, name=f"SIM_{i:03d}"))
    input_maps = np.array(input_maps)
    hdulist = fits.HDUList(hdus)

    maps_file = tmp_path / "test_maps.fits"
    hdulist.writeto(str(maps_file), overwrite=True)

    # Create active pixel indices for single field
    n_active = 3 * npix
    pixact = np.array([np.arange(npix) for _ in range(3)])

    maps_output = np.empty((n_active, n_sims))
    read_maps(
        maps_output,
        str(maps_file),
        pixact,
        ["T1", "T2", "T3"],  # Use multi-character field labels
    )

    for sim_idx in range(n_sims):
        for field_idx in range(3):
            start = field_idx * npix
            end = (field_idx + 1) * npix
            np.testing.assert_allclose(
                maps_output[start:end, sim_idx],
                input_maps[sim_idx, field_idx, :],
                rtol=1e-12,
            )


def test_read_maps_simple_healpix_format():
    """Test reading maps from simple HEALPix format (single simulation)."""
    import os

    # Use the fortran reference map which is a standard HEALPix file
    mapfile = "src/cosmoforge.qube/tests/data/nside8/B/fortran_reference/map.fits"
    if not os.path.exists(mapfile):
        import pytest

        pytest.skip("Fortran reference map not found")

    # Read the map directly with healpy for reference
    reference = hp.read_map(mapfile, field=None)  # shape: (3, 768)
    npix = reference.shape[1]

    # Test reading T field via read_maps (simple HEALPix format, nsims=1)
    n_active = npix // 2
    pixact = [np.arange(n_active)]
    maps_output = np.empty((n_active, 1))

    read_maps(maps_output, mapfile, pixact, ["T"])

    np.testing.assert_allclose(maps_output[:, 0], reference[0, :n_active], rtol=1e-12)

    # Test reading Q and U fields
    pixact_qu = [np.arange(n_active), np.arange(n_active)]
    maps_output_qu = np.empty((2 * n_active, 1))

    read_maps(maps_output_qu, mapfile, pixact_qu, ["Q", "U"])

    np.testing.assert_allclose(
        maps_output_qu[:n_active, 0], reference[1, :n_active], rtol=1e-12
    )
    np.testing.assert_allclose(
        maps_output_qu[n_active:, 0], reference[2, :n_active], rtol=1e-12
    )


def test_io_operations_signatures():
    """Test that all I/O functions have correct signatures and can be imported."""
    # Test that functions exist and have correct signatures
    assert callable(read_mask)
    assert callable(read_maps)
    assert callable(read_covmat)
    assert callable(writecl)

    # Test basic parameter checking without file I/O

    # Check read_mask signature
    sig = inspect.signature(read_mask)
    params = list(sig.parameters.keys())
    assert "maskfile" in params
    assert "mask" in params

    # Check read_covmat signature
    sig = inspect.signature(read_covmat)
    params = list(sig.parameters.keys())
    expected_params = ["covmatfile", "npix", "nmaps", "active", "C"]
    for param in expected_params:
        assert param in params

    # Check writecl signature
    sig = inspect.signature(writecl)
    params = list(sig.parameters.keys())
    assert "filename" in params
    assert "power_spectra" in params
