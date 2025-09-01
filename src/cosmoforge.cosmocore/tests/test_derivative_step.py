"""Test derivative step functionality from cosmocore.

This module tests the do_derivative_step function which is used in Fisher matrix
calculations to compute derivatives of the signal covariance matrix with respect
to power spectrum parameters.
"""

import os
import tempfile

import healpy as hp
import numpy as np

from cosmocore import (
    FieldCollection,
    InputParams,
    compute_pointings,
    create_field,
    do_derivative_step,
    read_mask,
)


def test_derivative_step_scalar_fields():
    """Test derivative step computation for scalar field correlations (TT)."""
    Par = InputParams()
    mock_config_dict = {
        "nside": 4,
        "lmax": 8,
        "spins": [0],
        "labels": ["T"],
        "physical_labels": ["T"],
        "maskfile": "tmp/mask.fits",
    }

    # Create temporary mask file
    mask = np.ones((1, hp.nside2npix(4)), dtype=np.float64)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_mask_file:
        hp.write_map(tmp_mask_file.name, mask, overwrite=True)
        mock_config_dict["maskfile"] = tmp_mask_file.name
    Par.update(mock_config_dict)

    npix = hp.nside2npix(Par.nside)
    mask = np.empty((Par.nfields, npix), dtype=np.float64)
    mask = read_mask(Par.maskfile, mask)

    # Create fields using the new design
    fields = []
    counter = 0
    for spin in Par.spins:
        if spin == 0:
            labels = Par.labels[counter]
            counter += 1
        else:
            labels = [Par.labels[counter], Par.labels[counter + 1]]
            counter += 2

        if mask.ndim == 1:
            mask = mask[:, np.newaxis]
        field = create_field(
            spin=spin,
            nside=Par.nside,
            lmax=Par.lmax,
            mask=mask[:, counter - 1],
            labels=labels,
        )
        fields.append(field)

    # Create collection using new design
    collection = FieldCollection(Par, fields)

    npixs = []
    for field in fields:
        npixs += field.n_active if field.spin == 0 else field.n_active * 2

    point_vectors = tuple(
        np.empty((npixs[i], 3), dtype=np.float64) for i in range(len(npixs))
    )
    pixact = collection.get_active_pixels()
    point_vectors = compute_pointings(
        Par.nside, npixs, point_vectors, pixact, Par.ordering
    )

    collection.set_pointing_vectors(point_vectors)

    # Test derivative step computation
    ntot = collection.total_active_pixels
    S = np.zeros((ntot, ntot), dtype=np.float64)

    # Test with TT spectrum (spectrum index 0)
    spectrum = 0
    current_ell = 5

    do_derivative_step(
        S=S,
        spectrum=spectrum,
        npixs=npixs,
        spins=Par.spins,
        current_ell=current_ell,
        fields=collection,
    )

    # Basic validity checks
    assert S.shape == (ntot, ntot)
    assert np.all(np.isfinite(S))

    # The result should be symmetric for scalar fields
    np.testing.assert_allclose(S, S.T, rtol=1e-12)

    # Should not be all zeros (assuming reasonable inputs)
    assert not np.allclose(S, 0)

    # Clean up
    os.remove(mock_config_dict["maskfile"])


def test_derivative_step_polarization_fields():
    """Test derivative step computation for polarization field correlations."""
    Par = InputParams()
    mock_config_dict = {
        "nside": 4,
        "lmax": 8,
        "spins": [2],
        "labels": ["E", "B"],
        "physical_labels": ["Q", "U"],
        "maskfile": "tmp/mask.fits",
    }

    # Create temporary mask file
    mask = np.ones((2, hp.nside2npix(4)), dtype=np.float64)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_mask_file:
        hp.write_map(tmp_mask_file.name, mask, overwrite=True)
        mock_config_dict["maskfile"] = tmp_mask_file.name
    Par.update(mock_config_dict)

    npix = hp.nside2npix(Par.nside)
    mask = np.empty((Par.nfields, npix), dtype=np.float64)
    mask = read_mask(Par.maskfile, mask)

    # Create fields using the new design
    fields = []
    counter = 0
    for spin in Par.spins:
        if spin == 0:
            labels = Par.labels[counter]
            counter += 1
        else:
            labels = [Par.labels[counter], Par.labels[counter + 1]]
            counter += 2

        if mask.ndim == 1:
            mask = mask[:, np.newaxis]
        field = create_field(
            spin=spin,
            nside=Par.nside,
            lmax=Par.lmax,
            mask=mask[:, counter - 1],
            labels=labels,
        )
        fields.append(field)

    # Create collection using new design
    collection = FieldCollection(Par, fields)

    npixs = []
    for field in fields:
        npixs += field.n_active if field.spin == 0 else field.n_active * 2

    point_vectors = tuple(
        np.empty((npixs[i], 3), dtype=np.float64) for i in range(len(npixs))
    )
    pixact = collection.get_active_pixels()
    point_vectors = compute_pointings(
        Par.nside, npixs, point_vectors, pixact, Par.ordering
    )

    collection.set_pointing_vectors(point_vectors)

    # Test derivative step computation for different polarization spectra
    ntot = collection.total_active_pixels
    current_ell = 5

    # Test EE spectrum (spectrum index 0 for polarization-only)
    S_EE = np.zeros((ntot, ntot), dtype=np.float64)
    do_derivative_step(
        S=S_EE,
        spectrum=0,  # EE
        npixs=npixs,
        spins=Par.spins,
        current_ell=current_ell,
        fields=collection,
    )

    # Test BB spectrum (spectrum index 1 for polarization-only)
    S_BB = np.zeros((ntot, ntot), dtype=np.float64)
    do_derivative_step(
        S=S_BB,
        spectrum=1,  # BB
        npixs=npixs,
        spins=Par.spins,
        current_ell=current_ell,
        fields=collection,
    )

    # Test EB spectrum (spectrum index 2 for polarization-only)
    S_EB = np.zeros((ntot, ntot), dtype=np.float64)
    do_derivative_step(
        S=S_EB,
        spectrum=2,  # EB
        npixs=npixs,
        spins=Par.spins,
        current_ell=current_ell,
        fields=collection,
    )

    # Basic validity checks for all spectra
    for S, label in [(S_EE, "EE"), (S_BB, "BB"), (S_EB, "EB")]:
        assert S.shape == (ntot, ntot), f"Wrong shape for {label}"
        assert np.all(np.isfinite(S)), f"Non-finite values in {label}"

        # The result should be symmetric
        np.testing.assert_allclose(
            S, S.T, rtol=1e-12, err_msg=f"Not symmetric for {label}"
        )

        # Should not be all zeros (assuming reasonable inputs)
        assert not np.allclose(S, 0), f"All zeros for {label}"

    # EE and BB should be different
    assert not np.allclose(S_EE, S_BB), "EE and BB should be different"

    # EB should be different from both EE and BB
    assert not np.allclose(S_EB, S_EE), "EB and EE should be different"
    assert not np.allclose(S_EB, S_BB), "EB and BB should be different"

    # Clean up
    os.remove(mock_config_dict["maskfile"])


def test_derivative_step_temperature_polarization():
    """Test derivative step computation for T-P correlations."""
    Par = InputParams()
    mock_config_dict = {
        "nside": 4,
        "lmax": 8,
        "spins": [0, 2],
        "labels": ["T", "E", "B"],
        "physical_labels": ["T", "Q", "U"],
        "maskfile": "tmp/mask.fits",
    }

    # Create temporary mask file
    mask = np.ones((3, hp.nside2npix(4)), dtype=np.float64)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_mask_file:
        hp.write_map(tmp_mask_file.name, mask, overwrite=True)
        mock_config_dict["maskfile"] = tmp_mask_file.name
    Par.update(mock_config_dict)

    npix = hp.nside2npix(Par.nside)
    mask = np.empty((Par.nfields, npix), dtype=np.float64)
    mask = read_mask(Par.maskfile, mask)

    # Create fields using the new design
    fields = []
    counter = 0
    for spin in Par.spins:
        if spin == 0:
            labels = Par.labels[counter]
            counter += 1
        else:
            labels = [Par.labels[counter], Par.labels[counter + 1]]
            counter += 2

        if mask.ndim == 1:
            mask = mask[:, np.newaxis]
        field = create_field(
            spin=spin,
            nside=Par.nside,
            lmax=Par.lmax,
            mask=mask[:, counter - 1],
            labels=labels,
        )
        fields.append(field)

    # Create collection using new design
    collection = FieldCollection(Par, fields)

    npixs = []
    for field in fields:
        npixs += field.n_active if field.spin == 0 else field.n_active * 2

    point_vectors = tuple(
        np.empty((npixs[i], 3), dtype=np.float64) for i in range(len(npixs))
    )
    pixact = collection.get_active_pixels()
    point_vectors = compute_pointings(
        Par.nside, npixs, point_vectors, pixact, Par.ordering
    )

    collection.set_pointing_vectors(point_vectors)

    # Test derivative step computation for temperature-polarization cross-correlations
    ntot = collection.total_active_pixels
    current_ell = 5

    # Test TE spectrum
    S_TE = np.zeros((ntot, ntot), dtype=np.float64)
    do_derivative_step(
        S=S_TE,
        spectrum=4,  # TE (assuming TT=0, EE=1, BB=2, TE=3)
        npixs=npixs,
        spins=Par.spins,
        current_ell=current_ell,
        fields=collection,
    )

    # Test TB spectrum
    S_TB = np.zeros((ntot, ntot), dtype=np.float64)
    do_derivative_step(
        S=S_TB,
        spectrum=5,  # TB (assuming TT=0, EE=1, BB=2, TE=3, TB=4)
        npixs=npixs,
        spins=Par.spins,
        current_ell=current_ell,
        fields=collection,
    )

    # Basic validity checks
    for S, label in [(S_TE, "TE"), (S_TB, "TB")]:
        assert S.shape == (ntot, ntot), f"Wrong shape for {label}"
        assert np.all(np.isfinite(S)), f"Non-finite values in {label}"

        # The result should be symmetric
        np.testing.assert_allclose(
            S, S.T, rtol=1e-12, err_msg=f"Not symmetric for {label}"
        )

        # Should not be all zeros (assuming reasonable inputs)
        assert not np.allclose(S, 0), f"All zeros for {label}"

    # TE and TB should be different
    assert not np.allclose(S_TE, S_TB), "TE and TB should be different"

    # Clean up
    os.remove(mock_config_dict["maskfile"])


def test_derivative_step_scalar_temperature_polarization():
    """Test derivative step computation for T-P correlations."""
    Par = InputParams()
    mock_config_dict = {
        "nside": 4,
        "lmax": 8,
        "spins": [0, 0, 0],
        "labels": ["T", "E", "B"],
        "physical_labels": ["T", "E", "B"],
        "maskfile": "tmp/mask.fits",
    }

    # Create temporary mask file
    mask = np.ones((3, hp.nside2npix(4)), dtype=np.float64)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_mask_file:
        hp.write_map(tmp_mask_file.name, mask, overwrite=True)
        mock_config_dict["maskfile"] = tmp_mask_file.name
    Par.update(mock_config_dict)

    npix = hp.nside2npix(Par.nside)
    mask = np.empty((Par.nfields, npix), dtype=np.float64)
    mask = read_mask(Par.maskfile, mask)

    # Create fields using the new design
    fields = []
    counter = 0
    for spin in Par.spins:
        if spin == 0:
            labels = Par.labels[counter]
            counter += 1
        else:
            labels = [Par.labels[counter], Par.labels[counter + 1]]
            counter += 2

        if mask.ndim == 1:
            mask = mask[:, np.newaxis]
        field = create_field(
            spin=spin,
            nside=Par.nside,
            lmax=Par.lmax,
            mask=mask[:, counter - 1],
            labels=labels,
        )
        fields.append(field)

    # Create collection using new design
    collection = FieldCollection(Par, fields)

    npixs = []
    for field in fields:
        npixs += field.n_active if field.spin == 0 else field.n_active * 2

    point_vectors = tuple(
        np.empty((npixs[i], 3), dtype=np.float64) for i in range(len(npixs))
    )
    pixact = collection.get_active_pixels()
    point_vectors = compute_pointings(
        Par.nside, npixs, point_vectors, pixact, Par.ordering
    )

    collection.set_pointing_vectors(point_vectors)

    # Test derivative step computation for temperature-polarization cross-correlations
    ntot = collection.total_active_pixels
    current_ell = 5

    # Test TE spectrum
    S_TE = np.zeros((ntot, ntot), dtype=np.float64)
    do_derivative_step(
        S=S_TE,
        spectrum=3,  # TE (assuming TT=0, EE=1, BB=2, TE=3)
        npixs=npixs,
        spins=Par.spins,
        current_ell=current_ell,
        fields=collection,
    )

    # Basic validity checks
    for S, label in [(S_TE, "TE")]:
        assert S.shape == (ntot, ntot), f"Wrong shape for {label}"
        assert np.all(np.isfinite(S)), f"Non-finite values in {label}"

        # The result should be symmetric
        np.testing.assert_allclose(
            S, S.T, rtol=1e-12, err_msg=f"Not symmetric for {label}"
        )

        # Should not be all zeros (assuming reasonable inputs)
        assert not np.allclose(S, 0), f"All zeros for {label}"

    # Clean up
    os.remove(mock_config_dict["maskfile"])


def test_derivative_step_consistency():
    """Test consistency of derivative step function across different multipoles."""
    Par = InputParams()
    mock_config_dict = {
        "nside": 4,
        "lmax": 8,
        "spins": [0],
        "labels": ["T"],
        "physical_labels": ["T"],
        "maskfile": "tmp/mask.fits",
    }

    # Create temporary mask file
    mask = np.ones((1, hp.nside2npix(4)), dtype=np.float64)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_mask_file:
        hp.write_map(tmp_mask_file.name, mask, overwrite=True)
        mock_config_dict["maskfile"] = tmp_mask_file.name
    Par.update(mock_config_dict)

    npix = hp.nside2npix(Par.nside)
    mask = np.empty((Par.nfields, npix), dtype=np.float64)
    mask = read_mask(Par.maskfile, mask)

    # Create fields
    fields = []
    counter = 0
    for spin in Par.spins:
        if spin == 0:
            labels = Par.labels[counter]
            counter += 1
        else:
            labels = [Par.labels[counter], Par.labels[counter + 1]]
            counter += 2

        if mask.ndim == 1:
            mask = mask[:, np.newaxis]
        field = create_field(
            spin=spin,
            nside=Par.nside,
            lmax=Par.lmax,
            mask=mask[:, counter - 1],
            labels=labels,
        )
        fields.append(field)

    collection = FieldCollection(Par, fields)

    npixs = []
    for field in fields:
        npixs += field.n_active if field.spin == 0 else field.n_active * 2

    point_vectors = tuple(
        np.empty((npixs[i], 3), dtype=np.float64) for i in range(len(npixs))
    )
    pixact = collection.get_active_pixels()
    point_vectors = compute_pointings(
        Par.nside, npixs, point_vectors, pixact, Par.ordering
    )

    collection.set_pointing_vectors(point_vectors)

    ntot = collection.total_active_pixels
    spectrum = 0

    # Test across different multipoles
    results = []
    for current_ell in [3, 5, 7]:
        S = np.zeros((ntot, ntot), dtype=np.float64)
        do_derivative_step(
            S=S,
            spectrum=spectrum,
            npixs=npixs,
            spins=Par.spins,
            current_ell=current_ell,
            fields=collection,
        )
        results.append(S.copy())

    # Results should be different for different multipoles
    assert not np.allclose(results[0], results[1]), (
        "Results should differ for different ell"
    )
    assert not np.allclose(results[1], results[2]), (
        "Results should differ for different ell"
    )
    assert not np.allclose(results[0], results[2]), (
        "Results should differ for different ell"
    )

    # But they should all be well-behaved
    for i, S in enumerate(results):
        ell_value = [3, 5, 7][i]
        assert np.all(np.isfinite(S)), f"Non-finite values for ell={ell_value}"
        np.testing.assert_allclose(
            S, S.T, rtol=1e-12, err_msg=f"Not symmetric for ell={ell_value}"
        )

    # Clean up
    os.remove(mock_config_dict["maskfile"])
