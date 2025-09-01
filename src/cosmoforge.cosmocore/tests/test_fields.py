"""Test field management functionality extracted from quelo usage patterns."""

import healpy as hp
import numpy as np
import pytest

from cosmocore import (
    PolarizationField,
    ScalarField,
    create_field,
)


def test_create_scalar_field():
    """Test scalar field creation (extracted from test_signal_covmat.py)"""
    nside = 4
    lmax = 8
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=np.float64)

    # Test temperature field creation
    field = create_field(spin=0, nside=nside, lmax=lmax, mask=mask, labels="T")

    assert isinstance(field, ScalarField)
    assert field.spin == 0
    assert field.nside == nside
    assert field.lmax == lmax
    assert field.labels == ["T"]  # labels is always a list
    assert len(field.n_active) == 1  # scalar field has 1 component


def test_create_polarization_field():
    """Test polarization field creation (extracted from test_signal_covmat.py)"""
    nside = 4
    lmax = 8
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=np.float64)

    # Test polarization field creation
    field = create_field(
        spin=2,
        nside=nside,
        lmax=lmax,
        mask=mask,
        labels=["E", "B"],  # Use E/B labels for polarization
    )

    assert isinstance(field, PolarizationField)
    assert field.spin == 2
    assert field.labels == ["E", "B"]
    assert len(field.n_active) == 1  # polarization field has single Q,U component count


def test_field_config_validation():
    """Test FieldConfig validation (tests the dataclass validation)"""
    nside = 4
    lmax = 8
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=np.float64)

    # Test valid configurations
    valid_config_scalar = create_field(
        spin=0, nside=nside, lmax=lmax, mask=mask, labels="T"
    )
    assert valid_config_scalar is not None

    valid_config_pol = create_field(
        spin=2, nside=nside, lmax=lmax, mask=mask, labels=["E", "B"]
    )
    assert valid_config_pol is not None

    # Test invalid spin should raise error
    try:
        create_field(spin=1, nside=nside, lmax=lmax, mask=mask, labels="invalid")
        assert False, "Should have raised ValueError for invalid spin"
    except ValueError:
        pass  # Expected behavior


def test_active_pixels_and_pointings():
    """Test active pixel computation (extracted from test_signal_covmat.py)"""
    nside = 4
    lmax = 8
    npix = hp.nside2npix(nside)

    # Create fields with partial mask
    mask = np.ones(npix, dtype=np.float64)
    mask[npix // 2 :] = 0  # Mask half the pixels

    field_t = create_field(spin=0, nside=nside, lmax=lmax, mask=mask, labels="T")
    field_pol = create_field(spin=2, nside=nside, lmax=lmax, mask=mask, labels=["E", "B"])

    # Test active pixel computation using the correct property
    pixact_t = field_t.active_pixels  # This is the correct property
    pixact_pol = field_pol.active_pixels

    assert len(pixact_t) == npix // 2  # Half pixels active
    assert len(pixact_pol) == npix // 2
    assert np.all(pixact_t < npix // 2)  # Only first half active


def test_cross_spectrum_labels():
    """Test cross spectrum label generation for different field combinations."""
    nside = 4
    lmax = 8
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=np.float64)

    # Create different field types
    temp_field = create_field(spin=0, nside=nside, lmax=lmax, mask=mask, labels="T")
    pol_field = create_field(spin=2, nside=nside, lmax=lmax, mask=mask, labels=["Q", "U"])

    # Test PolarizationField x ScalarField (covers lines 659-663)
    cross_labels = pol_field.get_cross_spectrum_labels(temp_field)
    assert cross_labels == ["TQ", "TU"]

    # Test PolarizationField x PolarizationField (covers lines 664-668)
    cross_labels_pol = pol_field.get_cross_spectrum_labels(pol_field)
    assert cross_labels_pol == ["QQ", "QU", "UQ", "UU"]

    # Test invalid field type (covers line 669)
    with pytest.raises(TypeError, match="Unknown field type"):
        pol_field.get_cross_spectrum_labels("invalid_field")


def test_field_collection_properties():
    """Test FieldCollection properties for missing coverage."""
    nside = 4
    lmax = 8
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=np.float64)

    # Create fields
    temp_field = create_field(spin=0, nside=nside, lmax=lmax, mask=mask, labels="T")
    pol_field = create_field(spin=2, nside=nside, lmax=lmax, mask=mask, labels=["Q", "U"])

    # Test FieldCollection properties (covers lines 786, 791)
    from cosmocore import InputParams
    from cosmocore.fields import FieldCollection

    # Create minimal params for FieldCollection
    params = InputParams()
    collection = FieldCollection(params, [temp_field, pol_field])

    # Test lmax property (line 786)
    assert collection.lmax == lmax

    # Test spin property (line 791)
    expected_spins = [0, 2]
    assert collection.spin == expected_spins

    assert collection.nside == nside
