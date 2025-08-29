import os
import tempfile

import healpy as hp
import numpy as np

from cosmocore import (
    FieldCollection,
    InputParams,
    compute_pointings,
    compute_signal_matrix,
    create_field,
    read_mask,
)


def test_signal_covmat():
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

        # Use new factory function for type-safe field creation
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

    Cls = np.zeros((Par.lmax - 1, collection.n_spectra + 1), dtype=np.float64)
    Cls[:, 0] = np.arange(2, Par.lmax + 1)  # ell values
    Cls[:, 1] = 1e-4 / np.arange(2, Par.lmax + 1) ** 2  # TT
    Cls[:, 2] = 0.5e-4 / np.arange(2, Par.lmax + 1) ** 2  # EE
    Cls[:, 3] = 0.1e-4 / np.arange(2, Par.lmax + 1) ** 2  # BB
    Cls[:, 4] = 0.3e-4 / np.arange(2, Par.lmax + 1) ** 2  # TE
    Cls[:, 5] = 0.0  # TB
    Cls[:, 6] = 0.0  # EB

    with tempfile.NamedTemporaryFile(delete=False) as tmp_cls_file:
        Par.inputclfile = tmp_cls_file.name
    np.savetxt(Par.inputclfile, Cls, header="ell TT EE BB TE TB EB")

    collection.set_cls()
    collection.set_beams()

    ntot = collection.total_active_pixels
    signal_covmat = np.zeros((ntot, ntot), dtype=np.float64)
    signal_covmat = np.asfortranarray(signal_covmat, dtype=np.float64)

    compute_signal_matrix(
        S=signal_covmat,
        lmax=Par.lmax,
        fields=collection,
    )

    os.remove(mock_config_dict["maskfile"])

    ref = np.loadtxt("tests/data/ref_signal.dat")
    np.testing.assert_allclose(signal_covmat, ref, rtol=1e-12)
