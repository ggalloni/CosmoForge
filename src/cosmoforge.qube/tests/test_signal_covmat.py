import os

import healpy as hp
import numpy as np
import pytest

from cosmocore import (
    FieldCollection,
    InputParams,
    active_pixel_index,
    compute_pointings,
    compute_signal_matrix,
    create_field,
    read_covmat,
    read_mask,
)


def get_signal_covmat(fields, config_resolver, local_path):
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
    Par = InputParams.read_parameter_file(config_file)

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

    field_actives = [field.active_pixels for field in fields]
    npixs = [len(active) for active in field_actives]

    point_vectors = tuple(np.empty((n, 3), dtype=np.float64) for n in npixs)
    theta_vectors = tuple(np.empty(n, dtype=np.float64) for n in npixs)
    phi_vectors = tuple(np.empty(n, dtype=np.float64) for n in npixs)

    point_vectors, theta_vectors, phi_vectors = compute_pointings(
        Par.nside,
        npixs,
        point_vectors,
        theta_vectors,
        phi_vectors,
        field_actives,
        Par.ordering,
    )

    collection.set_pointing_vectors(point_vectors)

    concatenate_pixact = active_pixel_index(mask)
    noise_cov1 = np.empty(
        (concatenate_pixact.shape[0], concatenate_pixact.shape[0]), dtype=np.float64
    )
    if Par.do_cross:
        noise_cov2 = np.empty(
            (concatenate_pixact.shape[0], concatenate_pixact.shape[0]), dtype=np.float64
        )

    noise_cov1 = read_covmat(
        Par.covmatfile1, npix, Par.nfields, concatenate_pixact, noise_cov1
    )
    if Par.do_cross:
        noise_cov2 = read_covmat(
            Par.covmatfile2, npix, Par.nfields, concatenate_pixact, noise_cov2
        )

    collection.set_cls()
    collection.set_beams()

    signal_matrix = np.zeros_like(noise_cov1, dtype=np.float64)
    signal_matrix = np.asfortranarray(signal_matrix, dtype=np.float64)

    compute_signal_matrix(
        S=signal_matrix,
        lmax=Par.lmax,
        fields=collection,
    )

    # Clean up temporary config file
    os.unlink(config_file)

    return signal_matrix


@pytest.mark.parametrize("fields", ["T", "QU", "TQU", "TEB"])
def test_signal_covmat(local_path, fields, config_resolver):
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
    Par = InputParams.read_parameter_file(config_file)

    file = local_path + f"/tests/data/nside4/{fields}/ref_signal.bin"
    ref = np.fromfile(file, dtype=np.float64).reshape(
        (Par.npix * Par.nfields, Par.npix * Par.nfields)
    )

    signal_matrix = get_signal_covmat(fields, config_resolver, local_path)

    # Clean up temporary config file
    os.unlink(config_file)

    np.testing.assert_almost_equal(
        signal_matrix,
        ref,
        decimal=5,
        err_msg="TQU Signal covariance matrix does not match reference.",
    )


if __name__ == "__main__":
    fields_list = ["T", "QU", "TQU", "TEB"]

    path = os.path.abspath(__file__.split("/tests/test_signal_covmat.py")[0])

    print(f"Running tests in directory {path} for fields: {fields_list}")

    for fields in fields_list:
        test_signal_covmat(path, fields=fields)

    print("All tests passed successfully.")
