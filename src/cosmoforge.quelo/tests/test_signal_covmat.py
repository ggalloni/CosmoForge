import os

import healpy as hp
import numpy as np
from cosmocore import (
    FieldCollection,
    InputParams,
    compute_pointings,
    compute_signal_matrix,
    create_field,
    read_covmat,
    read_mask,
)
from matplotlib import pyplot as plt


def get_signal_covmat(fields, local_path):
    Par = InputParams.read_parameter_file(
        local_path + f"/tests/data/nside8/{fields}_nside8.yaml"
    )

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

    concatenate_pixact = np.concatenate(
        [pixact[i] + i * npix for i in range(len(pixact))]
    )
    NCov1 = np.empty(
        (concatenate_pixact.shape[0], concatenate_pixact.shape[0]), dtype=np.float64
    )
    if Par.do_cross:
        NCov2 = np.empty(
            (concatenate_pixact.shape[0], concatenate_pixact.shape[0]), dtype=np.float64
        )

    NCov1 = (
        read_covmat(Par.covmatfile1, npix, Par.nfields, concatenate_pixact, NCov1)
        * Par.calibration**2
    )
    if Par.do_cross:
        NCov2 = (
            read_covmat(Par.covmatfile2, npix, Par.nfields, concatenate_pixact, NCov2)
            * Par.calibration**2
        )

    collection.set_cls()
    collection.set_beams()

    Sig = np.zeros_like(NCov1, dtype=np.float64)
    Sig = np.asfortranarray(Sig, dtype=np.float64)

    compute_signal_matrix(
        S=Sig,
        lmax=Par.lmax,
        fields=collection,
    )

    return Sig


def test_signal_covmat_TQU(local_path, show_fig=False):
    Par = InputParams.read_parameter_file(
        local_path + "/tests/data/nside8/TQU_nside8.yaml"
    )
    npix = hp.nside2npix(Par.nside)

    file = local_path + "/tests/data/nside8/TQU_ref_signal.bin"
    ref = np.fromfile(file, dtype=np.float64).reshape((npix * 3, npix * 3))

    ref = np.asfortranarray(ref, dtype=np.float64)

    Sig = get_signal_covmat("TQU", local_path=local_path)

    diff = np.abs(Sig - ref)

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix T")

    plt.figure(figsize=(8, 6))
    plt.imshow(Sig[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix T")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices T")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[npix:, npix:], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix QU")

    plt.figure(figsize=(8, 6))
    plt.imshow(Sig[npix:, npix:], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix QU")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[npix:, npix:], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices QU")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[:npix, npix:], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix T-QU")

    plt.figure(figsize=(8, 6))
    plt.imshow(Sig[:npix, npix:], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix T-QU")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[:npix, npix:], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices T-QU")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix QU-T")

    plt.figure(figsize=(8, 6))
    plt.imshow(Sig[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix QU-T")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices QU-T")
    if show_fig:
        plt.show()

    np.testing.assert_almost_equal(
        diff,
        0.0,
        decimal=7,
        err_msg="TQU Signal covariance matrix does not match reference.",
    )


def test_signal_covmat_TEB(local_path, show_fig=False):
    Par = InputParams.read_parameter_file(
        local_path + "/tests/data/nside8/TEB_nside8.yaml"
    )
    npix = hp.nside2npix(Par.nside)

    file = local_path + "/tests/data/nside8/TEB_ref_signal.bin"
    ref = np.fromfile(file, dtype=np.float64).reshape((npix * 3, npix * 3))

    ref = np.asfortranarray(ref, dtype=np.float64)

    Sig = get_signal_covmat("TEB", local_path=local_path)

    diff = np.abs(Sig - ref)

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix T")

    plt.figure(figsize=(8, 6))
    plt.imshow(Sig[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix T")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices T")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[npix : 2 * npix, npix : 2 * npix], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix E")

    plt.figure(figsize=(8, 6))
    plt.imshow(Sig[npix : 2 * npix, npix : 2 * npix], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix E")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[npix : 2 * npix, npix : 2 * npix], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices E")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[-npix:, -npix:], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix B")

    plt.figure(figsize=(8, 6))
    plt.imshow(Sig[-npix:, -npix:], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix B")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[-npix:, -npix:], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices B")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix EB-T")

    plt.figure(figsize=(8, 6))
    plt.imshow(Sig[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix EB-T")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices EB-T")
    if show_fig:
        plt.show()

    np.testing.assert_almost_equal(
        diff,
        0.0,
        decimal=7,
        err_msg="TEB Signal covariance matrix does not match reference.",
    )


if __name__ == "__main__":
    path = os.path.abspath(__file__.split("/tests/test_signal_covmat.py")[0])

    print(f"Running tests in directory: {path}")

    test_signal_covmat_TQU(path, show_fig=True)
    test_signal_covmat_TEB(path, show_fig=True)

    print("All tests passed successfully.")
