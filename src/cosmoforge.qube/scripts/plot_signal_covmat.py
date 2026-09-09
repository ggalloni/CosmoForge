import os

import healpy as hp
import numpy as np
from matplotlib import pyplot as plt

from cosmocore import (
    FieldCollection,
    InputParams,
    compute_pointings,
    create_field,
    read_mask,
    signal_matrix,
)


def get_signal_covmat(fields, local_path):
    Par = InputParams.read_parameter_file(
        local_path + f"/tests/data/nside8/{fields}/config.yaml"
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

    field_actives = [field.active_pixels for field in fields]
    npixs = [len(active) for active in field_actives]

    point_vectors = tuple(np.empty((n, 3), dtype=np.float64) for n in npixs)
    theta_vectors = tuple(np.empty(n, dtype=np.float64) for n in npixs)
    phi_vectors = tuple(np.empty(n, dtype=np.float64) for n in npixs)

    point_vectors, _, _ = compute_pointings(
        Par.nside,
        npixs,
        point_vectors,
        theta_vectors,
        phi_vectors,
        field_actives,
        Par.ordering,
    )

    collection.set_pointing_vectors(point_vectors)

    collection.set_cls()
    collection.set_beams()

    return signal_matrix(collection, Par.lmax)


def plot_signal_covmat_TQU(local_path, show_fig=False, save_fig=False):
    Par = InputParams.read_parameter_file(
        local_path + "/tests/data/nside8/TQU/config.yaml"
    )
    npix = hp.nside2npix(Par.nside)

    file = local_path + "/tests/data/nside8/TQU/ref_signal.bin"
    ref = np.fromfile(file, dtype=np.float64).reshape((npix * 3, npix * 3))

    ref = np.asfortranarray(ref, dtype=np.float64)

    signal_matrix = get_signal_covmat("TQU", local_path=local_path)

    diff = np.abs(signal_matrix - ref)

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix T")

    plt.figure(figsize=(8, 6))
    plt.imshow(signal_matrix[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix T")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices T")
    if save_fig:
        plt.savefig("T_covmat_comparison.png")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[npix:, npix:], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix QU")

    plt.figure(figsize=(8, 6))
    plt.imshow(signal_matrix[npix:, npix:], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix QU")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[npix:, npix:], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices QU")
    if save_fig:
        plt.savefig("QU_covmat_comparison.png")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[:npix, npix:], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix T-QU")

    plt.figure(figsize=(8, 6))
    plt.imshow(signal_matrix[:npix, npix:], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix T-QU")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[:npix, npix:], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices T-QU")
    if save_fig:
        plt.savefig("T_QU_covmat_comparison.png")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix QU-T")

    plt.figure(figsize=(8, 6))
    plt.imshow(signal_matrix[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix QU-T")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices QU-T")
    if save_fig:
        plt.savefig("QU_T_covmat_comparison.png")
    if show_fig:
        plt.show()


def plot_signal_covmat_TEB(local_path, show_fig=False, save_fig=False):
    Par = InputParams.read_parameter_file(
        local_path + "/tests/data/nside8/TEB/config.yaml"
    )
    npix = hp.nside2npix(Par.nside)

    file = local_path + "/tests/data/nside8/TEB/ref_signal.bin"
    ref = np.fromfile(file, dtype=np.float64).reshape((npix * 3, npix * 3))

    ref = np.asfortranarray(ref, dtype=np.float64)

    signal_matrix = get_signal_covmat("TEB", local_path=local_path)

    diff = np.abs(signal_matrix - ref)

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix T")

    plt.figure(figsize=(8, 6))
    plt.imshow(signal_matrix[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix T")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[:npix, :npix], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices T")
    if save_fig:
        plt.savefig("T_covmat_comparison_TEB.png")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[npix : 2 * npix, npix : 2 * npix], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix E")

    plt.figure(figsize=(8, 6))
    plt.imshow(signal_matrix[npix : 2 * npix, npix : 2 * npix], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix E")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[npix : 2 * npix, npix : 2 * npix], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices E")
    if save_fig:
        plt.savefig("E_covmat_comparison_TEB.png")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[-npix:, -npix:], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix B")

    plt.figure(figsize=(8, 6))
    plt.imshow(signal_matrix[-npix:, -npix:], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix B")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[-npix:, -npix:], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices B")
    if save_fig:
        plt.savefig("B_covmat_comparison_TEB.png")
    if show_fig:
        plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(ref[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Test Covariance Matrix EB-T")

    plt.figure(figsize=(8, 6))
    plt.imshow(signal_matrix[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Signal Covariance Matrix EB-T")

    plt.figure(figsize=(8, 6))
    plt.imshow(diff[npix:, :npix], origin="lower")
    plt.colorbar()
    plt.title("Difference between Signal Matrices EB-T")
    if save_fig:
        plt.savefig("EB_T_covmat_comparison_TEB.png")
    if show_fig:
        plt.show()


if __name__ == "__main__":
    path = os.path.abspath(__file__.split("/scripts/plot_signal_covmat.py")[0])

    plot_signal_covmat_TQU(path, show_fig=True, save_fig=False)
    plot_signal_covmat_TEB(path, show_fig=True, save_fig=False)

    print("All plots done.")
