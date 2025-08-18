import healpy as hp
import numpy as np
from matplotlib import pyplot as plt

from quelo.fields import LogicalField, LogicalFieldCollection
from quelo.harmonic import compute_beam
from quelo.in_out import read_covmat, read_mask, readcl
from quelo.pixel import compute_pointings, compute_signal_matrix
from quelo.settings import InputParams


def get_signal_covmat(fields):
    Par = InputParams.read_parameter_file(f"quelo/dofisher/{fields}_defaults.yaml")

    npix = hp.nside2npix(Par.nside)
    mask = np.empty((Par.nfields, npix), dtype=np.float64)
    mask = read_mask(Par.maskfile, mask)

    logical_fields: list[LogicalField] = []
    counter = 0
    for spin in Par.spins:
        if spin == 0:
            label = Par.labels[counter]
            counter += 1
        else:
            label = [Par.labels[counter], Par.labels[counter + 1]]
            counter += 2
        logical_fields.append(
            LogicalField(
                spin=spin,
                nside=Par.nside,
                lmax=Par.lmax,
                mask=mask[:, counter - 1],
                maps_label=label,
            )
        )

    collection = LogicalFieldCollection(logical_fields)

    npixs = []
    for lf in logical_fields:
        npixs += lf.N_active

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

    clfid = readcl(Par.inputclfile, Par)

    collection.set_cls(clfid)

    beam_dict = compute_beam(
        lmax=Par.lmax,
        nside=Par.nside,
        smoothtype=Par.smoothing_type,
        fwhmarcmin=Par.fwhmarcmin,
        beam_file=Par.beam_file,
    )

    counter = 0
    for i, lf in enumerate(collection.logical_fields):
        if lf.spin == 0:
            lf.set_beam(beam_dict[lf.maps_label[0]])
            counter += 1
        elif lf.spin == 2:
            beam_array = np.array([beam_dict["E"], beam_dict["B"]], dtype=np.float64)
            lf.set_beam(beam_array.T)
            counter += 2
    collection.beam = collection.get_beam()

    collection.apply_smoothing()

    Sig = np.zeros_like(NCov1, dtype=np.float64)
    Sig = np.asfortranarray(Sig, dtype=np.float64)

    compute_signal_matrix(
        S=Sig,
        lmax=Par.lmax,
        fields=collection,
    )

    return Sig


def test_signal_covmat_TQU(show_fig=False):
    Par = InputParams.read_parameter_file("quelo/dofisher/TQU_defaults.yaml")
    npix = hp.nside2npix(Par.nside)

    file = "inputs/TQU_ref_signal.bin"
    ref = np.fromfile(file, dtype=np.float64).reshape((npix * 3, npix * 3))

    ref = np.asfortranarray(ref, dtype=np.float64)

    Sig = get_signal_covmat("TQU")

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
        diff, 0.0, decimal=7, err_msg="Signal covariance matrix does not match reference."
    )


def test_signal_covmat_TEB(show_fig=False):
    Par = InputParams.read_parameter_file("quelo/dofisher/TEB_defaults.yaml")
    npix = hp.nside2npix(Par.nside)

    file = "inputs/TEB_ref_signal.bin"
    ref = np.fromfile(file, dtype=np.float64).reshape((npix * 3, npix * 3))

    ref = np.asfortranarray(ref, dtype=np.float64)

    Sig = get_signal_covmat("TEB")

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
        diff, 0.0, decimal=7, err_msg="Signal covariance matrix does not match reference."
    )


if __name__ == "__main__":
    test_signal_covmat_TQU(show_fig=True)
    test_signal_covmat_TEB(show_fig=True)
    print("All tests passed successfully.")
