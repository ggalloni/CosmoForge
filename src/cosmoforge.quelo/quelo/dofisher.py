import time

import healpy as hp
import numpy as np
from cosmocore import (
    InputParams,
    LogicalField,
    LogicalFieldCollection,
    compute_beam,
    compute_pointings,
    compute_signal_matrix,
    do_derivative_step,
    matrix_inverse_symm,
    matrix_mult,
    matrix_trace,
    output_geometry,
    read_covmat,
    read_mask,
    readcl,
    write_covmat_reduced,
    write_out_matrix,
)
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ==============================
# Preliminary setup
# ==============================

if rank == 0:
    Par = InputParams.read_parameter_file("quelo/dofisher/TEB_defaults.yaml")

    npix = hp.nside2npix(Par.nside)
    mask = np.empty((Par.nfields, npix), dtype=np.float64)
    mask = read_mask(Par.maskfile, mask)

    if len(mask.shape) == 1:
        mask = mask[:, np.newaxis]

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
    if Par.feedback > 4:
        print("Logical fields created:")
        for lf in logical_fields:
            print(f" - {lf.maps_label} with spin {lf.spin} and nside {lf.nside}")

    collection = LogicalFieldCollection(logical_fields)

    if Par.feedback > 4:
        print("Spectra labels:", collection.spectra_labels)
        print("Number of spectra:", collection.n_spectra)

    npixs = []
    for lf in logical_fields:
        npixs += lf.N_active

    point_vectors = tuple(
        np.empty((npixs[i], 3), dtype=np.float64) for i in range(len(npixs))
    )

    pixact = collection.get_active_pixels()
    if Par.feedback > 4:
        print("Active pixels:", pixact)
        print("Active pixels shape:", pixact.shape)

    point_vectors = compute_pointings(
        Par.nside, npixs, point_vectors, pixact, Par.ordering
    )
    if Par.feedback > 4:
        print("Pointing vectors shape:", point_vectors[0].shape)
        print("Pointing vectors first 5 elements:")
        print(point_vectors[0][:5])

    collection.set_pointing_vectors(point_vectors)

    output_geometry(Par.output_geometry_file, npixs, point_vectors, pixact)

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
    if Par.feedback > 4:
        print("Noise covariance matrix shape:", NCov1.shape)
        print("Noise covariance matrix first row:", NCov1[0, :10])
        print("Noise covariance matrix second row:", NCov1[1, :10])

    clfid = readcl(Par.inputclfile, Par)

    if Par.feedback > 4:
        print("Read clfid:", clfid)

    collection.set_cls(clfid)

    if Par.feedback > 4:
        print("Set Cls in collection")
        print(collection.cls_dict)

    beam_dict = compute_beam(
        lmax=Par.lmax,
        nside=Par.nside,
        smoothtype=Par.smoothing_type,
        fwhmarcmin=Par.fwhmarcmin,
        beam_file=Par.beam_file,
    )
    if Par.feedback > 4:
        print("Computed beam")
        print("Beam:", beam_dict)

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
    if Par.feedback > 4:
        print("Beams in collection")
        print(collection.beam)

    collection.apply_smoothing()
    if Par.feedback > 4:
        print("Smoothed Cls in collection")
        print(collection.cls_dict)

    Sig = np.zeros_like(NCov1, dtype=np.float64)
    Sig = np.asfortranarray(Sig, dtype=np.float64)

    if Par.feedback > 3:
        start = time.time()
    compute_signal_matrix(
        S=Sig,
        lmax=Par.lmax,
        fields=collection,
    )
    if Par.feedback > 3:
        end = time.time()
        print(f"Signal matrix computed in {end - start:.2f} seconds")
    if Par.feedback > 4:
        print("Signal matrix shape:", Sig.shape)
        print("Signal matrix first row:", Sig[0, :10])
        print("Signal matrix second row:", Sig[1, :10])

    NCov1 = NCov1 + Sig
    if Par.feedback > 4:
        print("Noise + Sig covariance matrix shape:", NCov1.shape)
        print("Noise + Sig covariance matrix first row:", NCov1[0, :10])
        print("Noise + Sig covariance matrix second row:", NCov1[1, :10])

    if not Par.do_cross:
        write_covmat_reduced(Par.outnoisecovmat1, NCov1)

    NCov1 = matrix_inverse_symm(NCov1)
    if Par.feedback > 4:
        print("Inv covariance matrix shape:", NCov1.shape)
        print("Inv covariance matrix first row:", NCov1[0, :10])
        print("Inv covariance matrix second row:", NCov1[1, :10])
    write_covmat_reduced(Par.outinvcovmatfile1, NCov1)

    if Par.do_cross:
        Ncov2 = np.asfortranarray(NCov2)
        NCov2 = matrix_inverse_symm(NCov2)
        write_covmat_reduced(Par.outinvcovmatfile2, NCov2)

comm.Barrier()

# ==============================
# Broadcast shared variables
# ==============================

Par: InputParams = comm.bcast(Par if rank == 0 else None, root=0)
collection: LogicalFieldCollection = comm.bcast(collection if rank == 0 else None, root=0)
npix = hp.nside2npix(Par.nside)

npixs = comm.bcast(npixs if rank == 0 else None, root=0)

pixact = comm.bcast(pixact if rank == 0 else None, root=0)

point_vectors = comm.bcast(point_vectors if rank == 0 else None, root=0)

# nl = comm.bcast(nl if rank == 0 else None, root=0)
NCov1 = comm.bcast(NCov1 if rank == 0 else None, root=0)
Sig = comm.bcast(Sig if rank == 0 else None, root=0)
if Par.do_cross:
    NCov2 = comm.bcast(NCov2 if rank == 0 else None, root=0)

comm.Barrier()

# ==============================
# Compute Fisher matrix
# ==============================

n_ell = Par.lmax - 1
nell = Par.nspectra * n_ell

fisher = np.zeros((nell, nell))
derSil = np.zeros_like(NCov1)
derSjl = np.zeros_like(NCov1)

fisher = np.asfortranarray(fisher)
derSil = np.asfortranarray(derSil)
derSjl = np.asfortranarray(derSjl)

if rank == 0 and Par.feedback > 2:
    print("Starting Computation fisher matrix")
    count_computed = 0
if rank == 0 and Par.feedback > 3:
    t1 = time.time()

ellperproc = np.ceil((nell + 1.0) * nell / 2.0 / size)
if Par.feedback > 2:
    print(f"Rank {rank} will compute {ellperproc} elements")
counter = 0
appil = -1

for il in range(nell):
    spectrum_i = il // n_ell
    curr_ell_i = (il % n_ell) + 2

    for jl in range(il, nell):
        spectrum_j = jl // n_ell
        curr_ell_j = jl % n_ell + 2

        counter += 1
        if not (counter > rank * ellperproc and counter <= (rank + 1) * ellperproc):
            continue

        if rank == 0 and Par.feedback > 2:
            print("-" * 80)
            print(
                f"Rank {rank} ---> "
                f"Spec {collection.spectra_labels[spectrum_i]} l= {curr_ell_i} VS "
                f"Spec {collection.spectra_labels[spectrum_j]} l= {curr_ell_j}"
            )
            count_computed += 1
            print(f"Computed {count_computed} of {int(ellperproc)} elements")

        if il != appil:
            derSil.fill(0.0)
            do_derivative_step(
                derSil, spectrum_i, npixs, Par.spins, curr_ell_i, collection
            )
            if rank == 0 and Par.feedback > 4:
                print("DerSil shape:", derSil.shape)
                print("DerSil first row:", derSil[0, :10])
                print("DerSil second row:", derSil[1, :10])

            if jl == il:
                if Par.do_cross:
                    Sig = matrix_mult(NCov2, matrix_mult(derSil, NCov1))
                else:
                    Sig = matrix_mult(NCov1, matrix_mult(derSil, NCov1))
                fisher[il, il] = 0.5 * matrix_trace(derSil, Sig)
                if rank == 0 and Par.feedback > 4:
                    print(f"Fisher diagonal element [{il}, {il}]: {fisher[il, il]}")

            derSil = matrix_mult(derSil, NCov1)
            if Par.do_cross:
                derSil = matrix_mult(NCov2, derSil)
            else:
                derSil = matrix_mult(NCov1, derSil)
        if jl != il:
            derSjl.fill(0.0)
            do_derivative_step(
                derSjl, spectrum_j, npixs, Par.spins, curr_ell_j, collection
            )
            fisher[il, jl] = 0.5 * matrix_trace(derSjl, derSil)
            if rank == 0 and Par.feedback > 4:
                print(f"Fisher off-diagonal element [{il}, {jl}]: {fisher[il, jl]}")
            fisher[jl, il] = fisher[il, jl]

        appil = il


comm.Barrier()
redfisher = np.zeros_like(fisher)
comm.Reduce(fisher, redfisher, op=MPI.SUM, root=0)

if rank == 0:
    if Par.feedback > 1:
        print("-" * 80)
        print("Fisher matrix computed")
    if Par.feedback > 3:
        t2 = time.time()
        print(f"Computation time: {t2 - t1:.2f} seconds")

    write_out_matrix(Par.outfilefisher, redfisher)

    if Par.feedback > 4:
        print("Fisher matrix written to", Par.outfilefisher)
        print("Fisher matrix shape:", redfisher.shape)
        print("Fisher matrix first row:", redfisher[0, :10])
        print("Fisher matrix second row:", redfisher[1, :10])

comm.Barrier()
MPI.Finalize()
