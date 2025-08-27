import healpy as hp
import numpy as np
from astropy.io import fits

from .settings import InputParams


def read_covmat(covmatfile, npix, nmaps, active, C):
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
    with open(outcovmatfile.strip(), "wb") as f:
        C.tofile(f)


def read_mask(maskfile, mask):
    nmaps, _ = mask.shape
    m = hp.read_map(maskfile.strip(), field=range(nmaps), dtype=np.float64)
    return np.array(m).T  # transpose to (npix, nmaps)


def output_geometry(filegeometry, npixs, point_vectors, active):
    nmaps = len(npixs)

    with open(filegeometry.strip(), "w") as f:
        for field_idx in range(nmaps):
            ntemp = npixs[field_idx]
            f.write(f"{field_idx + 1:6d}{ntemp:6d}{0:6d}{0:6d}\n")
            for i in range(ntemp):
                idx = active[field_idx, i]
                vec = point_vectors[field_idx][i, :]
                f.write(f"{idx:6d}{vec[0]:15.6e}{vec[1]:15.6e}{vec[2]:15.6e}\n")


def readcl(inputclfile, Params: InputParams):
    """
    Reads a text file of Cl's with a header line.
    Returns a dictionary mapping spectrum labels to arrays.
    """
    with open(inputclfile.strip()) as f:
        header = f.readline()
        if not header.lstrip().startswith("#"):
            raise ValueError("First line must be a header starting with '#'")
        labels = header.strip().lstrip("#").split()
        arr = np.loadtxt(f, dtype=np.float64)
        if Params.feedback > 3:
            print("Read Cls:", arr.shape, labels)
        cls_dict = {}
        for i, label in enumerate(labels):
            if label.lower() == "ell":
                continue  # skip the ell column
            cls_dict[label] = arr[: Params.lmax - 1, i]
    return cls_dict


def write_out_matrix(outfilematrix, matrix):
    n = matrix.shape[0]
    with open(outfilematrix, "w") as f:
        for i in range(n):
            line = "".join(f"{matrix[i, j]:15.7E}" for j in range(n))
            f.write(line + "\n")


def read_maps(maps, filename, pixact, calibration: float = 1.0):
    assert maps.shape[0] == sum(len(p) for p in pixact), "maps array has incorrect shape"
    nsims = maps.shape[1]

    with fits.open(filename) as hdul:
        for isim in range(nsims):
            sim_data = hdul[f"SIM_{isim:03d}"].data

            counter = 0
            for field_idx in range(len(pixact)):
                pixels = pixact[field_idx].astype(int)
                if len(pixels) > 0 and (
                    pixels.min() < 0 or pixels.max() >= sim_data.shape[1]
                ):
                    raise ValueError(f"Pixel indices out of bounds for field {field_idx}")
                n_active = pixels.size
                maps[counter : counter + n_active, isim] = sim_data[field_idx][pixels]
                counter += n_active

    maps *= calibration
