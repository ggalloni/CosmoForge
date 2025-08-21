import os
import sys

import healpy as hp
import numpy as np

from cosmocore import InputParams


compute_case = sys.argv[1]
outpath = sys.argv[2]

path = os.path.abspath(__file__.split("/scripts/produce_mock_inputs.py")[0])
Par = InputParams.read_parameter_file(f"{path}/quelo/{compute_case}_defaults.yaml")

nside = Par.nside
npix = hp.nside2npix(nside)

# =========================
# Mask Production
# =========================

N_masks = Par.nfields
print("Number of masks:", N_masks)

mask = np.ones((N_masks, npix))

cut_sky = True if "--mask" in sys.argv else False
if cut_sky:
    print("Cutting sky...")
    gal_cut = np.radians(30)
    mask[:, hp.query_strip(nside, np.pi / 2 - gal_cut, np.pi / 2 + gal_cut)] = 0.0
hp.write_map(f"{outpath}/{compute_case}_mask.fits", mask, overwrite=True)

# =========================
# Matrices Production
# =========================

if compute_case == "T":
    sigma_pix = 100 / 180 / 60 * np.pi / hp.nside2resol(Par.nside)
    Mat = np.diag(np.ones(npix * 1)) * sigma_pix**2
elif compute_case == "QU":
    sigma_pix = 100 / 180 / 60 * np.pi / hp.nside2resol(Par.nside)
    Mat = np.diag(np.ones(npix * 2)) * sigma_pix**2
else:  # TQU or TEB
    sigma_pix_T = 100 / 180 / 60 * np.pi / hp.nside2resol(Par.nside)
    sigma_pix_P = 10 / 180 / 60 * np.pi / hp.nside2resol(Par.nside)
    Mat = np.diag(np.ones(npix * 3))
    Mat[:npix, :npix] *= sigma_pix_T**2  # T
    Mat[npix:, npix:] *= sigma_pix_P**2  # E - B


Mat.tofile(Par.covmatfile1)
Mat.tofile(Par.covmatfile2)

# =========================
# Spectra Production
# =========================

clfid = np.loadtxt(f"{outpath}/dls.txt")

new_clfid = np.zeros((clfid.shape[0], 1 + Par.nspectra))

print(Par.nspectra)
print(Par.nfields)
print("new_clfid shape:", new_clfid.shape)

if compute_case == "T":
    new_clfid[:, 0] = clfid[:, 0]
    new_clfid[:, 1] = clfid[:, 1]
elif compute_case == "QU":
    new_clfid[:, 0] = clfid[:, 0]
    new_clfid[:, 1] = clfid[:, 2]
    new_clfid[:, 2] = clfid[:, 3]
    new_clfid[:, 3] = clfid[:, 4] * 0.0
elif compute_case == "TQU" or compute_case == "TEB":
    new_clfid[:, 0] = clfid[:, 0]
    new_clfid[:, 1] = clfid[:, 1]
    new_clfid[:, 2] = clfid[:, 2]
    new_clfid[:, 3] = clfid[:, 3]
    new_clfid[:, 4] = clfid[:, 4]
    new_clfid[:, 5] = clfid[:, 4] * 0.0
    new_clfid[:, 6] = clfid[:, 4] * 0.0

print(new_clfid.shape)
print(new_clfid[:5, 0])
print(new_clfid[:5, 1])

if compute_case == "T":
    np.savetxt(f"{outpath}/{compute_case}_dls.txt", new_clfid, header="ell TT")
elif compute_case == "QU":
    np.savetxt(
        f"{outpath}/{compute_case}_dls.txt",
        new_clfid,
        header="ell EE BB EB",
    )
elif compute_case == "TQU" or compute_case == "TEB":
    np.savetxt(
        f"{outpath}/{compute_case}_dls.txt",
        new_clfid,
        header="ell TT EE BB TE TB EB",
    )
