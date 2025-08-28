import os
import sys

import healpy as hp
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from cosmocore import InputParams

compute_case = sys.argv[1]

outpath = sys.argv[2]

path = os.path.abspath(__file__.split("/scripts/produce_mock_inputs.py")[0])
Par = InputParams.read_parameter_file(sys.argv[3])

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
hp.write_map(f"{outpath}/{compute_case}/mask.fits", mask, overwrite=True)

# =========================
# Matrices Production
# =========================

sigma_pix_T = 100 / 180 / 60 * np.pi / hp.nside2resol(Par.nside)
sigma_pix_P = 10 / 180 / 60 * np.pi / hp.nside2resol(Par.nside)
if compute_case == "T":
    Mat = np.diag(np.ones(npix * 1)) * sigma_pix_T**2
elif compute_case == "QU":
    Mat = np.diag(np.ones(npix * 2)) * sigma_pix_P**2
else:  # TQU or TEB
    Mat = np.diag(np.ones(npix * 3))
    Mat[:npix, :npix] *= sigma_pix_T**2  # T
    Mat[npix:, npix:] *= sigma_pix_P**2  # E - B


Mat.tofile(Par.covmatfile1)
Mat.tofile(Par.covmatfile2)

# =========================
# Spectra Production
# =========================

path = os.path.abspath(__file__.split("/produce_mock_inputs.py")[0])
clfid = np.loadtxt(f"{path}/dls.txt")

new_clfid = np.zeros((clfid.shape[0], 1 + 6))

print(Par.nfields)
print("new_clfid shape:", new_clfid.shape)

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

np.savetxt(
    f"{outpath}/{compute_case}/dls.txt",
    new_clfid,
    header="ell TT EE BB TE TB EB",
)


ell = new_clfid[:, 0]
todl = ell * (ell + 1) / (2 * np.pi)

new_clfid[:, 1:] = new_clfid[:, 1:] / todl[:, None]

new_clfid = np.insert(new_clfid, 0, 0.0, axis=0)
new_clfid = np.insert(new_clfid, 1, 0.0, axis=0)

noise = np.zeros((3, npix))

path = os.path.abspath(__file__.split("/produce_mock_inputs.py")[0])
beam = hp.read_cl(f"{path}/beam_440TP_pixwin16.fits")
print("Beam shape:", beam.shape)

os.makedirs(f"{outpath}/maps", exist_ok=True)
nsims = Par.nsims

# Initialize HDU lists for both TQU and TEB
tqu_hdus = [fits.PrimaryHDU()]  # Primary HDU
teb_hdus = [fits.PrimaryHDU()]

# Set up proper seeding for reproducible simulations
master_seed = 123456789
sequence = np.random.SeedSequence(master_seed)
child_sequence = sequence.spawn(nsims)

rngs = [np.random.default_rng(s) for s in child_sequence]

print(f"Using master seed: {master_seed}")
print(f"Generating {nsims} simulations with reproducible seeding...")

for i in tqdm(range(nsims), desc="Producing simulations...".center(30)):
    # Use the seeded RNG for noise generation
    noise[0] = rngs[i].normal(0, sigma_pix_T, npix)
    noise[1] = rngs[i].normal(0, sigma_pix_P, npix)
    noise[2] = rngs[i].normal(0, sigma_pix_P, npix)

    # Generate alm with seed for reproducibility
    sim_seed = rngs[i].integers(0, 2**30)
    np.random.seed(sim_seed)
    alm = hp.synalm(new_clfid[:, 1:].T, lmax=Par.lmax, new=True)

    alm[0] = hp.almxfl(alm[0], beam[0, : Par.lmax + 1])
    alm[1] = hp.almxfl(alm[1], beam[1, : Par.lmax + 1])
    alm[2] = hp.almxfl(alm[2], beam[2, : Par.lmax + 1])

    # TQU maps (polarization=True)
    sim_tqu = hp.alm2map(alm, nside=nside, pol=True) + noise

    # Create header with simulation info
    header = fits.Header()
    header["SIM_NUM"] = i
    header["NSIDE"] = nside
    header["ORDERING"] = "RING"
    header["COORDSYS"] = "G"
    header["FIELDS"] = "T,Q,U"
    header["SEED"] = sim_seed  # Store the seed used for this simulation
    header["MASTER"] = master_seed  # Store the master seed

    # Add as ImageHDU
    tqu_hdus.append(fits.ImageHDU(data=sim_tqu, header=header, name=f"SIM_{i:03d}"))

    sim_teb = hp.alm2map(alm, nside=nside, pol=False) + noise  # Keep pol=True for T,E,B

    header_teb = header.copy()
    header_teb["FIELDS"] = "T,E,B"

    teb_hdus.append(fits.ImageHDU(data=sim_teb, header=header_teb, name=f"SIM_{i:03d}"))

# Write multi-extension FITS files
tqu_hdulist = fits.HDUList(tqu_hdus)
teb_hdulist = fits.HDUList(teb_hdus)

tqu_hdulist.writeto(f"{outpath}/maps/all_sims_TQU.fits", overwrite=True)
teb_hdulist.writeto(f"{outpath}/maps/all_sims_TEB.fits", overwrite=True)

print(f"Saved {nsims} simulations to multi-extension FITS files")
print(f"TQU file: {outpath}/maps/all_sims_TQU.fits")
print(f"TEB file: {outpath}/maps/all_sims_TEB.fits")
