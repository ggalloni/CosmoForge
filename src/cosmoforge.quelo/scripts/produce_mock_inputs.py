"""
Mock input data generation for CosmoForge QML analysis.

This script generates synthetic CMB observations including maps, masks, noise
covariance matrices, and power spectra for testing and validation of the
CosmoForge QML analysis pipeline. It produces realistic mock data that mimics
real CMB observations while maintaining full control over the input parameters.

The script generates:
1. Sky masks with optional Galactic cut
2. Noise covariance matrices for different field configurations
3. Fiducial power spectra in the required format
4. Monte Carlo CMB map simulations with beam convolution and noise

Usage:
    python produce_mock_inputs.py <compute_case> <output_path> <config_file> [--mask]

Arguments:
    compute_case : str
        Analysis type - one of "T" (temperature only), "QU" (polarization),
        "TQU" (temperature + polarization), or "TEB" (temperature + E/B modes)
    output_path : str
        Base directory where mock data will be written
    config_file : str
        Path to YAML parameter file containing analysis configuration
    --mask : optional
        If present, applies a Galactic plane cut to the sky mask

Examples:
    python produce_mock_inputs.py T /data/mocks config.yaml
    python produce_mock_inputs.py TQU /data/mocks config.yaml --mask

References:
    .. [1] Gorski, K.M. et al. "HEALPix: A Framework for High-Resolution
       Discretization and Fast Analysis of Data Distributed on the Sphere"
       Astrophys. J. 622, 759 (2005)
"""

"""
Mock input data generation for CosmoForge QML analysis.

This script generates synthetic CMB observations including maps, masks, noise
covariance matrices, and power spectra for testing and validation of the
CosmoForge QML analysis pipeline. It produces realistic mock data that mimics
real CMB observations while maintaining full control over the input parameters.

The script generates:
1. Sky masks with optional Galactic cut
2. Noise covariance matrices for different field configurations
3. Fiducial power spectra in the required format
4. Monte Carlo CMB map simulations with beam convolution and noise

Usage:
    python produce_mock_inputs.py <compute_case> <output_path> <config_file> [--mask]

Arguments:
    compute_case : str
        Analysis type - one of "T" (temperature only), "QU" (polarization),
        "TQU" (temperature + polarization), or "TEB" (temperature + E/B modes)
    output_path : str
        Base directory where mock data will be written
    config_file : str
        Path to YAML parameter file containing analysis configuration
    --mask : optional
        If present, applies a Galactic plane cut to the sky mask

Examples:
    python produce_mock_inputs.py T /data/mocks config.yaml
    python produce_mock_inputs.py TQU /data/mocks config.yaml --mask

References:
    .. [1] Gorski, K.M. et al. "HEALPix: A Framework for High-Resolution
       Discretization and Fast Analysis of Data Distributed on the Sphere"
       Astrophys. J. 622, 759 (2005)
"""

import os
import sys

import healpy as hp
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from cosmocore import InputParams

# Parse command line arguments
compute_case = sys.argv[1]  # Analysis case: T, QU, TQU, or TEB
outpath = sys.argv[2]  # Output directory path
config_file = sys.argv[3]  # Parameter configuration file

# Load analysis parameters from configuration file
path = os.path.abspath(__file__.split("/scripts/produce_mock_inputs.py")[0])
Par = InputParams.read_parameter_file(config_file)

# Set up HEALPix pixelization parameters
nside = Par.nside  # HEALPix resolution parameter
npix = hp.nside2npix(nside)  # Total number of pixels in full sky

print(f"Generating mock inputs for case: {compute_case}")
print(f"Output directory: {outpath}")
print(f"HEALPix nside: {nside} (npix: {npix})")
print(f"Number of fields: {Par.nfields}")
print(f"Number of simulations: {Par.nsims}")
print("-" * 50)

# =========================
# Sky Mask Generation
# =========================

N_masks = Par.nfields
print(f"Generating {N_masks} sky masks...")

# Initialize masks for all fields (default: full sky observation)
mask = np.ones((N_masks, npix))

# Apply Galactic plane cut if requested
cut_sky = True if "--mask" in sys.argv else False
if cut_sky:
    print("Applying Galactic plane cut (±30°)...")
    gal_cut = np.radians(30)  # 30-degree cut around Galactic plane
    # Query pixels in Galactic plane strip and set to zero
    galactic_pixels = hp.query_strip(nside, np.pi / 2 - gal_cut, np.pi / 2 + gal_cut)
    mask[:, galactic_pixels] = 0.0
    n_masked = len(galactic_pixels)
    percent_masked = n_masked / npix * 100
    print(f"Masked {n_masked} pixels ({percent_masked:.1f}% of sky)")
else:
    print("Using full sky (no masking applied)")

# Write mask to FITS file
hp.write_map(f"{outpath}/{compute_case}/mask.fits", mask, overwrite=True)
print(f"Mask saved to: {outpath}/{compute_case}/mask.fits")

# =========================
# Noise Covariance Matrix Generation
# =========================

print("\nGenerating noise covariance matrices...")

# Define instrumental noise levels (in microkelvin-arcmin)
# These values represent typical CMB experiment sensitivities
sigma_temp_uK_arcmin = 100  # Temperature noise: 100 μK⋅arcmin
sigma_pol_uK_arcmin = 10  # Polarization noise: 10 μK⋅arcmin

# Convert noise levels to per-pixel standard deviations
# Factor accounts for unit conversion and pixel size
pixel_area_factor = 1 / 180 / 60 * np.pi / hp.nside2resol(Par.nside)
sigma_pix_T = sigma_temp_uK_arcmin * pixel_area_factor
sigma_pix_P = sigma_pol_uK_arcmin * pixel_area_factor

print(f"Temperature noise: {sigma_temp_uK_arcmin} μK⋅arcmin")
print(f"Polarization noise: {sigma_pol_uK_arcmin} μK⋅arcmin")
print(f"Per-pixel T noise: {sigma_pix_T:.2e}")
print(f"Per-pixel P noise: {sigma_pix_P:.2e}")

# Create noise covariance matrix based on analysis case
if compute_case == "T":
    # Temperature-only analysis: N_pix x N_pix diagonal matrix
    print("Creating temperature-only noise covariance matrix")
    Mat = np.diag(np.ones(npix * 1)) * sigma_pix_T**2
elif compute_case == "QU":
    # Polarization-only analysis: 2*N_pix x 2*N_pix diagonal matrix
    print("Creating polarization-only noise covariance matrix")
    Mat = np.diag(np.ones(npix * 2)) * sigma_pix_P**2
else:  # TQU or TEB cases
    # Combined analysis: 3*N_pix x 3*N_pix block-diagonal matrix
    print(f"Creating combined {compute_case} noise covariance matrix")
    Mat = np.diag(np.ones(npix * 3))
    Mat[:npix, :npix] *= sigma_pix_T**2  # Temperature block
    Mat[npix:, npix:] *= sigma_pix_P**2  # Polarization blocks (Q,U or E,B)

print(f"Noise covariance matrix shape: {Mat.shape}")
print(f"Matrix memory size: {Mat.nbytes / 1024**2:.1f} MB")

# Write noise covariance matrices to binary files
# These will be used by the QML analysis pipeline
Mat.tofile(Par.covmatfile1)  # Primary dataset covariance
Mat.tofile(Par.covmatfile2)  # Secondary dataset covariance (for cross-correlation)
print("Noise matrices saved to:")
print(f"  Primary: {Par.covmatfile1}")
print(f"  Secondary: {Par.covmatfile2}")

# =========================
# Fiducial Power Spectra Setup
# =========================

print("\nSetting up fiducial power spectra...")

# Load reference power spectra from input file
# These represent the theoretical CMB power spectra to be used for simulations
path = os.path.abspath(__file__.split("/produce_mock_inputs.py")[0])
clfid = np.loadtxt(f"{path}/dls.txt")
print(f"Loaded reference spectra from: {path}/dls.txt")
print(f"Input spectra shape: {clfid.shape}")

# Create extended power spectra array with all cross-correlation terms
# Format: [ell, TT, EE, BB, TE, TB, EB]
new_clfid = np.zeros((clfid.shape[0], 1 + 6))

print(f"Number of fields: {Par.nfields}")
print(f"Extended spectra shape: {new_clfid.shape}")

# Populate the extended spectra array
# Copy existing spectra and set additional cross-correlations
new_clfid[:, 0] = clfid[:, 0]  # Multipole ell
new_clfid[:, 1] = clfid[:, 1]  # TT power spectrum
new_clfid[:, 2] = clfid[:, 2]  # EE power spectrum
new_clfid[:, 3] = clfid[:, 3]  # BB power spectrum
new_clfid[:, 4] = clfid[:, 4]  # TE cross-correlation
new_clfid[:, 5] = clfid[:, 4] * 0.0  # TB cross-correlation (set to zero)
new_clfid[:, 6] = clfid[:, 4] * 0.0  # EB cross-correlation (set to zero)

print(f"Sample multipoles: {new_clfid[:5, 0]}")
print(f"Sample TT values: {new_clfid[:5, 1]}")

# Save D_l format spectra (for reference and plotting)
np.savetxt(
    f"{outpath}/{compute_case}/dls.txt",
    new_clfid,
    header="ell TT EE BB TE TB EB",
)
print(f"D_l spectra saved to: {outpath}/{compute_case}/dls.txt")

# Convert from D_l to C_l format for HEALPix simulation
# D_l = l(l+1)/(2π) * C_l, so C_l = D_l * (2π)/[l(l+1)]
ell = new_clfid[:, 0]
todl = ell * (ell + 1) / (2 * np.pi)

# Avoid division by zero at ell=0
todl[0] = 1.0 if todl[0] == 0 else todl[0]

new_clfid[:, 1:] = new_clfid[:, 1:] / todl[:, None]

# Add monopole and dipole terms (set to zero for CMB)
# HEALPix synalm expects C_l arrays starting from ell=0
new_clfid = np.insert(new_clfid, 0, 0.0, axis=0)  # ell=0 (monopole)
new_clfid = np.insert(new_clfid, 1, 0.0, axis=0)  # ell=1 (dipole)

print(f"Final C_l array shape for simulation: {new_clfid.shape}")
print("Power spectra conversion: D_l → C_l completed")

# =========================
# Monte Carlo Map Simulation
# =========================

print("\nStarting Monte Carlo map simulation...")

# Initialize noise arrays for T, Q, U components
noise = np.zeros((3, npix))

# Load instrumental beam transfer function
# Beam accounts for instrumental smoothing and pixel window functions
path = os.path.abspath(__file__.split("/produce_mock_inputs.py")[0])
beam = hp.read_cl(f"{path}/beam_440TP_pixwin16.fits")
print(f"Loaded beam function from: {path}/beam_440TP_pixwin16.fits")
print(f"Beam array shape: {beam.shape}")

# Create output directory for map simulations
os.makedirs(f"{outpath}/maps", exist_ok=True)

# Set up simulation parameters
nsims = Par.nsims
print(f"Generating {nsims} Monte Carlo simulations...")

# Initialize FITS HDU lists for both coordinate representations
tqu_hdus = [fits.PrimaryHDU()]  # T,Q,U (Stokes parameters)
teb_hdus = [fits.PrimaryHDU()]  # T,E,B (harmonic space representation)

# Set up reproducible random number generation
# Using modern numpy Generator for better statistical properties
master_seed = 123456789
sequence = np.random.SeedSequence(master_seed)
child_sequence = sequence.spawn(nsims)  # Create independent streams per simulation

# Create separate RNG for each simulation to ensure reproducibility
rngs = [np.random.default_rng(s) for s in child_sequence]

print(f"Using master seed: {master_seed}")
print("Each simulation uses independent, reproducible random streams")
print("Simulation progress:")

for i in tqdm(range(nsims), desc="Producing simulations".center(30)):
    # Generate independent noise realization for each field
    # Using the dedicated RNG for this simulation ensures reproducibility
    noise[0] = rngs[i].normal(0, sigma_pix_T, npix)  # Temperature noise
    noise[1] = rngs[i].normal(0, sigma_pix_P, npix)  # Q polarization noise
    noise[2] = rngs[i].normal(0, sigma_pix_P, npix)  # U polarization noise

    # Generate CMB signal in harmonic space
    # Use separate seed for HEALPix synalm to ensure reproducibility
    sim_seed = rngs[i].integers(0, 2**30)
    np.random.seed(sim_seed)  # Set seed for HEALPix random generation

    # Generate spherical harmonic coefficients from power spectra
    # alm shape: (n_fields, lmax+1, lmax+1) complex array
    alm = hp.synalm(new_clfid[:, 1:].T, lmax=Par.lmax, new=True)

    # Apply instrumental beam convolution in harmonic space
    # This is more efficient than convolving in pixel space
    alm[0] = hp.almxfl(alm[0], beam[0, : Par.lmax + 1])  # Temperature beam
    alm[1] = hp.almxfl(alm[1], beam[1, : Par.lmax + 1])  # E-mode beam
    alm[2] = hp.almxfl(alm[2], beam[2, : Par.lmax + 1])  # B-mode beam

    # Transform to pixel space and add noise
    # TQU maps: Stokes parameters (T, Q, U)
    sim_tqu = hp.alm2map(alm, nside=nside, pol=True) + noise

    # Create FITS header with simulation metadata
    header = fits.Header()
    header["SIM_NUM"] = i  # Simulation index
    header["NSIDE"] = nside  # HEALPix resolution
    header["ORDERING"] = "RING"  # Pixel ordering scheme
    header["COORDSYS"] = "G"  # Galactic coordinates
    header["FIELDS"] = "T,Q,U"  # Field content
    header["SEED"] = sim_seed  # Random seed for this simulation
    header["MASTER"] = master_seed  # Master seed for reproducibility
    header["NOISE_T"] = sigma_temp_uK_arcmin  # Temperature noise level
    header["NOISE_P"] = sigma_pol_uK_arcmin  # Polarization noise level

    # Add TQU simulation to HDU list
    tqu_hdus.append(fits.ImageHDU(data=sim_tqu, header=header, name=f"SIM_{i:03d}"))

    # TEB maps: Temperature + E/B modes (keep pol=True for proper E/B generation)
    sim_teb = hp.alm2map(alm, nside=nside, pol=False) + noise

    # Create header for TEB version
    header_teb = header.copy()
    header_teb["FIELDS"] = "T,E,B"

    # Add TEB simulation to HDU list
    teb_hdus.append(fits.ImageHDU(data=sim_teb, header=header_teb, name=f"SIM_{i:03d}"))

# =========================
# Output File Generation
# =========================

print("\nWriting simulation files...")

# Create FITS HDU lists and write to disk
tqu_hdulist = fits.HDUList(tqu_hdus)
teb_hdulist = fits.HDUList(teb_hdus)

# Write multi-extension FITS files
# Each extension contains one simulation with metadata in header
tqu_filename = f"{outpath}/maps/all_sims_TQU.fits"
teb_filename = f"{outpath}/maps/all_sims_TEB.fits"

tqu_hdulist.writeto(tqu_filename, overwrite=True)
teb_hdulist.writeto(teb_filename, overwrite=True)

print("=" * 60)
print("MOCK DATA GENERATION COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Generated {nsims} Monte Carlo simulations")
print(f"Analysis case: {compute_case}")
print(f"HEALPix resolution: nside={nside}")
print(f"Sky coverage: {'Masked (Galactic cut)' if cut_sky else 'Full sky'}")
noise_info = f"T={sigma_temp_uK_arcmin} μK⋅arcmin, P={sigma_pol_uK_arcmin} μK⋅arcmin"
print(f"Noise levels: {noise_info}")
print()
print("Output files:")
print(f"  Sky mask: {outpath}/{compute_case}/mask.fits")
print(f"  Power spectra: {outpath}/{compute_case}/dls.txt")
print(f"  Noise covariance: {Par.covmatfile1}")
print(f"  TQU simulations: {tqu_filename}")
print(f"  TEB simulations: {teb_filename}")
print()
print("Files are ready for CosmoForge QML analysis!")
