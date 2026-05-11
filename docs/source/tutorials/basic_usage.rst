Basic Usage Tutorial
====================

This tutorial covers the fundamental concepts and basic usage patterns of CosmoForge.

Overview
--------

CosmoForge is designed around three main packages:

1. **CosmoCore**: Provides the mathematical foundation
2. **QUBE**: Implements analysis algorithms
3. **Meta**: Handles metadata and utilities

Working with CosmoCore
----------------------

Parameter Management
^^^^^^^^^^^^^^^^^^^^

The foundation of any CosmoForge analysis is proper parameter management:

.. code-block:: python

   from cosmoforge.cosmocore.cosmocore.settings import InputParams
   
   # Create default parameters
   params = InputParams()
   
   # Inspect key parameters
   print(f"HEALPix nside: {params.nside}")
   print(f"Maximum multipole: {params.lmax}")
   print(f"Number of pixels: {params.npix}")
   print(f"Field labels: {params.labels}")

Parameter Customization
^^^^^^^^^^^^^^^^^^^^^^^

You can customize parameters in several ways:

.. code-block:: python

   # Method 1: Direct modification
   params.nside = 64
   params.lmax = 192
   params.compute_derived()  # Update derived parameters
   
   # Method 2: Using update method
   config = {
       'nside': 128,
       'lmax': 256,
       'fwhmarcmin': 5.0,
       'apply_pixwin': True
   }
   params.update(config)
   
   # Method 3: From YAML file
   params = InputParams.read_parameter_file('my_config.yaml')

Mathematical Operations
^^^^^^^^^^^^^^^^^^^^^^^

CosmoCore provides optimized mathematical functions:

.. code-block:: python

   from cosmoforge.cosmocore.cosmocore.basics import (
       legendre_00, legendre_22, legendre_02,
       scalar_prod, ext_prod
   )
   import numpy as np
   
   # Legendre polynomials for different spin cases
   x = 0.7  # cos(θ)
   lmax = 100
   
   # Temperature (spin-0) case
   P_l = legendre_00(x, lmax)
   
   # Polarization (spin-2) auto-correlation
   P_l_22 = legendre_22(x, lmax)
   
   # Temperature-polarization cross-correlation
   P_l_02 = legendre_02(x, lmax)
   
   # Vector operations
   v1 = np.array([1.0, 0.0, 0.0])
   v2 = np.array([0.0, 1.0, 0.0])
   
   dot_prod = scalar_prod(v1, v2)    # Dot product
   cross_prod = ext_prod(v1, v2)     # Cross product

Spectrum Results and Conventions
--------------------------------

Output power spectra and Fisher inputs are addressed by ``SpectrumKey`` —
a small dataclass that carries the component pair ``(comp_i, comp_j)``
and a ``SpectrumKind`` enum value (the ordered slot pair, e.g.
``GG`` for E×E or ``GC`` for E×B). Once you know your component spins
the same key works as both a list element and a dict key.

Reading a results dict
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cosmocore.spectrum_key import SpectrumKey, SpectrumKind

   spins = (0, 2)                      # T (spin-0), QU (spin-2)
   spectra = fisher_spectra.get_power_spectra(mode="deconvolved")

   # Iterate
   for key, c_ell in spectra.items():
       print(key.comp_i, key.comp_j, key.kind.name, c_ell[:3])

   # Look up a specific spectrum (e.g. TE between component 0 and 1)
   te = spectra[SpectrumKey(0, 1, SpectrumKind.SG, spins=spins)]

CMB aliases such as ``TT, EE, BB, EB, BE, TE, ET, TB, BT`` live in
``cosmocore.conventions.cmb`` and resolve to the underlying
``SpectrumKind`` (``SS, GG, CC, GC, CG, SG, GS, SC, CS``):

.. code-block:: python

   from cosmocore.conventions.cmb import TE, EE

   ee = spectra[SpectrumKey(1, 1, EE, spins=spins)]   # same as SpectrumKind.GG

If your component collection was declared with the spin-2 field first
(e.g. ``spins = (2, 0)``), the raw cross-spectrum is keyed as ``ET`` /
``BT`` rather than ``TE`` / ``TB``. Use ``to_cmb_canonical`` to re-key
the output dict to the conventional T-first ordering regardless of
declaration order:

.. code-block:: python

   from cosmocore.conventions.cmb import to_cmb_canonical

   spectra_tfirst = to_cmb_canonical(spectra, spins=spins)
   # Now all mixed-spin keys are SG/SC (TE/TB), never GS/CS.

Auto-pair vs cross-pair spin-2 ordering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For two QU components (``spins = (2, 2)``) there is a deliberate
ordering split:

* Same component (``comp_i == comp_j``): the kinds emitted are
  ``GG`` (EE), ``CC`` (BB), and ``GC`` (EB).
* Different components (``comp_i != comp_j``): the kinds are
  ``GG, GC, CG, CC`` — i.e. ``E_i × B_j`` and ``B_i × E_j`` are
  treated as distinct cross-pair entries. Whether both are emitted
  is controlled by ``SymmetryMode`` (next section).

Symmetric vs directional EB handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Fisher`` (and ``Spectra``, which inherits the flag) accept a
``symmetry_mode`` argument:

.. code-block:: python

   from cosmocore.spectrum_key import SymmetryMode
   from qube import Fisher, Spectra

   fisher = Fisher("config.yaml", symmetry_mode=SymmetryMode.DIRECTIONAL)
   spectra = Spectra("config.yaml", fisher=fisher)
   assert spectra.symmetry_mode is SymmetryMode.DIRECTIONAL

* ``SYMMETRIC`` (default) emits a single ``GC`` cross-pair entry per
  spin-2 × spin-2 component pair and uses a single ``C_EB`` in both
  off-diagonal Lambda blocks. For standard cosmology (parity-conserving,
  ``C_EB = 0``) this is numerically identical to ``DIRECTIONAL`` while
  saving one Fisher row/column per cross-pair.
* ``DIRECTIONAL`` emits both ``GC`` (``E_i × B_j``) and ``CG``
  (``B_i × E_j``) as separate spectra and lets Lambda carry independent
  values in each off-diagonal block. Opt-in for polarisation-angle
  calibration diagnostics across frequency pairs, parity-violation
  studies, and any analysis where the asymmetry between EB and BE is a
  signal of interest.

The ``symmetry_mode`` flag lives on ``Fisher``; ``Spectra`` inherits it
from its Fisher instance so the two cannot drift apart. See ADR-0011
(``docs/adr/0011-symmetry-mode-cross-eb.md``) for the full design
rationale.

Configuration Files
-------------------

Example YAML Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a configuration file for your analysis:

.. code-block:: yaml

   # analysis_config.yaml
   
   # HEALPix settings
   nside: 64
   ordering: 1  # RING ordering
   
   # Analysis parameters
   lmax: 192
   labels: ["T", "E", "B"]           # Analysis field labels
   physical_labels: ["T", "Q", "U"]  # Map field labels  
   spins: [0, 2]
   
   # Field expansion examples (alternative formats):
   # physical_labels: ["TQU"]        # Auto-expands to ["T", "Q", "U"] 
   # labels: ["T1_T2"]               # Auto-expands to ["T1", "T2"]
   
   # Instrument parameters
   fwhmarcmin: 5.0
   apply_pixwin: true
   smooth_pol: true
   calibration: 1.0
   
   # Input/output files
   inputclfile: "inputs/cls_theory.dat"
   maskfile: "inputs/analysis_mask.fits"
   beam_file: "inputs/beam_profile.fits"
   
   # Output settings
   feedback: 1
   outfilefisher: "outputs/fisher_matrix.dat"

Loading and Using Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Load configuration
   params = InputParams.read_parameter_file('analysis_config.yaml')
   
   # Verify settings
   print(f"Analysis will use nside={params.nside}, lmax={params.lmax}")
   print(f"Fields: {params.labels}")
   print(f"Total pixels: {params.npix}")
   print(f"Beam FWHM: {params.fwhmarcmin} arcmin")

Best Practices
--------------

1. **Always use configuration files** for reproducible analyses
2. **Check derived parameters** after updates using ``params.compute_derived()``
3. **Use appropriate nside** for your resolution requirements  
4. **Leverage Numba optimizations** by calling functions in loops
5. **Validate parameters** before starting computationally expensive operations

Next Steps
----------

- Learn about :doc:`configuration` for advanced parameter management
- Explore :doc:`mathematical_utilities` for detailed function references
- See :doc:`cmb_analysis` for complete analysis workflows
