Basic Usage Tutorial
====================

This tutorial covers the fundamental concepts and basic usage patterns of CosmoForge.

Overview
--------

CosmoForge is a uv workspace of three runtime packages plus an umbrella
metapackage:

1. **CosmoCore** (``cosmocore``): shared algebra, fields, computation bases, I/O.
2. **QUBE** (``qube``, PyPI name ``qube-qml``): Fisher and QML estimation.
3. **PICSLike** (``picslike``): pixel-space likelihood.
4. **CosmoForge** (``cosmoforge``): umbrella metapackage; depends on the three above.

Working with CosmoCore
----------------------

Parameter Management
^^^^^^^^^^^^^^^^^^^^

The foundation of any CosmoForge analysis is proper parameter management:

.. code-block:: python

   from cosmocore.settings import InputParams
   
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

   from cosmocore.basics import (
       legendre_00, legendre_22, legendre_02,
   )

   x = 0.7    # cos(theta)
   lmax = 100

   # Temperature (spin-0) case
   P_l = legendre_00(x, lmax)

   # Polarization (spin-2) auto-correlation
   P_l_22 = legendre_22(x, lmax)

   # Temperature-polarization cross-correlation
   P_l_02 = legendre_02(x, lmax)

Spectrum Results and Conventions
--------------------------------

Output power spectra and Fisher inputs are addressed by ``SpectrumKey`` —
a small dataclass that carries the component pair ``(comp_i, comp_j)``
and a ``SpectrumKind`` enum value (the ordered slot pair, e.g.
``GG`` for E×E or ``GC`` for E×B). Once you know your component spins
the same key works as both a list element and a dict key.

Reading bandpower estimates
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Spectra.get_power_spectra()`` returns a flat numpy array by default
(shape ``(n_sims, n_spectra * n_bins)``) for backward compatibility,
with columns ordered as ``TT, EE, BB, EB, TE, TB`` for TQU under
``SymmetryMode.SYMMETRIC``. Pass ``as_dict=True`` for label-keyed
access — usually what you want:

.. code-block:: python

   # Label-keyed dict (recommended for human-readable code)
   cl = spectra.get_power_spectra(mode="deconvolved", as_dict=True)
   ee = cl["EE"]                  # shape (n_sims, n_bins)
   te = cl["TE"]

   # Flat numpy array (back-compat)
   arr = spectra.get_power_spectra(mode="deconvolved")
   ee_arr = arr[:, 10:20]         # bins 0..9 of EE, manual slicing

For Fisher-side outputs, ``Fisher.get_error_bars(as_dict=True)``
returns the same label-keyed shape; ``Fisher.get_bandpower_slices()``
returns ``{label: slice}`` for navigating the flat Fisher matrix; and
``Fisher.get_fisher_block(label_i, label_j)`` extracts a single block::

   slc = fisher.get_bandpower_slices()
   F = fisher.get_fisher_matrix()
   F_tt = F[slc["TT"], slc["TT"]]
   F_tt = fisher.get_fisher_block("TT")          # same thing
   F_te_ee = fisher.get_fisher_block("TE", "EE")

Internally, the canonical key type is :class:`SpectrumKey`, exposing
``(comp_i, comp_j, kind)`` for code that needs to navigate by
component pair. CMB aliases live in ``cosmocore.conventions.cmb``:

.. code-block:: python

   from cosmocore.spectrum_key import SpectrumKey
   from cosmocore.conventions.cmb import EE, TE

   spins = (0, 2)
   key_ee = SpectrumKey(1, 1, EE, spins=spins)   # spin-2 EE
   key_te = SpectrumKey(0, 1, TE, spins=spins)   # spin-0 x spin-2 cross

If your component collection was declared with the spin-2 field first
(e.g. ``spins = (2, 0)``), the raw cross-spectrum is keyed as ``ET`` /
``BT`` rather than ``TE`` / ``TB``. Use ``to_cmb_canonical`` to re-key
a SpectrumKey-keyed dict to the conventional T-first ordering
regardless of declaration order:

.. code-block:: python

   from cosmocore.conventions.cmb import to_cmb_canonical

   keyed_dict = ...   # SpectrumKey-keyed dict
   tfirst = to_cmb_canonical(keyed_dict, spins=spins)
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

``Fisher`` accepts a ``symmetry_mode`` argument; ``Spectra`` inherits
the flag from its ``Fisher`` instance and cannot drift apart:

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

Convolving theory for inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For multi-spectrum likelihoods the ``"convolved"`` mode of
``get_power_spectra`` returns ``(y, W, convolve_theory_func)`` where
``convolve_theory_func`` accepts either a flat ``cl_theory`` vector
ordered like ``get_fisher_matrix`` or a label-keyed dict — the latter
avoids the ordering question entirely:

.. code-block:: python

   y, W, convolve = spectra.get_power_spectra(mode="convolved")
   # Flat form (requires you to know the column order):
   mu = convolve(cl_theory_flat)
   # Dict form (label-keyed, recommended):
   mu = convolve({"TT": tt_binned, "EE": ee_binned, "BB": bb_binned,
                  "EB": eb_binned, "TE": te_binned, "TB": tb_binned})

   # Important: the input values must be in the active output convention.
   # If params.output_convention == "Dl", pass D_ell-binned theory;
   # otherwise C_ell-binned. The window matrix is rotated to match the
   # output convention internally — feeding the wrong convention yields
   # silently-wrong predictions.

The single-spectrum convenience ``Spectra.convolve_theory_for_inference``
(which takes a per-ℓ theory and returns binned bandpowers) is
single-spectrum-only today; for multi-spectrum likelihoods, use the
convolved-mode callable above. Multi-spectrum extension of
``convolve_theory_for_inference`` is tracked as a follow-up.

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

- See the :doc:`../api/cosmocore` reference for the full CosmoCore surface
- See :doc:`../api/qube` for ``Fisher`` and ``Spectra``
- See :doc:`../api/picslike` for ``PICSLike`` and ``ParameterGrid``
