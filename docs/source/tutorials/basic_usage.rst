Basic Usage Tutorial
====================

This tutorial covers the fundamental concepts and basic usage patterns of CosmoForge.

Overview
--------

CosmoForge is designed around three main packages:

1. **CosmoCore**: Provides the mathematical foundation
2. **Quelo**: Implements analysis algorithms  
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
   labels: ["T", "E", "B"]
   spins: [0, 2]
   
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
