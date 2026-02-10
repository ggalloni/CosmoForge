Quick Start Guide
=================

This guide will get you started with CosmoForge's main packages quickly.

CosmoForge Architecture
-----------------------

CosmoForge is organized as a namespace package with three main components:

* **cosmoforge.cosmocore**: Core mathematical and computational utilities
* **cosmoforge.qube**: QML estimation and Fisher matrix analysis
* **cosmoforge.meta**: Metadata and project-wide utilities

Getting Started with CosmoCore
-------------------------------

Parameter Management
^^^^^^^^^^^^^^^^^^^^

CosmoCore uses a centralized parameter system:

.. code-block:: python

   from cosmoforge.cosmocore.cosmocore.settings import InputParams
   
   # Create with defaults
   params = InputParams()
   
   # Check key parameters
   print(f"HEALPix nside: {params.nside}")
   print(f"Maximum l: {params.lmax}")
   print(f"Field labels: {params.labels}")
   print(f"Number of pixels: {params.npix}")

Configuration Files
^^^^^^^^^^^^^^^^^^^

Use YAML files for configuration:

.. code-block:: yaml

   # config.yaml
   nside: 32
   lmax: 128
   labels: ["T", "E", "B"]          # Analysis field labels
   physical_labels: ["T", "Q", "U"] # Map field labels (or "TQU", "T1_T2")
   fwhmarcmin: 5.0
   apply_pixwin: true

.. code-block:: python

   # Load configuration
   params = InputParams.read_parameter_file('config.yaml')

Field Expansion
^^^^^^^^^^^^^^^

CosmoForge supports automatic field label expansion:

.. code-block:: yaml

   # Concatenated single-character fields
   physical_labels: ["QU"]      # Expands to ["Q", "U"]
   labels: ["TEB"]              # Expands to ["T", "E", "B"]
   
   # Underscore-separated multi-character fields  
   labels: ["T1_T2"]            # Expands to ["T1", "T2"]
   physical_labels: ["LOW_HIGH"] # Expands to ["LOW", "HIGH"]
   
   # Mixed configurations
   labels: ["T", "QU", "E1_E2"]  # Expands to ["T", "Q", "U", "E1", "E2"]

Mathematical Utilities
^^^^^^^^^^^^^^^^^^^^^^

Access optimized mathematical functions:

.. code-block:: python

   from cosmoforge.cosmocore.cosmocore.basics import legendre_00, scalar_prod
   import numpy as np
   
   # Compute Legendre polynomials
   x = 0.5
   lmax = 10
   legendre_values = legendre_00(x, lmax)
   
   # Vector operations
   vec1 = np.array([1.0, 0.0, 0.0])
   vec2 = np.array([0.0, 1.0, 0.0])
   dot_product = scalar_prod(vec1, vec2)

Getting Started with QUBE
-------------------------

Fisher Matrix Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cosmoforge.qube import Fisher

   # Initialize Fisher analysis
   fisher = Fisher("config/fisher_config.yaml")
   fisher.run()

   if fisher.rank == 0:
       errors = fisher.get_error_bars()

QML Power Spectrum Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cosmoforge.qube import Spectra

   # Initialize QML analysis
   spectra = Spectra("config/qml_config.yaml")
   spectra.run()

   if spectra.rank == 0:
       # Default: deconvolved estimates of true C_ell
       cl = spectra.get_power_spectra()
       # Other modes: "decorrelated", "convolved"
       cl_decorr = spectra.get_power_spectra(mode="decorrelated")

Next Steps
----------

* Read the :doc:`tutorials/index` for detailed examples
* Explore the :doc:`api/cosmocore` reference for CosmoCore functions
* Check the example notebooks in the repository
* See :doc:`api/qube` and :doc:`api/meta` for other package documentation
