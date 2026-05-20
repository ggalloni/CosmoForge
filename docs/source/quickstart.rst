Quick Start Guide
=================

This guide will get you started with CosmoForge's main packages quickly.

CosmoForge Architecture
-----------------------

CosmoForge is a uv workspace of three runtime packages plus an umbrella
metapackage:

* **CosmoCore** (``cosmocore``) — shared algebra, fields, computation bases, I/O.
* **QUBE** (``qube``, PyPI name ``qube-qml``) — Fisher and QML estimation.
* **PICSLike** (``picslike``) — pixel-space likelihood.
* **CosmoForge** (``cosmoforge``) — umbrella metapackage; depends on the three above.

Getting Started with CosmoCore
-------------------------------

Parameter Management
^^^^^^^^^^^^^^^^^^^^

CosmoCore uses a centralized parameter system:

.. code-block:: python

   from cosmocore.settings import InputParams
   
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
   input_convention: Dl              # "Cl" (default) or "Dl" for input files
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

   from cosmocore.basics import legendre_00, legendre_22, legendre_02

   x = 0.5     # cos(theta)
   lmax = 10

   P_l_TT = legendre_00(x, lmax)   # spin-0
   P_l_EE = legendre_22(x, lmax)   # spin-2 auto
   P_l_TE = legendre_02(x, lmax)   # spin-0 x spin-2 cross

Getting Started with QUBE
-------------------------

Fisher Matrix Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qube import Fisher

   # Initialize Fisher analysis
   fisher = Fisher("config/fisher_config.yaml")
   fisher.run()

   if fisher.rank == 0:
       errors = fisher.get_error_bars()

QML Power Spectrum Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qube import Spectra

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
* Explore the :doc:`api/cosmocore` reference for CosmoCore modules
* See :doc:`api/qube` and :doc:`api/picslike` for the other packages
* Check the example notebooks in the repository
