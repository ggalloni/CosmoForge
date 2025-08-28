Quick Start Guide
=================

This guide will get you started with CosmoForge's main packages quickly.

CosmoForge Architecture
-----------------------

CosmoForge is organized as a namespace package with three main components:

* **cosmoforge.cosmocore**: Core mathematical and computational utilities
* **cosmoforge.quelo**: QML estimation and Fisher matrix analysis  
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
   labels: ["T", "E", "B"]
   fwhmarcmin: 5.0
   apply_pixwin: true

.. code-block:: python

   # Load configuration
   params = InputParams.read_parameter_file('config.yaml')

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

Getting Started with Quelo
---------------------------

Fisher Matrix Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Example Fisher matrix analysis
   # (Full documentation pending Quelo package completion)
   from cosmoforge.quelo import Fisher
   
   # Initialize Fisher analysis
   fisher = Fisher("config/fisher_config.yaml")
   fisher.run()

QML Power Spectrum Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Example QML estimation
   # (Full documentation pending Quelo package completion)
   from cosmoforge.quelo import QML
   
   # Initialize QML analysis
   qml = QML("config/qml_config.yaml")
   qml.run()

Next Steps
----------

* Read the :doc:`tutorials/index` for detailed examples
* Explore the :doc:`api/cosmocore` reference for CosmoCore functions
* Check the example notebooks in the repository
* See :doc:`api/quelo` and :doc:`api/meta` for other package documentation
