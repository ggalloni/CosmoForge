.. CosmoForge documentation master file, created by
   sphinx-quickstart on Thu Aug 28 15:19:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CosmoForge Documentation
========================

**CosmoForge** is a comprehensive Python framework for power spectrum estimation and
likelihood analysis of spin-0 and spin-2 fields on the sphere, using Fisher matrix,
Quadratic Maximum Likelihood (QML), and pixel-based likelihood methods. While widely
applicable to any sky signal (e.g. CMB, galaxy surveys, 21 cm), it is particularly
optimized for the analysis of partial-sky, noisy observations with complex noise covariance.

Architecture
------------

CosmoForge is a uv workspace of three runtime packages plus an umbrella metapackage:

* **CosmoCore** (``cosmocore``): shared algebra, fields, computation bases, I/O.
* **QUBE** (``qube``, PyPI name ``qube-qml``): Fisher and QML power-spectrum estimation.
* **PICSLike** (``picslike``): Pixel-based Inference with Correlated-Skies Likelihood — pixel-space likelihood analysis.
* **CosmoForge** (``cosmoforge``): umbrella metapackage; one-line install for the whole framework.

Key Features
------------

* **Fisher Matrix Analysis**: Fast parameter forecasting and covariance estimation
* **QML Power Spectrum Estimation**: Optimal power spectrum recovery from noisy data
* **Pixel-Based Likelihood**: Direct likelihood evaluation in map pixel space
* **High-Performance Computing**: Numba-optimized functions and MPI parallelization support
* **HEALPix Integration**: Full support for HEALPix pixelization schemes
* **Flexible Field Management**: Support for scalar (spin-0) and tensor (spin-2) fields
* **Instrumental Effects**: Comprehensive beam and noise modeling

Quick Start
-----------

.. code-block:: python

   # Fisher Matrix Analysis
   from qube import Fisher
   fisher = Fisher(params_file="config/fisher_config.yaml")
   fisher.run()

   # Pixel-Based Likelihood
   from picslike import PICSLike
   picslike = PICSLike(params_file="config/pixel_config.yaml")
   picslike.run()

   # Core mathematical utilities
   from cosmocore import InputParams
   params = InputParams()
   print(f"HEALPix resolution: nside={params.nside}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/cosmocore
   api/qube
   api/picslike
   api/cosmoforge

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog

