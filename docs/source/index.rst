.. CosmoForge documentation master file, created by
   sphinx-quickstart on Thu Aug 28 15:19:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CosmoForge Documentation
========================

**CosmoForge** is a comprehensive Python framework for cosmological analysis, focusing on 
Cosmic Microwave Background (CMB) data analysis using Fisher matrix and Quadratic Maximum 
Likelihood (QML) power spectrum estimation methods.

Architecture
------------

CosmoForge is organized as a namespace package containing four main subpackages:

* **cosmoforge.cosmocore**: Core functionality for cosmological analysis including field management, matrix operations, and mathematical utilities
* **cosmoforge.quelo**: QML and Fisher matrix implementations for power spectrum estimation
* **cosmoforge.picslike**: Pixel-based likelihood analysis for parameter estimation
* **cosmoforge.meta**: Metadata and utilities package for project-wide configuration

Key Features
------------

* **Fisher Matrix Analysis**: Fast parameter forecasting and covariance estimation
* **QML Power Spectrum Estimation**: Optimal power spectrum recovery from noisy data
* **Pixel-Based Likelihood**: Direct likelihood evaluation in map pixel space
* **High-Performance Computing**: Numba-optimized functions and MPI parallelization support
* **HEALPix Integration**: Full support for HEALPix pixelization schemes
* **Flexible Field Management**: Support for scalar (temperature) and tensor (polarization) fields
* **Instrumental Effects**: Comprehensive beam and noise modeling

Quick Start
-----------

.. code-block:: python

   # Fisher Matrix Analysis
   from quelo import Fisher
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
   api/quelo
   api/picslike
   api/meta

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog

