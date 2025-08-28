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

CosmoForge is organized as a namespace package containing three main subpackages:

* **cosmoforge.cosmocore**: Core functionality for cosmological analysis including field management, matrix operations, and mathematical utilities
* **cosmoforge.quelo**: QML and Fisher matrix implementations for power spectrum estimation
* **cosmoforge.meta**: Metadata and utilities package for project-wide configuration

Key Features
------------

* **Fisher Matrix Analysis**: Fast parameter forecasting and covariance estimation
* **QML Power Spectrum Estimation**: Optimal power spectrum recovery from noisy data
* **High-Performance Computing**: Numba-optimized functions and MPI parallelization support
* **HEALPix Integration**: Full support for HEALPix pixelization schemes
* **Flexible Field Management**: Support for scalar (temperature) and tensor (polarization) fields
* **Instrumental Effects**: Comprehensive beam and noise modeling

Quick Start
-----------

.. code-block:: python

   # Fisher Matrix Analysis
   from cosmoforge.quelo import Fisher
   fisher = Fisher("config/fisher_config.yaml")
   
   # Core mathematical utilities
   from cosmoforge.cosmocore.cosmocore.settings import InputParams
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
   api/meta

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog

