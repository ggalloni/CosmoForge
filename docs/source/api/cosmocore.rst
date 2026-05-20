CosmoCore Package
=================

CosmoCore is the foundational package of CosmoForge, providing core functionality for 
cosmological analysis including field management, matrix operations, I/O utilities, 
and fundamental mathematical operations.

Overview
--------

CosmoCore serves as the base layer for all cosmological computations in CosmoForge. It provides:

* **Field Management**: Scalar (spin-0) and tensor (spin-2) field handling with HEALPix integration
* **Matrix Operations**: Optimized linear algebra operations with Numba acceleration  
* **I/O Utilities**: Reading and writing of cosmological data formats
* **Harmonic Analysis**: Power spectrum management and beam handling
* **Pixel Operations**: HEALPix pixel-based computations

Package Structure
-----------------

The CosmoCore package contains the following modules, each documented with comprehensive
mathematical foundations and implementation details:

Core Modules
------------

.. toctree::
   :maxdepth: 1

   cosmocore/basics
   cosmocore/basis
   cosmocore/beam
   cosmocore/bins
   cosmocore/conventions_cmb
   cosmocore/core
   cosmocore/fields
   cosmocore/in_out
   cosmocore/mpi_utils
   cosmocore/settings
   cosmocore/signal_kernels
   cosmocore/spectra_io
   cosmocore/spectrum_key
