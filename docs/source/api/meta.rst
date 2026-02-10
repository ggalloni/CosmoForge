CosmoForge.Meta Package
=======================

Meta is the umbrella package for CosmoForge, designed to provide a convenient way to 
install and manage all CosmoForge components. It serves as the main entry point for 
users who want to install the complete CosmoForge framework.

Overview
--------

The Meta package is a meta-package that:

* **Dependency Management**: Automatically installs cosmocore and qube packages
* **Unified Installation**: Provides single-command installation of entire framework
* **Version Coordination**: Ensures compatible versions across all CosmoForge components
* **Project Organization**: Maintains project-wide metadata and documentation

Purpose
-------

Installation Convenience
^^^^^^^^^^^^^^^^^^^^^^^^

Instead of installing packages individually:

.. code-block:: bash

   # Without meta package
   pip install cosmocore
   pip install qube

Users can install everything at once:

.. code-block:: bash

   # With meta package  
   pip install cosmoforge.meta

Dependency Coordination
^^^^^^^^^^^^^^^^^^^^^^^

The meta package ensures that:

* Compatible versions of all components are installed together
* Dependencies between cosmocore and qube are properly resolved
* Future CosmoForge packages are automatically included

Project Structure
-----------------

The meta package coordinates the following CosmoForge components:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Package
     - Description
   * - :doc:`cosmocore`
     - Core mathematical and computational infrastructure
   * - :doc:`qube`
     - Fisher matrix and QML power spectrum estimation

Installation
------------

To install the complete CosmoForge framework:

.. code-block:: bash

   # Install from PyPI (when available)
   pip install cosmoforge.meta
   
   # Install from source
   git clone https://github.com/ggalloni/CosmoForge.git
   cd CosmoForge
   pip install -e .

This will automatically install:

* ``cosmoforge.cosmocore`` - Core functionality
* ``cosmoforge.qube`` - Analysis engines
* All required dependencies (numpy, scipy, healpy, etc.)

Usage
-----

After installation, import and use any CosmoForge component:

.. code-block:: python

   # Import from cosmocore
   from cosmoforge.cosmocore import Core, InputParams

   # Import from qube
   from cosmoforge.qube import Fisher, Spectra

   # Use the full framework
   params = InputParams("config.yaml")
   fisher = Fisher(params)
   fisher.run()

Development
-----------

For developers working on CosmoForge:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/ggalloni/CosmoForge.git
   cd CosmoForge
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -e ".[dev]"

This installs all packages in editable mode, allowing changes to be immediately 
reflected without reinstallation.

Package Information
-------------------

:Version: 0.1.0
:Author: Giacomo Galloni
:License: [License information]
:Repository: https://github.com/ggalloni/CosmoForge

Dependencies
^^^^^^^^^^^^

The meta package automatically installs:

* **cosmoforge.cosmocore**: Core computational framework
* **cosmoforge.qube**: Fisher matrix and QML estimation
* **Python >= 3.11**: Minimum Python version requirement

See Also
--------

* :doc:`cosmocore` : Core mathematical and computational infrastructure
* :doc:`qube` : Fisher matrix and QML power spectrum estimation
* :doc:`../installation` : Detailed installation instructions
* :doc:`../quickstart` : Getting started guide
