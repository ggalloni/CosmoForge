# CosmoForge.Meta

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-meta-blue.svg)](https://ggalloni.github.io/CosmoForge/api/meta.html)

> **📚 [Meta Documentation](https://ggalloni.github.io/CosmoForge/api/meta.html) | [Main Documentation](https://ggalloni.github.io/CosmoForge/) | [Contributing Guide](https://ggalloni.github.io/CosmoForge/contributing.html)**

Meta is the metadata and utilities package for CosmoForge, providing project-wide configuration, version management, and helper utilities that support the entire CosmoForge ecosystem.

## Overview

The Meta package serves as the organizational hub for CosmoForge, containing:

- **Project Metadata**: Version information, authorship, and project details
- **Configuration Management**: Global configuration utilities and defaults
- **Utility Functions**: Helper functions used across packages
- **Documentation Assets**: Shared documentation resources
- **Build Configuration**: Setup and installation configurations

## Features

### Metadata Management

- **Version Control**: Centralized version management for all packages
- **Author Information**: Contributor and maintainer details
- **License Information**: Project licensing and copyright details
- **Dependencies**: Shared dependency specifications

### Configuration Utilities

- **Global Settings**: Project-wide configuration management
- **Environment Detection**: System and environment information
- **Path Management**: Standardized path handling across packages
- **Logging Configuration**: Centralized logging setup

### Development Tools

- **Build Helpers**: Utilities for package building and distribution
- **Testing Utilities**: Shared testing infrastructure
- **Documentation Tools**: Helpers for documentation generation
- **CI/CD Support**: Continuous integration utilities

## Installation

Meta is automatically installed as part of CosmoForge:

```bash
pip install -e /path/to/CosmoForge
```

## Documentation

For comprehensive project documentation:

- **[Meta API Documentation](https://ggalloni.github.io/CosmoForge/api/meta.html)** - Meta package utilities and configuration
- **[Contributing Guide](https://ggalloni.github.io/CosmoForge/contributing.html)** - How to contribute to CosmoForge
- **[Project Documentation](https://ggalloni.github.io/CosmoForge/)** - Complete CosmoForge documentation
- **[Installation Guide](https://ggalloni.github.io/CosmoForge/installation.html)** - Detailed installation instructions

## Usage

### Version Information

```python
from cosmoforge.meta import __version__, get_version_info

# Get version string
print(f"CosmoForge version: {__version__}")

# Get detailed version information
version_info = get_version_info()
print(version_info)
```

### Configuration Management

```python
from cosmoforge.meta import get_global_config, set_global_option

# Get global configuration
config = get_global_config()

# Set global options
set_global_option('logging_level', 'DEBUG')
set_global_option('parallel_backend', 'mpi')
```

### Path Management

```python
from cosmoforge.meta import get_package_root, get_data_dir

# Get package root directory
root_dir = get_package_root()

# Get data directory
data_dir = get_data_dir()

# Get test data path
test_data = get_data_dir() / "test_data"
```

### Logging Setup

```python
from cosmoforge.meta import setup_logging

# Setup centralized logging
logger = setup_logging(
    level='INFO',
    log_file='cosmoforge.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Package Structure

```text
cosmoforge.meta/
├── __init__.py          # Main package interface
├── version.py           # Version management
├── config.py            # Configuration utilities
├── paths.py             # Path management
├── logging.py           # Logging configuration
├── build_utils.py       # Build and installation helpers
├── testing.py           # Testing utilities
└── docs/               # Documentation assets
    ├── templates/       # Documentation templates
    ├── assets/         # Images and other assets
    └── examples/       # Usage examples
```

## Version Management

### Version String Format

CosmoForge follows semantic versioning (SemVer):

```text
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Examples:
- 1.0.0          (stable release)
- 1.1.0-alpha.1  (pre-release)
- 1.0.1+git.abc123 (development build)
```

### Version API

```python
from cosmoforge.meta.version import (
    __version__,
    version_info,
    is_development,
    get_git_info
)

# Version components
major, minor, patch = version_info[:3]

# Check if development version
if is_development():
    git_info = get_git_info()
    print(f"Development build: {git_info['commit']}")
```

## Configuration System

### Global Configuration

The meta package provides a hierarchical configuration system:

```python
from cosmoforge.meta.config import GlobalConfig

# Access global configuration
config = GlobalConfig()

# Set configuration values
config.set('computation.backend', 'numpy')
config.set('computation.n_threads', 4)
config.set('logging.level', 'INFO')

# Get configuration values
backend = config.get('computation.backend', default='numpy')
n_threads = config.get('computation.n_threads', default=1)
```

### Configuration Files

Configuration can be loaded from files:

```yaml
# ~/.cosmoforge/config.yaml
computation:
  backend: "numba"
  n_threads: 8
  use_mpi: true

logging:
  level: "INFO"
  file: "cosmoforge.log"

paths:
  data_dir: "/path/to/data"
  cache_dir: "/path/to/cache"
```

## Environment Detection

### System Information

```python
from cosmoforge.meta.environment import get_system_info

# Get system information
sys_info = get_system_info()
print(f"Platform: {sys_info['platform']}")
print(f"Python version: {sys_info['python_version']}")
print(f"NumPy version: {sys_info['numpy_version']}")
print(f"MPI available: {sys_info['mpi_available']}")
```

### Resource Detection

```python
from cosmoforge.meta.environment import detect_resources

# Detect available resources
resources = detect_resources()
print(f"CPU cores: {resources['cpu_cores']}")
print(f"Memory: {resources['memory_gb']:.1f} GB")
print(f"MPI processes: {resources['mpi_size']}")
```

## Testing Utilities

### Test Data Management

```python
from cosmoforge.meta.testing import get_test_data, create_mock_data

# Get test data path
test_mask = get_test_data("masks/test_mask_nside32.fits")

# Create mock data for testing
mock_cl = create_mock_data(
    data_type="power_spectrum",
    lmax=100,
    noise_level=0.1
)
```

### Test Configuration

```python
from cosmoforge.meta.testing import TestConfig

# Setup test environment
test_config = TestConfig(
    data_dir="test_data",
    output_dir="test_output",
    cleanup=True
)

with test_config:
    # Run tests with temporary environment
    pass
```

## Build and Distribution

### Build Utilities

```python
from cosmoforge.meta.build_utils import (
    get_build_info,
    check_dependencies,
    compile_extensions
)

# Get build information
build_info = get_build_info()

# Check if all dependencies are available
deps_ok = check_dependencies()

# Compile native extensions
if deps_ok:
    compile_extensions()
```

### Package Information

```python
from cosmoforge.meta import get_package_info

# Get comprehensive package information
pkg_info = get_package_info()
print(f"Name: {pkg_info['name']}")
print(f"Version: {pkg_info['version']}")
print(f"Author: {pkg_info['author']}")
print(f"License: {pkg_info['license']}")
```

### Code Quality

```python
from cosmoforge.meta.dev_tools import run_checks

# Run code quality checks
results = run_checks(
    check_style=True,
    check_types=True,
    check_tests=True
)
```

### Performance Monitoring

```python
from cosmoforge.meta.performance import ProfilerContext

# Profile code execution
with ProfilerContext("analysis_profile"):
    # Run analysis code
    pass

# Get profiling results
results = ProfilerContext.get_results("analysis_profile")
```

## API Reference

### Core Functions

```python
def get_version_info() -> dict:
    """Get detailed version information."""

def get_global_config() -> GlobalConfig:
    """Get global configuration instance."""

def setup_logging(level='INFO', **kwargs) -> logging.Logger:
    """Setup centralized logging."""

def get_package_root() -> Path:
    """Get CosmoForge package root directory."""
```

### Configuration Classes

```python
class GlobalConfig:
    """Global configuration management."""
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        
    def set(self, key: str, value):
        """Set configuration value."""
        
    def load_file(self, filename: str):
        """Load configuration from file."""
```

## Integration Examples

### Package Initialization

```python
# Example: Initialize a CosmoForge package
from cosmoforge.meta import setup_package_environment

def initialize_package():
    # Setup package environment
    env = setup_package_environment(
        package_name="quelo",
        logging_level="INFO"
    )
    
    # Package-specific initialization
    return env
```

### Cross-Package Communication

```python
# Example: Share configuration between packages
from cosmoforge.meta import get_shared_config

def get_analysis_config():
    # Get configuration shared across packages
    config = get_shared_config()
    
    # Extract relevant settings
    return {
        'backend': config.get('computation.backend'),
        'n_threads': config.get('computation.n_threads'),
        'cache_dir': config.get('paths.cache_dir')
    }
```

## Extending Meta

### Custom Configuration

```python
# Add custom configuration sections
from cosmoforge.meta.config import register_config_section

register_config_section('my_analysis', {
    'parameter_1': 'default_value',
    'parameter_2': 42
})
```

### Custom Utilities

```python
# Add custom utilities
from cosmoforge.meta import register_utility

@register_utility('my_helper')
def my_helper_function():
    """Custom helper function."""
    pass
```

## Changelog

See `CHANGELOG.md` for version history and changes.

## Contributing

Meta package contributions should focus on:

- Cross-package utilities
- Configuration management improvements
- Build and deployment enhancements
- Documentation infrastructure

Follow the main CosmoForge contribution guidelines.
