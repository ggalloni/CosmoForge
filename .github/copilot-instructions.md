# GitHub Copilot Instructions for CosmoForge

## Project Overview

CosmoForge is a comprehensive Python framework for cosmological analysis, organized as a monorepo with three main packages:

- **cosmocore**: Core mathematical and computational utilities for cosmological calculations
- **quelo**: Quadratic Maximum Likelihood (QML) estimation tools for power spectra analysis
- **meta**: Project metadata, configuration management, and shared utilities

## Code Style and Standards

### Python Code Guidelines
- Follow PEP 8 standards with 90-character line limit (Ruff formatting)
- Use type hints for all function parameters and return values
- Prefer descriptive variable names over abbreviations
- Use docstrings in NumPy/SciPy style for all public functions and classes
- Import order: standard library, third-party, local imports (separated by blank lines)
- Use double quotes for strings (as configured in Ruff)

### Documentation Standards
- All public functions must have comprehensive docstrings with Parameters, Returns, and Examples sections
- Use RST format for Sphinx documentation
- Include mathematical formulas using LaTeX notation when relevant
- Provide usage examples for complex functions

### Testing Requirements
- Maintain minimum 90% test coverage across all packages
- Write unit tests for all public functions
- Include integration tests for complex workflows
- Use pytest with descriptive test names following `test_<functionality>_<scenario>` pattern
- Mock external dependencies and file I/O operations

## Architecture Patterns

### Core Principles
- **Separation of Concerns**: Keep mathematical operations, I/O, and UI separate
- **Modular Design**: Each module should have a single, well-defined responsibility
- **Performance Focus**: Use NumPy vectorization, numba compilation where appropriate
- **Scientific Rigor**: Include proper error handling and validation for scientific computations

### Package Structure
```
src/
├── cosmoforge.cosmocore/     # Core mathematical utilities
│   ├── cosmocore/
│   │   ├── basics.py         # Basic mathematical operations
│   │   ├── core.py           # Core computational routines
│   │   ├── fields.py         # Field management and operations
│   │   ├── harmonic.py       # Harmonic analysis utilities
│   │   ├── in_out.py         # Input/output operations
│   │   ├── pixel.py          # Pixel-space operations
│   │   ├── settings.py       # Configuration and settings
│   │   └── ...
├── cosmoforge.quelo/         # QML estimation tools
├── cosmoforge.meta/          # Meta-package and workflows
```

### Design Patterns to Follow
- **Factory Pattern**: For creating field objects with different configurations
- **Strategy Pattern**: For different analysis methods (QML, Fisher matrix, etc.)
- **Observer Pattern**: For progress tracking in long-running computations
- **Builder Pattern**: For complex parameter configurations

## Domain-Specific Knowledge

### Cosmology and CMB Analysis
- **Power Spectra**: TT, EE, BB, TE, TB correlations for temperature and polarization
- **Harmonic Analysis**: Spherical harmonics, multipole moments (ℓ values)
- **HEALPix**: Hierarchical Equal Area isoLatitude Pixelization scheme
- **Fisher Matrix**: Parameter estimation and forecasting
- **QML Estimation**: Quadratic Maximum Likelihood for power spectrum estimation

### Key Scientific Concepts
- **Spin-weighted fields**: Scalar (spin-0) temperature, tensor (spin-2) polarization
- **Masking**: Handling incomplete sky coverage
- **Beam effects**: Instrumental response corrections
- **Cross-correlations**: Between different fields, frequencies, or surveys

### Mathematical Conventions
- Use `ell` for multipole moments, not `l` (avoid confusion with 1)
- Use `nside` for HEALPix resolution parameter
- Use `lmax` for maximum multipole moment
- Follow cosmology naming: `Cl` for power spectra, `alm` for harmonic coefficients

## Code Review Focus Areas

### Performance and Efficiency
- Suggest NumPy vectorization over Python loops
- Recommend numba compilation for computationally intensive functions
- Identify opportunities for memory optimization in large array operations
- Check for proper use of in-place operations where appropriate

### Scientific Accuracy
- Verify correct handling of edge cases in mathematical operations
- Check for proper normalization in power spectrum calculations
- Ensure correct coordinate system conversions
- Validate error propagation in statistical calculations

### Testing and Coverage
- Suggest additional test cases for edge conditions
- Recommend integration tests for complex workflows
- Identify untested code paths, especially in error handling
- Suggest property-based testing for mathematical functions

### Documentation Quality
- Ensure all scientific functions have clear mathematical descriptions
- Check for proper citation of algorithms and methods
- Verify that examples run correctly and are pedagogically useful
- Suggest improvements to docstring clarity and completeness

## Common Anti-Patterns to Avoid

### Performance Issues
- Avoid Python loops over large arrays (suggest NumPy alternatives)
- Don't recreate expensive objects unnecessarily
- Avoid memory leaks in long-running computations
- Don't use `append()` in loops for large datasets

### Scientific Computing Issues
- Never ignore numerical precision issues
- Avoid hardcoded magic numbers (suggest named constants)
- Don't mix units without clear documentation
- Avoid silent failures in scientific calculations

### Code Organization Issues
- Don't mix different levels of abstraction in single functions
- Avoid circular imports between modules
- Don't put business logic in utility functions
- Avoid overly complex inheritance hierarchies

## Dependencies and Tools

### Core Scientific Stack
- **NumPy**: For numerical computations
- **SciPy**: For advanced mathematical functions
- **HealPy**: For HEALPix operations
- **numba**: For performance-critical code compilation

### Development Tools
- **pytest**: Testing framework with coverage reporting
- **ruff**: Linting and formatting (replaces Black + flake8 + isort)
- **uv**: Package management and virtual environments
- **codecov**: Coverage reporting and enforcement

### Documentation
- **Sphinx**: Documentation generation
- **MyST**: Markdown support in Sphinx
- **matplotlib**: For plots in documentation

## Review Workflow Integration

### CI/CD Considerations
- All suggestions should be compatible with the existing test pipeline
- Consider impact on build times when suggesting changes
- Ensure suggestions align with codecov requirements (90% coverage)
- Respect the monorepo structure and change detection system

### Pull Request Guidelines
- Focus on the specific packages affected by changes
- Consider backward compatibility for public APIs
- Suggest appropriate test additions for new functionality
- Recommend documentation updates for user-facing changes

## Special Instructions

### When Reviewing Mathematical Code
- Pay special attention to array shapes and broadcasting rules
- Verify numerical stability of algorithms
- Check for proper handling of special values (NaN, infinity)
- Ensure correct statistical interpretations

### When Reviewing Scientific Workflows
- Verify that the scientific methodology is sound
- Check for proper error handling and user feedback
- Ensure reproducibility through proper random seeding
- Validate input parameter ranges and constraints

### When Suggesting Improvements
- Prioritize correctness over performance optimizations
- Consider the expertise level of likely contributors
- Provide educational context for domain-specific suggestions
- Include references to relevant scientific literature when appropriate

Remember: CosmoForge is a scientific computing project where correctness and reproducibility are paramount. Always prioritize these over style preferences or minor performance gains.
