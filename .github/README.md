# CosmoForge CI/CD Workflow

This directory contains the automated testing and CI/CD workflows for the CosmoForge project.

## Overview

The CosmoForge project uses intelligent change detection to run tests only for the packages that have been modified, making CI/CD faster and more efficient.

## Workflows

### 1. Main Test Workflow (`.github/workflows/test.yml`)

This is the comprehensive testing workflow that:
- Detects which packages have changed
- Runs appropriate tests only for affected packages
- Includes integration tests when needed
- Provides detailed test coverage

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main`, `master`, or `develop` branches
- Manual trigger via GitHub Actions UI

**Jobs:**
- **detect-changes**: Analyzes git diff to determine affected packages
- **test-cosmocore**: Tests the cosmocore package (if affected)
- **test-quelo**: Tests the quelo package (if affected)  
- **test-meta**: Tests the meta package (if affected)
- **integration-tests**: Runs cross-package integration tests (if multiple packages affected)
- **summary**: Provides a summary of all test results

### 2. Quick CI Workflow (`.github/workflows/quick-ci.yml`)

A simplified, faster workflow for quick feedback:
- Single job that detects changes and runs tests sequentially
- Faster startup time, good for rapid iteration
- Less detailed reporting but faster feedback

## Local Testing Scripts

### Change Detection Script (`scripts/detect-changes.py`)

Analyzes git changes and determines which packages need testing.

```bash
# Basic usage - compare with master branch
python scripts/detect-changes.py

# Compare with specific branch
python scripts/detect-changes.py develop

# Example output:
🔍 Detecting changes in CosmoForge workspace...
📁 Found 3 changed files:
   src/cosmoforge.cosmocore/cosmocore/core.py
   src/cosmoforge.quelo/quelo/fisher.py
   tests/test_integration.py

📦 Affected packages:
   cosmocore: 1 files
   quelo: 1 files

🧪 Test strategy:
   cosmocore: lint, test, build
   quelo: lint, test, build

# Test Strategy Output (for CI/CD)
AFFECTED_PACKAGES=cosmocore,quelo
TEST_COSMOCORE=lint,test,build
TEST_QUELO=lint,test,build
```

### Test Runner Script (`scripts/test-runner.sh`)

Comprehensive local testing script with multiple options.

```bash
# Detect changes and test affected packages (recommended)
./scripts/test-runner.sh

# Test specific package
./scripts/test-runner.sh cosmocore
./scripts/test-runner.sh quelo lint
./scripts/test-runner.sh meta test

# Test all packages
./scripts/test-runner.sh all

# Test types
./scripts/test-runner.sh cosmocore lint    # Only linting
./scripts/test-runner.sh cosmocore test    # Only tests
./scripts/test-runner.sh cosmocore build   # Only build
./scripts/test-runner.sh cosmocore all     # All test types (default)
```

## Package Dependencies

The workflow understands package dependencies:

```
cosmocore (base package)
├── quelo (depends on cosmocore)
└── meta (depends on cosmocore and quelo)
```

**Change Impact:**
- Changes to `cosmocore` → Test `cosmocore`, `quelo`, and `meta`
- Changes to `quelo` → Test `quelo` and `meta`
- Changes to `meta` → Test `meta` only
- Changes to root files (pyproject.toml, etc.) → Test all packages

## Test Types

Each package can run different types of tests based on what changed:

1. **lint**: Code quality checks using `ruff`
2. **test**: Unit tests using `pytest` 
3. **build**: Package building using `uv build`

**Selection Logic:**
- Documentation-only changes → Only linting
- Source code changes → Linting + testing + building
- Test file changes → Linting + testing
- Root configuration changes → All test types for all packages

## Setting Up Your Environment

### Prerequisites

1. **uv**: Fast Python package manager
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Git**: For change detection
   ```bash
   # Already installed on most systems
   git --version
   ```

### Local Development Workflow

1. **Make your changes** to any package
2. **Run change detection** to see what will be tested:
   ```bash
   python scripts/detect-changes.py
   ```
3. **Run targeted tests**:
   ```bash
   ./scripts/test-runner.sh  # Auto-detect and test changes
   ```
4. **Commit and push** - GitHub Actions will run the same logic

### Manual Testing

```bash
# Test a specific package thoroughly
./scripts/test-runner.sh cosmocore all

# Quick lint check
./scripts/test-runner.sh quelo lint

# Run tests for changed packages only
./scripts/test-runner.sh changes
```

## GitHub Actions Configuration

### Required Secrets

No secrets are required for basic functionality. Optional:

- `CODECOV_TOKEN`: For test coverage reporting (recommended)

### Customization

You can customize the workflow by modifying:

1. **Python version**: Change `PYTHON_VERSION` in workflow files
2. **Test commands**: Modify the test steps in each job
3. **Triggers**: Add/remove branch names or event types
4. **Dependencies**: Update package dependency mapping in `detect-changes.py`

### Workflow Outputs

The workflows provide several outputs for downstream use:

- `affected-packages`: Comma-separated list of changed packages
- `test-cosmocore`, `test-quelo`, `test-meta`: Boolean flags
- `test-all`: Boolean flag for root changes
- `strategy`: Detailed test strategy description

## Troubleshooting

### Common Issues

1. **"No changes detected"**
   - Check git history: `git log --oneline -10`
   - Verify branch comparison: `git diff --name-only master..HEAD`

2. **"Package directory not found"**
   - Ensure package structure matches expected paths
   - Check `pyproject.toml` exists in package directories

3. **"uv command not found"**
   - Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Restart shell or run: `source ~/.bashrc`

4. **Tests fail locally but pass in CI**
   - Check Python version consistency
   - Verify all dependencies are properly specified
   - Check for environment-specific configurations

### Debug Mode

Run scripts with verbose output:

```bash
# Enable detailed git output
GIT_TRACE=1 python scripts/detect-changes.py

# Enable bash debug mode
bash -x scripts/test-runner.sh cosmocore
```

## Contributing

When adding new packages or modifying the workflow:

1. Update `detect-changes.py` with new package paths
2. Add corresponding test job in `test.yml`
3. Update dependency mappings if needed
4. Test locally with `test-runner.sh`
5. Update this documentation

## Performance

**Typical CI Times:**
- Change detection: ~10-30 seconds
- Single package test (cosmocore): ~2-5 minutes  
- All packages: ~5-15 minutes
- Full workflow (with changes to all packages): ~10-20 minutes

The intelligent change detection can reduce CI time by 60-80% for typical development workflows.
