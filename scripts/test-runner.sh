#!/bin/bash
# Test runner script for CosmoForge packages
# Usage: ./scripts/test-runner.sh [package_name] [test_type]
#
# package_name: cosmocore, quelo, meta, all
# test_type: lint, test, build, all (default: all)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run tests for a package
test_package() {
    local package=$1
    local test_type=$2
    local package_dir="$ROOT_DIR/src/cosmoforge.$package"
    
    if [ ! -d "$package_dir" ]; then
        print_error "Package directory not found: $package_dir"
        return 1
    fi
    
    print_status "Testing $package (type: $test_type)"
    cd "$package_dir"
    
    # Check if pyproject.toml exists
    if [ ! -f "pyproject.toml" ]; then
        print_warning "No pyproject.toml found in $package_dir"
        return 0
    fi
    
    # Install dependencies
    print_status "Installing dependencies for $package..."
    if ! uv sync --dev; then
        print_error "Failed to install dependencies for $package"
        return 1
    fi
    
    # Run the requested test type
    case $test_type in
        "lint"|"all")
            print_status "Running linting for $package..."
            if uv run ruff check .; then
                print_success "Linting passed for $package"
            else
                print_error "Linting failed for $package"
                return 1
            fi
            
            if uv run ruff format --check .; then
                print_success "Format check passed for $package"
            else
                print_error "Format check failed for $package"
                return 1
            fi
            
            # Only continue if test_type is "all"
            [ "$test_type" != "all" ] && return 0
            ;&
        "test")
            if [ -d "tests" ]; then
                print_status "Running tests for $package..."
                if uv run pytest tests/ -v; then
                    print_success "Tests passed for $package"
                else
                    print_error "Tests failed for $package"
                    return 1
                fi
            else
                print_warning "No tests directory found for $package"
            fi
            
            # Only continue if test_type is "all"
            [ "$test_type" != "all" ] && return 0
            ;&
        "build")
            print_status "Building $package..."
            if uv build; then
                print_success "Build succeeded for $package"
            else
                print_error "Build failed for $package"
                return 1
            fi
            ;;
        *)
            print_error "Unknown test type: $test_type"
            return 1
            ;;
    esac
    
    cd "$ROOT_DIR"
    return 0
}

# Function to detect and test changes
test_changes() {
    local base_branch=${1:-"master"}
    
    print_status "Detecting changes from base branch: $base_branch"
    
    if [ ! -f "$SCRIPT_DIR/detect-changes.py" ]; then
        print_error "Change detection script not found: $SCRIPT_DIR/detect-changes.py"
        return 1
    fi
    
    # Run change detection
    output=$(python "$SCRIPT_DIR/detect-changes.py" "$base_branch")
    echo "$output"
    
    # Parse output and run tests
    if echo "$output" | grep -q "TEST_ALL="; then
        print_status "Root files changed, testing all packages..."
        test_package "cosmocore" "all"
        test_package "quelo" "all" 
        test_package "meta" "all"
    else
        # Test individual packages
        if echo "$output" | grep -q "TEST_COSMOCORE="; then
            test_package "cosmocore" "all"
        fi
        
        if echo "$output" | grep -q "TEST_QUELO="; then
            test_package "quelo" "all"
        fi
        
        if echo "$output" | grep -q "TEST_META="; then
            test_package "meta" "all"
        fi
    fi
}

# Main script logic
main() {
    cd "$ROOT_DIR"
    
    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install it first: https://docs.astral.sh/uv/"
        exit 1
    fi
    
    # Parse arguments
    package=${1:-"changes"}
    test_type=${2:-"all"}
    
    case $package in
        "changes")
            test_changes
            ;;
        "cosmocore"|"quelo"|"meta")
            test_package "$package" "$test_type"
            ;;
        "all")
            print_status "Testing all packages..."
            test_package "cosmocore" "$test_type"
            test_package "quelo" "$test_type"
            test_package "meta" "$test_type"
            ;;
        *)
            echo "Usage: $0 [package_name] [test_type]"
            echo ""
            echo "package_name:"
            echo "  changes    - Detect changes and test affected packages (default)"
            echo "  cosmocore  - Test only cosmocore package"
            echo "  quelo      - Test only quelo package"
            echo "  meta       - Test only meta package"
            echo "  all        - Test all packages"
            echo ""
            echo "test_type:"
            echo "  lint       - Run only linting"
            echo "  test       - Run only tests"
            echo "  build      - Run only build"
            echo "  all        - Run all test types (default)"
            echo ""
            echo "Examples:"
            echo "  $0                    # Detect changes and test affected packages"
            echo "  $0 cosmocore lint     # Run linting for cosmocore only"
            echo "  $0 all test           # Run tests for all packages"
            exit 1
            ;;
    esac
}

main "$@"
