#!/usr/bin/env python3
"""
Detect changes in the CosmoForge workspace and determine which packages need testing.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


def run_git_command(cmd: List[str], cwd: str = None) -> str:
    """Run a git command and return its output."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return ""


def get_changed_files(base_branch: str = "master") -> List[str]:
    """Get list of changed files compared to base branch."""
    repo_root = Path(__file__).parent.parent

    # Primary commands for CI/PR - compare against base branch
    primary_commands = [
        ["git", "diff", "--name-only", f"{base_branch}...HEAD"],
        ["git", "diff", "--name-only", f"{base_branch}..HEAD"],
    ]

    # Fallback commands for local development
    fallback_commands = [
        ["git", "status", "--porcelain"],  # Uncommitted changes
        ["git", "diff", "--name-only", "HEAD~1..HEAD"],  # Last commit only
    ]

    # Try primary commands first (for CI/PR)
    for cmd in primary_commands:
        output = run_git_command(cmd, cwd=repo_root)
        if output:
            # Parse git diff output - filter out empty lines and clean whitespace
            files = []
            for line in output.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    files.append(line)
            if files:
                return files

    # Try fallback commands for local development
    for cmd in fallback_commands:
        output = run_git_command(cmd, cwd=repo_root)
        if output:
            if cmd[1] == "status":
                # Parse porcelain output for uncommitted changes
                files = []
                for line in output.split("\n"):
                    if line.strip():
                        # Remove status indicators and get filename
                        filename = line[3:].strip()
                        if filename:
                            # Handle renamed files (old -> new format)
                            if " -> " in filename:
                                filename = filename.split(" -> ")[-1]
                            files.append(filename)
                return files
            else:
                # Parse git diff output - filter out empty lines and clean whitespace
                files = []
                for line in output.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        files.append(line)
                if files:
                    return files

    return []


def map_files_to_packages(changed_files: List[str]) -> Dict[str, Set[str]]:
    """Map changed files to affected packages."""
    package_mapping = {
        "cosmocore": "src/cosmoforge.cosmocore/",
        "quelo": "src/cosmoforge.quelo/",
        "meta": "src/cosmoforge.meta/",
        "root": ["pyproject.toml", "uv.lock", "README.md", ".github/", "scripts/"],
    }

    affected_packages = {pkg: set() for pkg in package_mapping.keys()}

    for file in changed_files:
        file = file.strip()
        if not file:
            continue

        # Check if file belongs to a specific package
        matched = False
        for package, path_prefix in package_mapping.items():
            if package == "root":
                continue  # Handle root separately
            elif file.startswith(path_prefix):
                affected_packages[package].add(file)
                matched = True
                break

        # If not matched to a specific package, consider it root
        if not matched:
            # Check if it's explicitly a root file/path
            root_files = package_mapping["root"]
            matches_file = any(
                file.endswith(root_file)
                for root_file in root_files
                if not root_file.endswith("/")
            )
            matches_path = any(
                file.startswith(root_path) or file.startswith(root_path.lstrip("."))
                for root_path in root_files
                if root_path.endswith("/")
            )
            # Or if it's in the root directory (no subdirectories)
            is_root_file = (
                "/" not in file
                or file.startswith(".github/")
                or file.startswith("github/")  # Handle missing dot
                or file.startswith("scripts/")
            )

            if matches_file or matches_path or is_root_file:
                affected_packages["root"].add(file)

    # Remove empty sets
    return {pkg: files for pkg, files in affected_packages.items() if files}


def determine_test_strategy(
    affected_packages: Dict[str, Set[str]],
) -> Dict[str, List[str]]:
    """Determine which packages need testing and what kind of tests."""
    test_strategy = {}

    # Package dependencies (packages that depend on others)
    dependencies = {
        "cosmocore": [],  # Base package
        "quelo": ["cosmocore"],  # Depends on cosmocore
        "meta": ["cosmocore", "quelo"],  # Depends on both
    }

    # If root files changed, test everything
    if "root" in affected_packages:
        test_strategy["all"] = ["lint", "test"]
        return test_strategy

    # For each affected package, determine what to test
    packages_to_test = set()

    for package in affected_packages:
        if package != "root":
            packages_to_test.add(package)

            # Add dependent packages
            for dep_pkg, deps in dependencies.items():
                if package in deps:
                    packages_to_test.add(dep_pkg)

    # Determine test types for each package
    for package in packages_to_test:
        test_types = []

        # Always run linting
        test_types.append("lint")

        # Check if source code changed in this package OR if it's a dependent package
        excluded_exts = [".md", ".txt", ".yaml", ".yml"]
        source_files = [
            f
            for f in affected_packages.get(package, [])
            if not any(f.endswith(ext) for ext in excluded_exts)
        ]

        # Check if this package has changes OR if its dependencies have changes
        has_dependency_changes = False
        for affected_pkg in affected_packages:
            if affected_pkg in dependencies.get(package, []):
                # This package depends on an affected package
                has_dependency_changes = True
                break

        if source_files or has_dependency_changes:
            test_types.append("test")

        # Check if test files changed
        test_files = [
            f for f in affected_packages.get(package, []) if "test" in f.lower()
        ]

        if test_files:
            if "test" not in test_types:
                test_types.append("test")

        test_strategy[package] = test_types

    return test_strategy


def main():
    """Main function."""
    if len(sys.argv) > 1:
        base_branch = sys.argv[1]
    else:
        base_branch = "master"

    print("🔍 Detecting changes in CosmoForge workspace...")

    # Get changed files
    changed_files = get_changed_files(base_branch)

    if not changed_files:
        print("✅ No changes detected.")
        return

    print(f"📁 Found {len(changed_files)} changed files:")
    for file in changed_files[:10]:  # Show first 10
        print(f"   {file}")
    if len(changed_files) > 10:
        print(f"   ... and {len(changed_files) - 10} more")

    # Map files to packages
    affected_packages = map_files_to_packages(changed_files)

    print("\n📦 Affected packages:")
    for package, files in affected_packages.items():
        print(f"   {package}: {len(files)} files")

    # Determine test strategy
    test_strategy = determine_test_strategy(affected_packages)

    print("\n🧪 Test strategy:")
    for package, tests in test_strategy.items():
        print(f"   {package}: {', '.join(tests)}")

    # Output for CI/CD consumption
    print("\n# Test Strategy Output (for CI/CD)")
    print(f"AFFECTED_PACKAGES={','.join(affected_packages.keys())}")
    for package, tests in test_strategy.items():
        print(f"TEST_{package.upper()}={','.join(tests)}")


if __name__ == "__main__":
    main()
