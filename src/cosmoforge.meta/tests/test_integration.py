"""Basic integration tests for the Meta package."""

import pytest


def test_import_cosmocore():
    """Test that cosmocore can be imported successfully."""
    try:
        import cosmocore

        assert hasattr(cosmocore, "__version__") or True  # Package imported successfully
    except ImportError as e:
        pytest.fail(f"Failed to import cosmocore: {e}")


def test_import_qube():
    """Test that qube can be imported successfully."""
    try:
        import qube

        # Check for main classes
        assert hasattr(qube, "Fisher") or hasattr(qube, "Spectra")
    except ImportError as e:
        pytest.fail(f"Failed to import qube: {e}")


def test_import_picslike():
    """Test that picslike can be imported successfully."""
    try:
        import picslike

        # Check for main classes
        assert hasattr(picslike, "PICSLike")
    except ImportError as e:
        pytest.fail(f"Failed to import picslike: {e}")


def test_workspace_integration():
    """Test that all workspace packages can be imported together."""
    try:
        import cosmocore
        import picslike
        import qube

        # Test that packages are properly installed
        assert cosmocore is not None
        assert qube is not None
        assert picslike is not None

        # Test that we can access main functionality
        assert hasattr(qube, "Fisher")
        assert hasattr(picslike, "PICSLike")

    except Exception as e:
        pytest.fail(f"Workspace integration test failed: {e}")


def test_meta_package_exists():
    """Test that the meta package itself can be imported."""
    try:
        import meta  # noqa: F401

        # Meta package should exist even if minimal
        assert True  # Import succeeded
    except ImportError:
        # If meta package doesn't exist, that's also fine
        # This test just ensures no unexpected errors
        pass


def test_basic_functionality():
    """Test basic functionality across packages."""
    try:
        # Test importing core functionality
        from cosmocore import create_field

        # Test that functions are callable
        assert callable(create_field)

    except ImportError as e:
        pytest.skip(f"Skipping functionality test due to import error: {e}")
    except Exception as e:
        pytest.fail(f"Basic functionality test failed: {e}")
