import os

import pytest


@pytest.fixture
def local_path():
    path = __file__.split("tests/conftest.py")[0]
    return os.path.abspath(path)
