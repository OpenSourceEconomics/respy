import os

import pytest


@pytest.fixture(scope="function", autouse=True)
def fresh_directory(tmp_path):
    """Each test is executed in a fresh directory."""
    os.chdir(tmp_path)
