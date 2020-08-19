import os

import pytest


@pytest.fixture(autouse=True)
def _fresh_directory(tmp_path):
    """Each test is executed in a fresh directory."""
    os.chdir(tmp_path)
