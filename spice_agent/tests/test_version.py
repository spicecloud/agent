# tests/test_version.py

import os
import toml
from spice_agent import __version__


def test_version():
    # Get the path to the pyproject.toml file
    project_dir = os.path.dirname(os.path.abspath("agent"))
    toml_file = os.path.join(project_dir, "pyproject.toml")

    # Read the version from pyproject.toml
    with open(toml_file) as f:
        config = toml.load(f)
        expected_version = config["tool"]["poetry"]["version"]

    # Ensure package version is the same as project version
    assert __version__.__version__ == expected_version
