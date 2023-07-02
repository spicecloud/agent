from importlib.metadata import version
import logging
import os
from typing import Tuple

from outdated import check_outdated

from spice_agent.daemons.actions import Daemons

LOGGER = logging.getLogger(__name__)


def get_current_version():
    return version("spice_agent")


def get_outdated_and_get_latest_version(current_version: str) -> Tuple[bool, str]:
    if not current_version:
        current_version = get_current_version()
    is_outdated, latest_version = check_outdated("spice_agent", current_version)
    return is_outdated, latest_version


def update_if_outdated():
    current_version = get_current_version()
    is_outdated, latest_version = get_outdated_and_get_latest_version(
        current_version=current_version
    )
    if is_outdated:
        LOGGER.warn(
            f"The current spice_agent version is at {latest_version} \
                {get_current_version()}. Updating..."
        )
        os.system("pip install --upgrade spice_agent")
        daemons = Daemons(spice=None)
        daemons.uninstall()
        daemons.install()
