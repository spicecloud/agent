import configparser
from pathlib import Path

HOME_DIRECTORY = Path.home()

SPICE_BINARY_PATH = Path(HOME_DIRECTORY / ".pyenv" / "shims" / "spice")

SPICE_AGENT_SERVICE_LABEL = "cloud.spice.agent"

SPICE_AGENT_SERVICE_FILEPATH = Path(
    HOME_DIRECTORY
    / ".config"
    / "systemd"
    / "user"
    / str(SPICE_AGENT_SERVICE_LABEL + ".service")
)


def populate_service_file():
    # SPICE_LAUNCH_AGENT_LOGS.parent.mkdir(parents=True, exist_ok=True)
    # SPICE_LAUNCH_AGENT_LOGS.touch()

    if SPICE_AGENT_SERVICE_FILEPATH.exists():
        SPICE_AGENT_SERVICE_FILEPATH.unlink()
    SPICE_AGENT_SERVICE_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    SPICE_AGENT_SERVICE_FILEPATH.touch()

    config = configparser.ConfigParser()
    config.optionxform = str  # type: ignore

    config["Unit"] = {
        "Description": "Spice Agent",
        "After": "network.target",
    }

    config["Service"] = {
        "ExecStart": f"{SPICE_BINARY_PATH} whoami",
        "Restart": "always",
    }

    config["Install"] = {
        "WantedBy": "default.target",
    }

    try:
        with open(SPICE_AGENT_SERVICE_FILEPATH, "w") as service_file:
            config.write(service_file)
    except IOError as exception:
        print(f"populate_service_file: {exception}")

    SPICE_AGENT_SERVICE_FILEPATH.chmod(0o644)
