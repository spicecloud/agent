import configparser
import subprocess
from pathlib import Path

HOME_DIRECTORY = Path.home()

SPICE_BINARY_PATH = Path(HOME_DIRECTORY / ".pyenv" / "shims" / "spice")

SPICE_AGENT_SERVICE = "cloud.spice.agent.service"

SPICE_AGENT_SERVICE_FILEPATH = Path(
    HOME_DIRECTORY / ".config" / "systemd" / "user" / SPICE_AGENT_SERVICE
)


def stop_service():
    try:
        stop_existing_process = f"systemctl --user stop {SPICE_AGENT_SERVICE}"
        subprocess.check_output(stop_existing_process.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("stop_service: ", exception)
        return False


def remove_service():
    try:
        remove_existing_service = f"systemctl --user disable {SPICE_AGENT_SERVICE}"
        subprocess.check_output(remove_existing_service.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("remove_service: ", exception)
        return False


def start_service():
    try:
        start_new_service = f"systemctl --user start {SPICE_AGENT_SERVICE}"
        subprocess.check_output(start_new_service.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("start_service: ", exception)
        return False


def populate_service_file():
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


def verify_service_file():
    systemctl_check = f"systemctl --user show {SPICE_AGENT_SERVICE} -p CanStart --value"
    try:
        output = subprocess.check_output(systemctl_check.split(" ")).decode().strip()
        if output == "no":
            raise Exception(f"{systemctl_check} yielded {output}")
        return True
    except subprocess.CalledProcessError as exception:
        print("verify_service_file: ", exception)
        return False
