import os
import configparser
import subprocess
from pathlib import Path

HOME_DIRECTORY = Path.home()

SPICE_BINARY_PATH = Path(HOME_DIRECTORY / ".pyenv" / "shims" / "spice")

SPICE_AGENT_SERVICE = "cloud.spice.agent.service"
SPICE_AGENT_LOGS = "cloud.spice.agent.log"
SPICE_AGENT_SERVICE_FILEPATH = Path(
    HOME_DIRECTORY / ".config" / "systemd" / "user" / SPICE_AGENT_SERVICE
)
SPICE_AGENT_LOGS_FILEPATH = Path(HOME_DIRECTORY / ".cache" / "spice" / SPICE_AGENT_LOGS)


def stop_service():
    try:
        if is_service_active():
            stop_existing_service = f"systemctl --user stop {SPICE_AGENT_SERVICE}"
            subprocess.check_output(stop_existing_service.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("stop_service: ", exception)
        return False


def is_service_active():
    try:
        is_service_active = (
            f"systemctl --user show {SPICE_AGENT_SERVICE} -p ActiveState --value"
        )
        output = subprocess.check_output(is_service_active.split(" ")).decode().strip()
        return output == "active"
    except subprocess.CalledProcessError as exception:
        print("is_service_active: ", exception)
        return False


def disable_service():
    try:
        if is_service_enabled():
            disable_existing_service = f"systemctl --user disable {SPICE_AGENT_SERVICE}"
            subprocess.check_output(disable_existing_service.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("disable_service: ", exception)
        return False


def is_service_enabled():
    try:
        is_service_active = f"systemctl --user show {SPICE_AGENT_SERVICE} -p UnitFileState --value"  # noqa
        output = subprocess.check_output(is_service_active.split(" ")).decode().strip()
        return output == "enabled"
    except subprocess.CalledProcessError as exception:
        print("is_service_enabled: ", exception)
        return False


def populate_service_file():
    SPICE_AGENT_LOGS_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    SPICE_AGENT_LOGS_FILEPATH.touch()

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
        "ExecStart": f"{SPICE_BINARY_PATH} worker start",
        "Restart": "always",
        "StandardError": f"append:{SPICE_AGENT_LOGS_FILEPATH}",
        "StandardOutput": f"append:{SPICE_AGENT_LOGS_FILEPATH}",
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
    verify_service_file()


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


def create_stdout_file():
    if not SPICE_AGENT_LOGS_FILEPATH.exists():
        SPICE_AGENT_LOGS_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
        SPICE_AGENT_LOGS_FILEPATH.touch()


def delete_stdout_file():
    if SPICE_AGENT_LOGS_FILEPATH.exists():
        SPICE_AGENT_LOGS_FILEPATH.unlink()


def enable_service():
    try:
        enable_service = f"systemctl --user enable {SPICE_AGENT_SERVICE}"
        subprocess.check_output(enable_service.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("enable_service: ", exception)
        return False


def start_service():
    try:
        start_new_service = f"systemctl --user start {SPICE_AGENT_SERVICE}"
        subprocess.check_output(start_new_service.split(" "))
        return True
    except subprocess.CalledProcessError as exception:
        print("start_service: ", exception)
        return False


def view_service_logs():
    follow_logs = f"tail -f -n +1 {SPICE_AGENT_LOGS_FILEPATH}"
    os.system(follow_logs)


def full_service_install():
    stop_service()
    disable_service()
    populate_service_file()
    create_stdout_file()
    enable_service()
    start_service()


def full_service_uninstall():
    stop_service()
    disable_service()
    if SPICE_AGENT_SERVICE_FILEPATH.exists():
        SPICE_AGENT_SERVICE_FILEPATH.unlink()
    delete_stdout_file()
